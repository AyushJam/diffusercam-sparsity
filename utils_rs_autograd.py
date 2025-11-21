import numpy as np
import torch
import torch.fft as fft
import global_vars as gv
from IPython import display
from tqdm import tqdm
from dataloader import export_gif
import total_variation
import matplotlib.pyplot as plt


def initMatrices(h, shutter, rgb=gv.rgb, large_psf=True, grid_shape=None):
    """
    Inputs:
    - PSF
    - shutter object

    Outputs:
    - PSF FFT matrix H
    - Utility functions
        - crop and pad

    """

    if large_psf:
        if grid_shape is None:
            raise ValueError("grid_shape cannot be None")
        init_shape = grid_shape[1:3] + (1,)
    else:
        # shape of PSF
        init_shape = h.shape

    # random starter for deconvolution gradient descent
    pixel_start = (torch.max(h) + torch.min(h)) / 2
    x = torch.randn([shutter.sf_num] + list(init_shape)) * pixel_start
    # x = torch.randn([1] + list(init_shape)) * pixel_start  # use only one time frame

    # zero padding (for both scene and psf)
    padded_shape = [nextPow2(2 * n - 1) for n in init_shape]

    # Don't pad channel dimension
    if rgb:
        padded_shape[2] = 3
    else:
        padded_shape[2] = 1

    starti = (padded_shape[0] - init_shape[0]) // 2
    endi = starti + init_shape[0]
    startj = (padded_shape[1] // 2) - (init_shape[1] // 2)
    endj = startj + init_shape[1]

    if large_psf:
        H = fft.rfft2(fft.ifftshift(h, dim=(0, 1)), norm="ortho", dim=(0, 1)).to(
            gv.device
        )
    else:
        hpad = torch.zeros(padded_shape)
        hpad[starti:endi, startj:endj] = h
        H = fft.rfft2(fft.ifftshift(hpad, dim=(0, 1)), norm="ortho", dim=(0, 1))

    Hadj = torch.conj(H)

    # Cropping and Padding functions are nested within initMatrices
    # The variables they use are within their "enclosing scope"
    # Hence, the crop function retains access to starti and endi

    def crop(X, vid=True):
        """
        for each frame in video, crop frame and stack results
        """
        # video data is (T, W, H, C)
        # no changes along the T and C dimensions
        # crop along W and H
        if vid:
            return X[:, starti:endi, startj:endj, :]
        else:
            return X[starti:endi, startj:endj, :]

    def pad(v, specify_dim=True, vid=True):
        """ "
        vid = True: For each frame in video, pad frame and stack results
                    Assumes video data is (t, w, h)
        vid = False: pad frame (w, h)
        """
        if vid:
            if specify_dim:
                # Assume number of frames frame is v.shape[0]
                vpad = torch.zeros([v.shape[0]] + padded_shape).type(gv.dtype)
            else:
                # use the number of shutter frames
                vpad = torch.zeros([shutter.sf_num] + padded_shape).type(gv.dtype)

            vpad[:, starti:endi, startj:endj, :] = v[:, :, :, :]
        else:
            vpad = torch.zeros(padded_shape).type(gv.dtype)
            vpad[starti:endi, startj:endj] = v[:, :]

        return vpad

    utils = [crop, pad]

    # If PSF convolution, pad
    if gv.psf_convolve:
        v = torch.real(pad(x))
    else:  # No pad
        v = torch.real(x)

    return H, Hadj, v, utils


def nextPow2(n):
    return int(2 ** np.ceil(np.log2(n)))


def calcA(shutter, H, vk, crop, partition=None):
    """
    Applies rolling shutter foward model to (t,w,h) spatiotemporal data vk

    :param shutter: shutter object containing functions and parameters
    :type shutter: shutter
    :param H: padded fft of the psf.
    :type H: NxN tensor
    :param vk: (t,w,h,c) scene estimate
    :type vk: (t,w,h,c) tensor
    :param crop: cropping function specific to psf dimensions
    :type crop: function
    :param psf_convolve: Toggle for psf convolution to simulate measurement with phase mask (default is True)
    :type psf_convolve: Boolean
    :param partition: timepoints for which we compute the forward model
    :type:


    :return:
    :rtype: (w, h, c) image tensor
    """

    if torch.is_grad_enabled():
        print("calcA tracking gradients")
    else:
        print("calcA NOT tracking gradients")


    # Normalization
    height_H = H.shape[0]
    width_H = 2 * (H.shape[1] - 1)  # H uses an "rfft"
    fft_normalizer = np.sqrt(height_H * width_H)

    # Convolve with PSF and apply shutter function
    Vk = fft.rfft2(
        fft.ifftshift(vk, dim=(1, 2)), norm="ortho", dim=(1, 2)
    )  # Data is (T,H,W,C)

    A_dVk = (
        crop(fft.fftshift(fft.irfft2(H * Vk, norm="ortho", dim=(1, 2)), dim=(1, 2)))
        * fft_normalizer
    )
    # compensate for double ortho normalization that occurs during product
    # make sure multiplication is spatial 2D elementwise for each time point
    # NOTE: H*Vk won't throw a dimension problem, pytorch matches dims backward

    return shutter.shutter_output(A_dVk, partition)
    # return A_dVk[0]  # NO SHUTTER EFFECT


def simple_gaussian_blur(frames, kernel_size=5, sigma=1.0):
    # Create a Gaussian kernel
    x = torch.arange(kernel_size) - (kernel_size - 1) / 2.0
    gauss_kernel = torch.exp(-0.5 * (x**2) / sigma**2)
    gauss_kernel = gauss_kernel / gauss_kernel.sum()

    # Create a 2D Gaussian kernel by outer product
    gauss_kernel_2d = gauss_kernel[:, None] * gauss_kernel[None, :]
    gauss_kernel_2d = gauss_kernel_2d.expand(1, 1, -1, -1).to(frames.device)

    # Use nn.Conv2d to apply the blur
    conv = torch.nn.Conv2d(
        1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
    )
    conv.weight.data = gauss_kernel_2d

    # Apply the convolution to each frame
    frames = frames.unsqueeze(1)  # Add a channel dimension
    blurred_frames = conv(frames).squeeze(
        1
    )  # Apply the blur and remove channel dimension

    return blurred_frames


def non_neg(xi):
    xi = torch.maximum(xi, torch.zeros(xi.size()))
    return xi


def grad_descent(
    h, b, gt, shutter, step_list, niter=100, proj_type="haar", update_method="momentum"
):
    """
    Given the psf h and the captured image b,
    find what the scene looked like.

    Solve for v: Av = b

    Update methods
    - gd: Gradient Descent
    - momentum: Nestorov Momentum
    - fista: FISTA

    Projection Type
    - haar: wavelet denoising
    - non_neg: simple non-negative clipping

    """
    H, Hadj, v0, utils = initMatrices(
        h, shutter, gv.rgb, large_psf=gv.large_psf, grid_shape=shutter.sf.shape
    )
    crop, pad = utils[0], utils[1]

    df_losslist = []
    mse_losslist = []
    tv_losslist = []
    total_losslist = []

    if proj_type == "non_neg":
        proj = non_neg
    elif proj_type == "haar":
        proj = total_variation.tv3dApproxHaar_proj
    else:
        proj = lambda x: x

    # hyperparameters from global vars
    alpha, tau, tau_t = step_list

    # random initialization
    vk = torch.nn.Parameter(v0, requires_grad=True)

    # hyperparameters
    tk = 1
    zk = vk.clone()

    # the optimizer is only for zero grad, we're using FISTA
    optimizer = torch.optim.SGD([vk], lr=alpha)

    # loss function
    loss_fn = torch.nn.MSELoss()

    # update loop
    for itr in tqdm(range(niter), desc="Iterations"):
        # FISTA scheduler
        tkp1 = (1 + np.sqrt(1 + 4 * tk**2)) / 2
        beta = (tk - 1) / tkp1

        # empty the gradients
        optimizer.zero_grad()

        # forward model
        Av = calcA(shutter, H, vk, crop, partition=None)

        # compute loss
        loss = loss_fn(Av, b[0])
        loss.backward()

        with torch.no_grad():
            yk = vk - alpha * vk.grad
            zk_new = non_neg(proj(yk, tau, tau_t))
            v_next = zk_new + beta * (zk_new - zk)

        vk = torch.nn.Parameter(v_next.detach(), requires_grad=True)
        zk = zk_new.detach()

        del Av, loss
        torch.cuda.empty_cache()

        with torch.no_grad():
            # Get scene at this iteration
            vk_log = vk.detach().clone()
            scene_out = crop(vk_log)
            img_out = non_neg(calcA(shutter, H, vk_log, crop).real)

            # loss metrics
            df_loss = torch.log(loss_fn(img_out, b[0].real))  # compare with input image
            mse_loss = loss_fn(scene_out[0], gt[0])  # compare with GT for scene
            tv_loss = total_variation.tv3dApproxHaar_norm(
                scene_out
            )  # TV regularization on estimate
            total_loss = df_loss + tau * tv_loss

            # Detach and store for plotting/logging
            df_losslist.append(df_loss.item())
            mse_losslist.append(mse_loss.item())
            tv_losslist.append(tv_loss.item())
            total_losslist.append(total_loss.item())

            if itr % gv.solver_plot_f == 0:
                scene = scene_out.detach().cpu().numpy()
                img = img_out.detach().cpu().numpy()
                img_gt = b.detach().cpu().numpy()
                fig, ax = plt.subplots(1, 3, figsize=(25, 10))

                ax[0].imshow(scene[0] / np.max(scene[0]))
                ax[0].set_title("Estimated First Frame")
                ax[1].imshow(img / np.max(img))
                ax[1].set_title("Estimated Measured Image")
                ax[2].imshow(np.real(img_gt[0]) / np.max(img_gt[0]))  # first frame
                ax[2].set_title("Measurement")

                plt.suptitle(
                    "Reconstruction after iteration {}".format(itr)
                    + "\n"
                    + "alpha: "
                    + str(alpha)
                    + "\n"
                    + "tau: "
                    + str(tau)
                    + "\n"
                    + "tau_t: "
                    + str(tau_t)
                )
                fig.tight_layout()

                plt.savefig(
                    (gv.solver_output_dir + "slices_itr_{}.png").format(itr)
                )  # recon output figure filename
                plt.close(fig)

                # plt.figure()
                fig, ax = plt.subplots(1, 5, figsize=(25, 15))

                ax[0].plot(df_losslist, color="blue", label="log df_loss")
                ax[0].plot(mse_losslist, color="red", label="mse_loss")
                ax[0].plot(tv_losslist, color="green", label="tv3d_loss")
                ax[0].plot(total_losslist, color="black", label="tv3d+df_loss")
                ax[0].legend()
                ax[1].plot(df_losslist[-3000:], color="blue", label="log df_loss")
                ax[1].legend()

                ax[2].plot(mse_losslist, color="red", label="mse_loss≥–")
                ax[2].legend()

                ax[3].plot(tv_losslist, color="green", label="tv3d_loss")
                ax[3].legend()
                ax[4].plot(total_losslist, color="black", label="tv3d+df_loss")
                ax[4].legend()

                plt.title((gv.solver_plot_dir + "fista_itr {}.png").format(itr))
                fig.tight_layout()
                plt.savefig((gv.solver_plot_dir + "fista_itr.png"))
                plt.close(fig)

                display.clear_output(wait=True)

                print(
                    f"Inner loop {itr}: df_loss {df_loss.detach().cpu().item():.3e}, "
                    f"mse_loss: {mse_loss.detach().cpu().item():.3e}, tv3d_loss: "
                    f"{tv_loss.detach().cpu().item():.3e}, total_loss: {total_loss.detach().cpu().item():.3e}"
                )

                plt.close("all")

    losslist = [df_losslist, mse_losslist, tv_losslist, total_losslist]
    export_gif(
        scene_out.detach().cpu().numpy(), gv.solver_output_dir + "scene", npy=False
    )
    export_gif(
        shutter.sf.detach().cpu().numpy(), gv.solver_output_dir + "sf", npy=False
    )
    # np.save((gv.recon_gif_dir + "sceneout.npy"), scene_out.detach().cpu().numpy())

    return scene_out, losslist
