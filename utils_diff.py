import numpy as np
import torch
import torch.fft as fft
from IPython import display
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    from . import haar
    from . import global_vars as gv
except ImportError:
    import haar
    import global_vars as gv


def initMatrices(h, rgb=gv.rgb, large_psf=True):
    """
    Inputs:
    - PSF

    Outputs:
    - PSF FFT matrix H
    - Utility functions
        - crop and pad

    """

    # shape of PSF
    init_shape = h.shape

    # random starter for deconvolution gradient descent
    pixel_start = (torch.max(h) + torch.min(h)) / 2
    x = torch.randn([1] + list(init_shape)) * pixel_start  # use only one time frame

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

    # print("hpad shape: ", hpad.shape) -> 512, 1024, 3
    # print("H shape: ", H.shape) -> 512, 513, 3

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

    def pad(v, vid=True):
        """
        vid = True: For each frame in video, pad frame and stack results
                    Assumes video data is (t, w, h)
        vid = False: pad frame (w, h)
        """
        if vid:
            vpad = torch.zeros([v.shape[0]] + padded_shape).type(gv.dtype)
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


def calcA_diffuser(H, vk, crop):
    """
    Calculates Av in the diffuser forward model
    """

    # Normalization
    height_H = H.shape[0]
    width_H = 2 * (H.shape[1] - 1)  # H uses an "rfft"
    fft_normalizer = np.sqrt(height_H * width_H)

    # Convolve with PSF
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

    assert A_dVk.ndim == 4 and A_dVk.shape[0] == 1, A_dVk.shape

    return A_dVk


def calcAHerm_diffuser(Hadj, diff, pad):
    """
    Calculates action of A^H on diff = Av - b

    """

    # Normalization
    height_H = Hadj.shape[0]
    width_H = 2 * (Hadj.shape[1] - 1)  # H uses an "rfft"
    fft_normalizer = np.sqrt(height_H * width_H)

    xpad = pad(diff)
    X = fft.rfft2(fft.ifftshift(xpad, dim=(1, 2)), norm="ortho", dim=(1, 2))
    Ah_diff = (
        fft.fftshift(fft.irfft2(Hadj * X, norm="ortho", dim=(1, 2)), dim=(1, 2))
        * fft_normalizer
    )
    return Ah_diff  # return the 4D tensor


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


def compute_grad(H, Hadj, vk, b, crop, pad):
    """
    Calculates gradient A_herm * (Av - b)

    """
    Av = calcA_diffuser(H, vk, crop)
    diff = Av - b  # 1, h, w, c
    loss = torch.linalg.norm(diff)
    grad = np.real(calcAHerm_diffuser(Hadj, diff, pad))

    return grad, loss


def gd_update(vk, parent_vars, proj_params):
    [H, Hadj, b, crop, pad, alpha, proj] = parent_vars

    # compute gradient
    gradient, loss = compute_grad(H, Hadj, vk, b, crop, pad)

    # step
    vk -= alpha * gradient

    # ensure all vk values are +ve
    # projected gradient descent
    vkp1 = proj(vk, proj_params[0], proj_params[1])
    vkp1 = non_neg(vkp1)

    return vkp1, loss


def momentum_update(vk, p, mu, parent_vars, proj_params):
    [H, Hadj, b, crop, pad, alpha, proj] = parent_vars

    # compute gradient
    gradient, loss = compute_grad(H, Hadj, vk, b, crop, pad)

    # velocity
    p_next = mu * p - alpha * gradient

    # update
    vk += -mu * p + (1 + mu) * p_next

    # projection
    vkp1 = proj(vk, proj_params[0], proj_params[1])
    vkp1 = non_neg(vkp1)

    return vkp1, p_next, loss


def fista_update(vk, tk, zk, parent_vars, proj_params):
    """
    Required fix.
    Known bug: doesn't work. 
    Try momentum update.
    """
    [H, Hadj, b, crop, pad, alpha, proj] = parent_vars
    # xkm1 = xk.clone()
    # gradient, loss = compute_grad(Hadj, H, vk, b, crop, pad)
    # xk = proj(vk - alpha * gradient, proj_params[0], proj_params[1])
    # # xk = non_neg(vk)
    # t_kp1 = (1 + np.sqrt(1 + 4 * tk**2)) / 2
    # vkp1 = xk + ((tk - 1) / t_kp1) * (xk - xkm1)

    tkp1 = (1 + np.sqrt(1 + 4 * tk**2)) / 2
    beta = (tk - 1) / tkp1
    gradient, loss = compute_grad(H, Hadj, vk, b, crop, pad)
    ykp1 = vk - alpha * gradient
    zkp1 = non_neg(proj(ykp1, proj_params[0], proj_params[1]))
    vkp1 = zkp1 + beta * (zkp1 - zk)

    return vkp1, tkp1, zkp1, loss


def grad_descent(
    h, b, gt, step_list, niter=100, proj_type="haar", update_method="fista"
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
    H, Hadj, v0, utils = initMatrices(h, gv.rgb, large_psf=gv.large_psf)
    crop, pad = utils[0], utils[1]

    df_losslist = []
    mse_losslist = []
    tv_losslist = []
    total_losslist = []

    if proj_type == "non_neg":
        proj = non_neg
    elif proj_type == "haar":
        proj = haar.tv3dApproxHaar_proj
    else:
        proj = lambda x: x

    # hyperparameters from global vars
    alpha, tau, tau_t = step_list

    parent_vars = [H, Hadj, b, crop, pad, alpha, proj]
    proj_params = [tau, tau_t]

    # random initialization
    vk = v0

    # if momentum is used
    p = 0
    mu = 0.9

    # if fista is used
    tk = 1
    zk = vk

    loss_fn = torch.nn.MSELoss()

    # update loop
    for itr in tqdm(range(niter), desc="Iterations"):
        if update_method == "gd":
            vk, loss = gd_update(vk, parent_vars, proj_params)
        elif update_method == "momentum":
            vk, p, loss = momentum_update(vk, p, mu, parent_vars, proj_params)
        elif update_method == "fista":
            vk, tk, zk, loss = fista_update(vk, tk, zk, parent_vars, proj_params)
        else:
            raise ValueError("Invalid Update Method")

        if gv.psf_convolve:
            scene_out = crop(vk)
        else:
            scene_out = proj(vk, tau, tau_t)

        # estimated scene
        scene_out = non_neg(scene_out)

        # what the data should look like at this vk
        # forward model on vk
        # NOTE: Might throw an error here, img_out is 4 dim
        img_out = (calcA_diffuser(H, vk, crop)).real

        # compute loss using known gt
        # note: this loss calc is only for observation and
        # does not play a role in the gradient descent algorithm
        df_loss = torch.log(loss_fn(img_out, b[0].real))  # take the first frame
        mse_loss = loss_fn(scene_out, gt)
        tv_loss = haar.tv3dApproxHaar_norm(scene_out)
        total_loss = df_loss + tau * tv_loss

        df_losslist.append(df_loss.detach().cpu().item())
        mse_losslist.append(mse_loss.detach().cpu().item())
        tv_losslist.append(tv_loss.detach().cpu().item())
        total_losslist.append(total_loss.detach().cpu().item())

        # Disable to save memory when unrolling! (move to a function!)
        # Handle plotting stuff
        with torch.no_grad():
            # Convert this into another function...
            if itr % gv.solver_plot_f == 0:
                scene = scene_out.detach().cpu().numpy()
                img = img_out.detach().cpu().numpy()
                img_gt = b.detach().cpu().numpy()
                fig, ax = plt.subplots(1, 3, figsize=(25, 10))

                ax[0].imshow(scene[0] / np.max(scene[0]))
                ax[0].set_title("Estimated Scene")
                ax[1].imshow(img[0] / np.max(img[0]))
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
                # plt.savefig((gv.solver_output_dir + "slices_itr_{}.png").format(itr))
                plt.close(fig)

                # plt.figure()
                fig, ax = plt.subplots(5, 1, figsize=(15, 25))

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

                # plt.title((gv.solver_plot_dir + "fista_itr_{}.png").format(itr))
                fig.tight_layout()
                # plt.savefig((gv.solver_plot_dir + "fista_itr.png"))
                plt.close(fig)

                display.clear_output(wait=True)

                df_losslist.append(df_loss.detach().cpu().item())
                mse_losslist.append(mse_loss.detach().cpu().item())
                tv_losslist.append(tv_loss.detach().cpu().item())
                total_losslist.append(total_loss.detach().cpu().item())

                if itr % gv.solver_plot_f == 0:
                    print(
                        f"Inner loop {itr}: df_loss {df_loss.detach().cpu().item():.3e}, "
                        f"mse_loss: {mse_loss.detach().cpu().item():.3e}, tv3d_loss: "
                        f"{tv_loss.detach().cpu().item():.3e}, total_loss: {total_loss.detach().cpu().item():.3e}"
                    )

                plt.close("all")

    losslist = [df_losslist, mse_losslist, tv_losslist, total_losslist]
    # np.save((gv.recon_gif_dir + "sceneout.npy"), scene_out.detach().cpu().numpy())

    return scene_out, losslist
