import os
import numpy as np
import torch
import torch.fft as fft
from IPython import display
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

try:
    from . import total_variation
    from . import haar
    from . import global_vars as gv
except ImportError:
    import total_variation
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


def grad_descent(h, b, gt, niter=100, proj_type="haar", update_method="fista"):
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

    # ECE 251C: Choose projection type
    if proj_type == "non_neg":
        proj = non_neg
    elif proj_type == "tv":
        proj = total_variation.tv3dApproxHaar_proj
        proj_params = (gv.tau_tv, gv.tau_t)
    elif proj_type == "haar":
        proj = haar.wavelet_denoising_haar
        proj_params = (gv.tau_haar, gv.level)
    else:
        proj = lambda x: x

    # hyperparameters from global vars
    parent_vars = [H, Hadj, b, crop, pad, gv.alpha, proj]

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
            scene_out = proj(vk, *proj_params)

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
        tv_loss = total_variation.tv3dApproxHaar_norm(scene_out)

        df_losslist.append(df_loss.detach().cpu().item())
        mse_losslist.append(mse_loss.detach().cpu().item())
        tv_losslist.append(tv_loss.detach().cpu().item())

    losslist = [df_losslist, mse_losslist, tv_losslist]

    # calculate psnr
    final_psnr = calculate_psnr(scene_out, gt, data_range=1.0)
    final_psnr_rounded = round(final_psnr, 2)

    # calculate ssim
    final_ssim = calculate_ssim(scene_out, gt, data_range=1.0)

    # plotting
    plot_filename = os.path.join(
        gv.plots_dir, "{}_loss_{}.png".format(gv.image.split(".")[0], gv.use_denoiser)
    )
    fig, ax = plt.subplots(
        1, 2, figsize=(12, 5)
    )  # Adjusted figsize for 3 horizontal plots
    iterations = range(1, len(df_losslist) + 1)

    ax[0].plot(
        iterations, df_losslist, color="blue", label="Measurement Loss ($\log f$)"
    )
    ax[0].set_title("1) Measurement Loss (Log Likelihood)")
    ax[0].set_xlabel("Iterations")
    ax[0].set_ylabel("Loss Value")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(iterations, mse_losslist, color="red", label="Scene Loss (MSE)")
    ax[1].set_title(
        f"2) Scene Loss (MSE vs. GT)\nFinal PSNR: {final_psnr_rounded} dB\nFinal SSIM: {final_ssim:.4f}"
    )
    ax[1].set_xlabel("Iterations")
    ax[1].set_ylabel("Loss Value")
    ax[1].grid(True)
    ax[1].legend()

    # don't plot tv norm
    # if gv.use_denoiser == "tv":
    #     ax[2].plot(iterations, tv_losslist, color="green", label="TV Norm ($\|V\|_\text{TV}$)")
    #     ax[2].set_title("3) TV Norm Regularization")
    #     ax[2].set_xlabel("Iterations")
    #     ax[2].set_ylabel("Norm Value")
    #     ax[2].grid(True)
    #     ax[2].legend()

    tau = gv.tau_tv if gv.use_denoiser == "tv" else gv.tau_haar

    plt.suptitle(
        "Loss and Regularization Convergence over Iterations\n"
        f"Hyperparameters: $\\alpha$: {gv.alpha}, $\\tau$: {tau}"
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for suptitle

    plt.savefig(plot_filename)
    plt.close(fig)

    return scene_out, losslist


# --- PSNR Calculation Function ---
def calculate_psnr(img1, img2, data_range=None):
    """
    Calculates PSNR between two tensors/arrays.

    :param img1: Reconstructed scene (torch.Tensor or numpy.ndarray)
    :param img2: Ground truth scene (torch.Tensor or numpy.ndarray)
    :param data_range: Maximum possible pixel value (e.g., 1.0 or 255).
    :return: PSNR value (float)
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()

    # Ensure arrays are correctly shaped for MSE calculation (e.g., flattened or compatible)
    # Assuming img1 and img2 have the same shape.
    mse = np.mean((img1 - img2) ** 2)

    if mse == 0:
        return float("inf")

    # Determine the maximum possible pixel value
    # If the data is normalized between 0 and 1, max_val is 1.0.
    if data_range is None:
        # Default to 1.0 if not specified, common for float tensor data
        max_val = 1.0
    else:
        max_val = data_range

    psnr = 10 * np.log10(max_val**2 / mse)
    return psnr


def calculate_ssim(img1, img2, data_range=1.0):
    """
    Calculates SSIM between two scenes with shape (T, H, W, C)

    :param img1: Reconstructed scene (torch.Tensor or numpy.ndarray)
    :param img2: Ground truth scene (torch.Tensor or numpy.ndarray)
    :param data_range: Maximum possible pixel value.
    :return: Average SSIM value (float)
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()

    # Check if the shape is (T, H, W, C)
    if img1.ndim != 4:
        raise ValueError(
            f"SSIM input array must be 4D (T, H, W, C), but got {img1.ndim}D array with shape {img1.shape}"
        )

    T = img1.shape[0]  # Time dimension

    ssim_values = []

    # Iterate over the time dimension (T)
    for t in range(T):
        # Pass the (H, W, C) slice to the SSIM function
        ssim_slice = structural_similarity(
            im1=img1[t],  # (H, W, C)
            im2=img2[t],  # (H, W, C)
            data_range=data_range,
            multichannel=True,  # <-- Crucial: Tells skimage to treat the last dimension as color channels (C)
            channel_axis=-1,  # Explicitly specifies color channels are on the last axis
        )
        ssim_values.append(ssim_slice)

    return np.mean(ssim_values)
