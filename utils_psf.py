import numpy as np
import torch
from PIL import Image
import cv2
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import colors

try:
    from . import global_vars as gv
    from .dataloader import save_color_image_tensor
except:
    import global_vars as gv
    from dataloader import save_color_image_tensor

def read_psf():
    """
    Reads PSF and handles these formats: png, tiff, mat

    Inputs:
    - psf file name
    - rolling shutter image size (downsampled, if)

    Returns
    - a numpy array of psf
    - format (H, W, C)
    """

    in_format = gv.psf_format

    # .tif file
    if in_format == "tif" or in_format == "png" or in_format == "tiff":
        image = Image.open(gv.psf_path)
        image_data = np.asarray(image, dtype=np.float32)

    # .mat file
    elif in_format == "mat":
        mat_data = scipy.io.loadmat(gv.psf_path)
        image_data = mat_data["hdr_demosaic"]

    else:
        raise FileNotFoundError

    # Downsample the extracted region to the target size
    new_size = (gv.rs_img_size_W, gv.rs_img_size_H)
    image = cv2.resize(image_data, new_size, cv2.INTER_CUBIC)

    output = np.asarray(image, dtype=np.float32)
    output = torch.from_numpy(output.copy()).float()

    # Add noise to PSF
    if gv.add_noise:
        output = output + torch.randn_like(output) * gv.noise_std

    if gv.normalize_psf:
        psf_sum = torch.sum(
            output, dim=(0, 1), keepdim=True
        )  # Compute sum of each channel
        psf = output / psf_sum
    else:
        psf = output

    if psf.ndim == 2:
        # make it 3D
        psf = psf.unsqueeze(-1)

    # select first channel if monochrome
    if not gv.rgb:
        psf = psf[:, :, 0].unsqueeze(-1)
    # Else keep all 3 channels

    # return output normalized from 0 to 1 using sum, to be interpeted as probability distribution of photons
    return psf  # torch.Tensor


def plot_psf():
    psf = read_psf()

    print("PSF Shape: ", psf.shape)

    # Convert RGB to grayscale using standard luminosity weights
    if psf.ndim == 3 and psf.shape[2] == 3:
        psf_np = np.dot(psf[..., :3], [0.2989, 0.5870, 0.1140])

    # Avoid log 0
    psf_np = np.clip(psf_np, 1e-12, None)

    # Create figure
    fig = plt.figure(figsize=(6, 5))

    # Plot with LogNorm
    im = plt.imshow(
        psf_np, cmap="gray", norm=colors.LogNorm(vmin=1e-9, vmax=psf_np.max())
    )
    
    plt.colorbar(im)
    plt.title(f"PSF (Sum = {psf_np.sum():.2f})")
    plt.tight_layout()
    plt.savefig(gv.fileout_psf + "psf_fixed_og.png")
    plt.close(fig)

    # save psf as image
    save_color_image_tensor(psf, gv.rngcam_dir + "psf/psf_og.tif")
    


if __name__ == "__main__":
    plot_psf()
