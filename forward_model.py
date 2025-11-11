"""
Script for the Diffuser forward model

Takes a diffuser-captured image and returns a deconvolved image
representing the scene

Run as a script:
- Takes the ground truth image (static) and saves a camera capture
"""

import torch
from einops import rearrange

try:
    from . import global_vars as gv
    from . import utils_rs, utils_diff, camera
    from .utils_psf import read_psf
    from .dataloader import load_color_image_as_tensor, save_color_image_tensor
except ImportError:
    print("Falling back to absolute imports for standalone script execution!")
    import global_vars as gv
    import utils_diff, camera
    from utils_psf import read_psf
    from dataloader import load_color_image_as_tensor, save_color_image_tensor


def forward(scene, shutter, H, crop, pad, isVideo=True, mode="diffuser"):
    """
    Forward model (wrapper) function A(x)

    Input:
    :param scene: estimated or real scene to apply the forward model to
    :type scene: (t, h, w, c) where t is the number of frames given by shutter.sf_num
    :param shutter: characterizes the rolling shutter
    :type shutter: shutter object
    :param H: RFFT of the padded PSF (h, w, c)
    :type H: (h, w, c) tensor. In this case (512, 513, 3)
    :param crop and pad: functions for cropping and padding
    :param mode: choose whether to use
        RS+diffuser: "sensor"
        diffuser only: "diffuser"

    Output: sensor measurement y = A(x)
    :param image: (h, w, c)

    """

    # for compatibility with diffusion codebase
    # run rearrange only if imported
    if __name__ != "__main__":
        scene = rearrange(scene, "(t) c h w -> t h w c")

    # if mode == "diffuser"
    image = utils_diff.calcA_diffuser(H, torch.real(pad(scene)), crop)

    return image


def forward_T(b, shutter, H, crop, pad, partition=None):
    Hadj = torch.conj(H)
    ATb = crop(utils_diff.calcAHerm_diffuser(Hadj, b, pad).real)
    # reshape for diffusion compatiblity
    # return rearrange(ATb, "t h w c -> (b t) c h w", b=1)
    # return rearrange(ATb, "t h w c -> (t) c h w")
    return ATb  # t, h, w, c


def ATA(scene, shutter, H, crop, pad, b, partition=None):
    """
    Calculates gradient A_herm * (Av - b)
    where v = scene
    """
    # dimensions are handled in time deconv loop
    # requires f, h, w, c
    # in diffusion this function calculates ATA

    if scene.shape[1] != H.shape[0]:
        # ATA is called different times in the deconv loop
        # require padding
        assert scene.shape == (1, 256, 256, 3), scene.shape
        assert H.shape == (512, 257, 3), H.shape  # H uses RFFT
        scene = pad(scene)

    Av = utils_diff.calcA_diffuser(H, scene, crop)
    diff = Av - b

    assert scene.ndim == 4 and scene.shape[0] == 1, scene.shape
    assert diff.ndim == 4 and diff.shape[0] == 1, diff.shape
    assert diff.shape == (1, 256, 256, 3), diff.shape

    Hadj = torch.conj(H)
    grad = crop(utils_diff.calcAHerm_diffuser(Hadj, diff, pad).real)

    return grad


def setup(load_gt=True, load_psf=True, load_impulse=False):
    """
    Sets up the forward process by loading data
    1. load ground truth
    2. load PSF
    3. instantiate camera shutter
    4. get PSF FFT and utility functions

    """
    # 1. Load image
    # Format (1, W, H, C)
    if load_gt:
        # Use a real image
        image_gt = load_color_image_as_tensor(
            gv.sample,
            gv.rs_img_size_H,
            gv.rs_img_size_W,
            normalize=True,
            device=gv.device,
        )
    elif load_impulse:
        # Test with Delta Image (should get PSF)
        image_gt = torch.zeros((gv.rs_img_size_H, gv.rs_img_size_W))
        image_gt[gv.rs_img_size_H // 2, gv.rs_img_size_W // 2] = 1
        image_gt = image_gt.unsqueeze(-1).repeat(
            1, 1, 3
        )  # repeat along colour channels
        image_gt = image_gt.unsqueeze(0)  # add time dimension

    # # 3. Camera Shutter
    # # Ineffective in this branch!
    # shutter = camera.Shutter(
    #     gv.T_l,
    #     gv.T_e,
    #     gv.delta,
    #     gv.rs_img_size_W,
    #     gv.rs_img_size_H,
    #     gv.time_d,
    #     mode=gv.shutter_mode,
    # )
    shutter = None

    # 2. Load PSF
    if load_psf:
        psf = read_psf()

        # FFT of the PSF
        H, _, _, utils = utils_diff.initMatrices(psf.real, large_psf=gv.large_psf)

        # Utility functions for diffuser forward model
        crop, pad = utils[0], utils[1]

    return image_gt, shutter, H, crop, pad


if __name__ == "__main__":
    image_gt, shutter, H, crop, pad = setup(load_gt=True, load_psf=True)

    mode = "diffuser"

    captured_image = forward(image_gt, shutter, H, crop, pad, mode=mode)
    
    if captured_image.ndim == 4 and captured_image.shape[0] == 1:
        captured_image = captured_image.squeeze(0)

    if mode == "diffuser":
        save_color_image_tensor(captured_image, gv.measurement_path)
    elif mode == "sensor":
        save_color_image_tensor(captured_image, gv.save_rs_filepath)
