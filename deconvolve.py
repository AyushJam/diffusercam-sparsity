"""
Deconvolution using Iterative Gradient Descent
> operates on diffuser only image
> no rolling shutter
> only the diffuser forward model

"""

import numpy as np
from dataloader import load_color_image_as_tensor, save_color_image_tensor
import global_vars as gv
import utils_diff
from utils_psf import read_psf

diffuser_image = load_color_image_as_tensor(
    gv.measurement_path,
    gv.rs_img_size_H,
    gv.rs_img_size_W,
    normalize=True,
    device=gv.device,
)

gt = load_color_image_as_tensor(
    gv.sample,
    gv.rs_img_size_H,
    gv.rs_img_size_W,
    normalize=True,
    device=gv.device,
)

psf = read_psf()

step_list = [gv.alpha, gv.tau, gv.tau_t]

deconvolved_scene, losslist = utils_diff.grad_descent(
    psf,
    diffuser_image,
    gt,
    step_list,
    niter=gv.niter,
    proj_type="haar", # ECE 251C FOCUS
    update_method="fista",
)

deconvolved_image = deconvolved_scene[0]
savePath = gv.recon_path
save_color_image_tensor(deconvolved_image, savePath)
