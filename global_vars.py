"""
Global Variables
Required for the forward model
"""

import torch

""" Path """
image = "natural.png"

solver_output_dir = "./outputs/"
recon_dir = "./recons/"
plots_dir =  "./plots/"
# plots_dir =  "./plots/" + image.split(".")[0] + "/"
data_dir = "./input/"
captured_dir = "./captured/"

sample = data_dir + image
measurement_path = captured_dir + "rs_" + image

"""" System """
# Device settings (GPU/CPU)
dtype = torch.float32
device = None

if torch.cuda.is_available():
    # Set device to GPU
    device = torch.device("cuda")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print("Using GPU")
else:
    # Set device to CPU
    device = torch.device("cpu")
    print("Using CPU")


"""" PSF """
psf_format = "mat"
load_psf = True
psf_path = "./psf/hdrpsf.mat"
fileout_psf = "./psf/"
psf_convolve = True
large_psf = False
normalize_psf = True


""" System Parameters """
# RS Sensor
rs_physical_sensor_H = 256
rs_physical_sensor_W = 256

# RS Timing
T_l = 13e-6  # line time
T_e = 50.0 * T_l  # exposure time
delta = 1e-6  # pixel size
shutter_mode = "single"  # or "dual"

# Time dimensions: Number of shutter frames
if shutter_mode == "dual":
    time_d = int(
        (rs_physical_sensor_H / 2 + round(T_e / T_l) - 1)
    )  
elif shutter_mode == "single":
    time_d = int(
        (rs_physical_sensor_H + round(T_e / T_l) - 1)
    )  

# GS Sensor
gs_physical_sensor_H = 1200  # in pixels
gs_physical_sensor_W = 1920  # pixels


""" Data """
rgb = True

rs_img_size_H = int(rs_physical_sensor_H)
rs_img_size_W = int(rs_physical_sensor_W)

gs_img_size_H = int(gs_physical_sensor_H)
gs_img_size_W = int(gs_physical_sensor_W)


""" Optimization Parameters """
solver_plot_f = 50
niter = 200
alpha = 0.8  # step size for gradient update

add_noise = False
if add_noise:
    noise_std = 2e-6

# available: "tv" "haar"
use_denoiser = "tv"

if use_denoiser == "tv":
    tau_tv = 2e-6  # spatial regularization for TV
    tau_t = 0 # temporal denoising weight for TV
elif use_denoiser == "haar":
    tau_haar = 1
    level = 90
    pass


    
