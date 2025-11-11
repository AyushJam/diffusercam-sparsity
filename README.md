# Wavelet Sparsity for Lensless Imaging
Study of sparsity priors (TV and wavelets) to reconstruct images from the DiffuserCam lensless camera

### Process: 
0. Load the anaconda environment using the env file. 
1. Load your image (jpg or png) in input/
2. Run forward_model.py, this will simulate the  image capture. (captured image is stored in captured/)
3. Run deconvolve.py, this will run the gradient descent solver (FISTA) with the prior and store the reconstruction in recons/

### TODO
1. Analysis plots in utils_diff.py
2. Isolate the wavelet projection function from utils_diff.py


