import numpy as np
import matplotlib.pyplot as plt

# Generate 256x256 RGB noise image (values 0–255, uint8)
noise_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

print(noise_img.shape, noise_img.dtype)

# Display
plt.imshow(noise_img)
plt.title("256×256 RGB Noise Image")
plt.axis("off")
plt.show()

# Optional: save
from imageio import imwrite
imwrite("noise.png", noise_img)