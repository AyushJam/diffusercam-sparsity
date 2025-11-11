import torch
from PIL import Image
import cv2
import os
import numpy as np


def load_color_image_as_tensor(
    path, rs_img_size_H, rs_img_size_W, normalize=True, device="cpu", save_resized=False
):
    """
    Loads a color JPG image, resizes it, and returns it as a video-format torch tensor
    of shape (1, rs_img_size_H, rs_img_size_W, 3) in RGB format.

    Parameters:
    - path (str): Path to the image file.
    - rs_img_size_H (int): Resized height.
    - rs_img_size_W (int): Resized width.
    - normalize (bool): If True, normalize pixel values to [0, 1]. Otherwise, return uint8 tensor.
    - device (str or torch.device): Device to put the tensor on.
    - save_resized (bool): If True, saves the resized image with '_resized' suffix.

    Returns:
    - torch.Tensor: Tensor of shape (1, rs_img_size_H, rs_img_size_W, 3)
    """
    image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found at path: {path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Resize image to (W, H) since OpenCV expects (width, height)
    resized_rgb = cv2.resize(
        image_rgb, (rs_img_size_W, rs_img_size_H), interpolation=cv2.INTER_CUBIC
    )

    if save_resized:
        base, ext = os.path.splitext(path)
        resized_path = f"{base}_resized{ext}"
        cv2.imwrite(resized_path, cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR))
        print(f"Resized input image saved at: {resized_path}")

    if normalize:
        image_tensor = torch.from_numpy(resized_rgb.astype(np.float32) / 255.0)
    else:
        image_tensor = torch.from_numpy(resized_rgb)

    # Shape: (H, W, 3) → add time dimension → (1, H, W, 3)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor.to(device)


def save_color_image_tensor(
    image_tensor, path, normalize=True, scale=1, grayscale=False, channel=None
):
    """
    Saves a torch RGB image tensor of shape (H, W, 3) as a color, grayscale, or single-channel image.

    Parameters:
    - image_tensor (torch.Tensor): Tensor of shape (H, W, 3), float in [0,1] or uint8 in [0,255]
    - path (str): Output file path (e.g., 'image.jpg')
    - normalize (bool): If True, assumes float input in [0,1] and rescales to [0,255]
    - scale (float): Multiplier to scale image values before saving.
    - grayscale (bool): If True, converts RGB to grayscale before saving.
    - channel (int or None): If 0, 1, or 2, saves only that RGB channel as a grayscale image.
    """
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    if image_tensor.ndim != 3 or image_tensor.shape[2] != 3:
        raise ValueError("Expected tensor shape (H, W, 3) in RGB format.")

    frame = image_tensor.detach().cpu() * scale

    if normalize:
        frame = torch.clamp(frame, 0, 1) * 255.0

    if channel is not None:
        if channel not in [0, 1, 2]:
            raise ValueError("channel must be 0 (R), 1 (G), or 2 (B)")
        frame = frame[:, :, channel]  # shape: (H, W)
        image_uint8 = frame.to(torch.uint8).numpy()
        success = cv2.imwrite(path, image_uint8)
    else:
        image_uint8 = frame.to(torch.uint8).numpy()  # (H, W, 3)
        if grayscale:
            image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
        else:
            image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(path, image_bgr)

    if not success:
        raise IOError(f"Failed to write image to path: {path}")


import os
import numpy as np
from PIL import Image


def export_gif(
    input_np, outdir, scale=1.0, take_input=False, npy=True, save_frames=False
):
    """
    Exports a numpy file (stored in .npy) or numpy array input to a GIF, scaled by `scale`.
    Supports (T, H, W, C) format or (T, H, W) format.
    """
    if npy:
        imgs = np.load(input_np)
    else:
        imgs = input_np

    # Normalize
    imgs_norm = imgs / np.amax(imgs) * 255.0
    imgs_norm = imgs_norm.astype(np.uint8)

    # Handle scaling (if needed)
    if scale != 1.0:
        scaled_imgs = []
        for img in imgs_norm:
            pil_img = Image.fromarray(
                img if img.ndim == 3 else img.squeeze(),
                mode="RGB" if img.shape[-1] == 3 else "L",
            )
            new_size = (int(pil_img.width * scale), int(pil_img.height * scale))
            pil_img = pil_img.resize(new_size, Image.LANCZOS)
            scaled_imgs.append(pil_img)
        frames = scaled_imgs
    else:
        frames = []
        for i in range(imgs_norm.shape[0]):
            if imgs_norm.ndim == 4:  # (T, H, W, C)
                if imgs_norm.shape[3] == 3:
                    frames.append(Image.fromarray(imgs_norm[i]))  # RGB image
                else:
                    frames.append(
                        Image.fromarray(imgs_norm[i, :, :, 0])
                    )  # Single channel grayscale
            else:  # (T, H, W)
                frames.append(Image.fromarray(imgs_norm[i]))

    # Save GIF
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if take_input:
        filename = input("Enter output gif filename: ")
        output_path = os.path.join(outdir, filename + ".gif")
    else:
        if npy:
            fileout_name = os.path.splitext(os.path.basename(input_np))[0]
            output_path = os.path.join(outdir, fileout_name + "_recon.gif")
        else:
            output_path = os.path.join(outdir, "recon.gif")

    frames[0].save(
        output_path, save_all=True, append_images=frames[1:], duration=100, loop=0
    )

    # Optionally save individual frames
    if save_frames:
        for idx, img in enumerate(frames):
            frame_path = os.path.join(outdir, f"frame_{idx}.png")
            img.save(frame_path)
