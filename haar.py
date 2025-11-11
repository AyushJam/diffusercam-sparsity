"""
May 8 2023
Kevin Tandi

This file contains implementation of Haar transform and inverse Haar transform.
"""

import numpy as np
import torch


def tv3dApproxHaar_norm(x):
    """
    Computes 3D total variation approximated using Haar wavelets

    :param x: 3D (t,h,w) tensor to be projected
    :type x: 3D torch tensor dim = (T, H, W)

    :return: tv3d, 1-norm of wavelet transformed data
    :rtype: tensor
    """

    # Pad along time axis if needed
    if x.size(0) % 2 != 0:
        last_frame = x[-1:, :, :]  # shape: (1, h, w)
        x = torch.cat([x, last_frame], dim=0)  # (t+1, h, w)

    D = 3  # number of dimensions
    y = torch.zeros_like(x, dtype=torch.float)

    # Axes in (t, h, w): 0=time, 1=height, 2=width
    # So we loop over these same axes directly
    for axis in range(0, 3):
        y += ht3(x, axis, False, 0.0)
        y += ht3(x, axis, True, 0.0)

    y = y / (2.0 * D)
    tv3D_norm = torch.linalg.vector_norm(y, ord=1)

    return tv3D_norm


# def tv3dApproxHaar_proj(x, tau, alpha):
#     """
#     3D total variation proximal/projection operator approximated using Haar wavelets

#     :param x: 3D (t,h,w) tensor to be projected
#     :type x: torch.Tensor, shape = (T, H, W)

#     :param tau: threshold scaling value for soft thresholding operation on Haar transform output
#     :type tau: float

#     :param alpha: custom scaling for time axis soft thresholding
#     :type alpha: float

#     :return: y, projection of x using total variation
#     :rtype: torch.Tensor, shape = (T, H, W)
#     """

#     padded = False
#     if x.size(0) % 2 != 0:
#         last_frame = x[-1:, :, :]  # Extract the last frame (shape: [1, h, w])
#         x = torch.cat([x, last_frame], dim=0)  # Pad time axis
#         padded = True

#     D = 3.0  # Number of dimensions
#     gamma = 1.0
#     thresh = torch.sqrt(torch.tensor(2.0)) * D * tau * gamma
#     y = torch.zeros_like(x, dtype=torch.float)

#     for axis in range(3):  # 0=t, 1=h, 2=w
#         t_scale = alpha if axis == 0 else 1.0
#         y += iht3(ht3(x, axis, False, thresh * t_scale), axis, False)
#         y += iht3(ht3(x, axis, True, thresh * t_scale), axis, True)

#     y = y / (2.0 * D)

#     if padded:
#         return y[:-1, :, :]  # Remove padding on time axis
#     else:
#         return y


def tv3dApproxHaar_proj(x, tau, alpha):
    """
    Applies 3D TV projection separately to each channel.
    x: shape (T, H, W, C)
    Returns projected x of the same shape.
    """
    if x.size(-1) != 3:
        raise ValueError("Expected input shape (T, H, W, C) for RGB data.")

    out = torch.zeros_like(x)
    for c in range(x.size(-1)):
        x_c = x[..., c]  # shape: (T, H, W)
        padded = False
        if x_c.size(0) % 2 != 0:
            x_c = torch.cat([x_c, x_c[-1:].clone()], dim=0)
            padded = True

        D = 3.0
        gamma = 1.0
        thresh = torch.sqrt(torch.tensor(2.0)) * D * tau * gamma
        y = torch.zeros_like(x_c)

        for axis in range(3):
            t_scale = alpha if axis == 0 else 1.0
            y += iht3(ht3(x_c, axis, False, thresh * t_scale), axis, False)
            y += iht3(ht3(x_c, axis, True, thresh * t_scale), axis, True)

        y /= (2.0 * D)
        if padded:
            y = y[:-1]

        out[..., c] = y

    return out


def ht3(x, ax, shift, thresh): 
    '''
    3D forward Haar transform on x, with soft thresholding

    :param x: 3D (t, h, w) tensor to be transformed
    :type x: torch.Tensor
    :param ax: axis along which to apply Haar transform (0=time, 1=height, 2=width)
    :type ax: int
    :param shift: whether to apply a shift (rolling one unit along axis)
    :type shift: bool
    :param thresh: soft thresholding value (0 means no thresholding)
    :type thresh: float

    :return: Haar-transformed tensor with soft thresholding applied
    :rtype: torch.Tensor with same shape as input (t, h, w)
    '''
    alpha = torch.zeros_like(x, dtype=torch.float)
    C = 1.0 / torch.sqrt(torch.tensor(2.0))

    if shift:
        x = torch.roll(x, -1, ax)

    m = torch.floor(torch.tensor(x.size(ax) / 2)).to(torch.int)

    if ax == 0:
        alpha[0:m, :, :] = C * (x[1::2, :, :] + x[0::2, :, :])
        alpha[m:, :, :] = C * soft_thres(x[1::2, :, :] - x[0::2, :, :], thresh)

    elif ax == 1:
        alpha[:, 0:m, :] = C * (x[:, 1::2, :] + x[:, 0::2, :])
        alpha[:, m:, :] = C * soft_thres(x[:, 1::2, :] - x[:, 0::2, :], thresh)

    else:
        alpha[:, :, 0:m] = C * (x[:, :, 1::2] + x[:, :, 0::2])
        alpha[:, :, m:] = C * soft_thres(x[:, :, 1::2] - x[:, :, 0::2], thresh)

    return alpha


def iht3(alpha, ax, shift): 
    '''
    3D inverse Haar transform on alpha

    :param alpha: 3D (t, h, w) tensor of Haar coefficients
    :type alpha: torch.Tensor
    :param ax: axis to apply inverse transform (0=time, 1=height, 2=width)
    :type ax: int
    :param shift: whether to shift result back by one (inverse of what ht3 may have done)
    :type shift: bool

    :return: reconstructed signal from Haar coefficients
    :rtype: torch.Tensor with shape (t, h, w)
    '''
    y = torch.zeros_like(alpha, dtype=torch.float)
    C = 1.0 / torch.sqrt(torch.tensor(2.0))

    m = torch.floor(torch.tensor(alpha.size(ax) / 2)).to(torch.int)

    if ax == 0:
        y[0::2, :, :] = C * (alpha[0:m, :, :] - alpha[m:, :, :])
        y[1::2, :, :] = C * (alpha[0:m, :, :] + alpha[m:, :, :])
    elif ax == 1:
        y[:, 0::2, :] = C * (alpha[:, 0:m, :] - alpha[:, m:, :])
        y[:, 1::2, :] = C * (alpha[:, 0:m, :] + alpha[:, m:, :])
    else:
        y[:, :, 0::2] = C * (alpha[:, :, 0:m] - alpha[:, :, m:])
        y[:, :, 1::2] = C * (alpha[:, :, 0:m] + alpha[:, :, m:])

    if shift:
        y = torch.roll(y, 1, ax)

    return y


def soft_thres(x, tau):
    """
        Soft thresholds input x according to tau

        :param x: input tensor
        :type x: n-dimensional tensor
        :param tau: threshold constant
        :type tau: float?
    `
        :return: soft thresholded values
        :rtype: n-dimensional tensor
    """
    thresed = torch.maximum(torch.abs(x) - tau, torch.tensor(0))
    thresed *= torch.sign(x)

    return thresed


# test 2d versions for debugging


def tv2dApproxHaar(x, tau):
    D = 2  # Dimensionality
    gamma = 1  # step size
    thresh = torch.sqrt(torch.tensor(2.0)) * 2.0 * D * tau * gamma
    y = torch.zeros_like(x, dtype=torch.float)

    # Speed this up later #
    for axis in range(0, 2):

        # todo: not implemented shift trick yet.
        y += iht2(ht2(x, axis, False, thresh), axis, False)
        y += iht2(ht2(x, axis, True, thresh), axis, True)

    y = y / (
        2.0 * D
    )  # 2 is for shift scaling, D is for dimension scaling. See short explanation of theorem in notebook for reasoning.
    return y


def ht2(x, ax, shift, thresh):
    alpha = torch.zeros_like(x, dtype=torch.float)
    C = 1.0 / torch.sqrt(torch.tensor(2.0))

    if shift:
        x = torch.roll(x, -1, ax)

    m = torch.floor(torch.tensor(x.size(ax) / 2)).to(torch.int)

    if ax == 0:
        alpha[0:m, :] = C * (x[1::2, :] + x[0::2, :])
        alpha[m:, :] = C * soft_thres(x[1::2, :] - x[0::2, :], thresh)

    elif ax == 1:
        alpha[:, 0:m] = C * (x[:, 1::2] + x[:, 0::2])
        alpha[:, m:] = C * soft_thres(x[:, 1::2] - x[:, 0::2], thresh)

    return alpha


def iht2(alpha, ax, shift):

    y = torch.zeros_like(alpha, dtype=torch.float)
    C = 1.0 / torch.sqrt(torch.tensor(2.0))

    m = torch.floor(torch.tensor(alpha.size(ax) / 2)).to(torch.int)

    if ax == 0:
        y[0::2, :] = C * (alpha[0:m, :] - alpha[m:, :])
        y[1::2, :] = C * (alpha[0:m, :] + alpha[m:, :])
    elif ax == 1:
        y[:, 0::2] = C * (alpha[:, 0:m] - alpha[:, m:])
        y[:, 1::2] = C * (alpha[:, 0:m] + alpha[:, m:])

    if shift:
        y = torch.roll(y, 1, ax)

    return y


# 1d version for time traces solver


def tv1dApproxHaar(x, tau):
    D = 1  # Dimensionality
    gamma = 1  # step size
    thresh = torch.sqrt(torch.tensor(2.0)) * 2.0 * D * tau * gamma
    y = torch.zeros_like(x, dtype=torch.float)

    # Speed this up later #
    for axis in range(0, 1):

        # todo: not implemented shift trick yet.
        y += iht1(ht1(x, axis, False, thresh), axis, False)
        y += iht1(ht1(x, axis, True, thresh), axis, True)

    y = y / (
        2.0 * D
    )  # 2 is for shift scaling, D is for dimension scaling. See short explanation of theorem in notebook for reasoning.
    return y


def ht1(x, ax, shift, thresh):
    alpha = torch.zeros_like(x, dtype=torch.float)
    C = 1.0 / torch.sqrt(torch.tensor(2.0))

    if shift:
        x = torch.roll(x, -1, ax)

    m = torch.floor(torch.tensor(x.size(ax) / 2)).to(torch.int)

    if ax == 0:
        alpha[0:m] = C * (x[1::2, :] + x[0::2, :])
        alpha[m:] = C * soft_thres(x[1::2, :] - x[0::2, :], thresh)

    return alpha


def iht1(alpha, ax, shift):

    y = torch.zeros_like(alpha, dtype=torch.float)
    C = 1.0 / torch.sqrt(torch.tensor(2.0))

    m = torch.floor(torch.tensor(alpha.size(ax) / 2)).to(torch.int)

    if ax == 0:
        y[0::2, :] = C * (alpha[0:m, :] - alpha[m:, :])
        y[1::2, :] = C * (alpha[0:m, :] + alpha[m:, :])
    elif ax == 1:
        y[:, 0::2] = C * (alpha[:, 0:m] - alpha[:, m:])
        y[:, 1::2] = C * (alpha[:, 0:m] + alpha[:, m:])

    if shift:
        y = torch.roll(y, 1, ax)

    return y
