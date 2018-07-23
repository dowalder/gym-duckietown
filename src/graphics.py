#!/usr/bin/env python3
import random

import cv2
import numpy as np


def create_gradient(size=(120, 160), hor=True, ver=True, low=0.3):
    assert isinstance(size, tuple)
    assert len(size) == 2

    if hor:
        vec1 = np.linspace(low, 1.0, size[0])
        vec1 = vec1[:, None]
    else:
        vec1 = np.ones((size[0], 1))

    if ver:
        vec2 = np.linspace(low, 1.0, size[1])
        vec2 = vec2[None, :]
    else:
        vec2 = np.ones((1, size[1]))

    return vec1.dot(vec2)


def scale_no_overlflow(img: np.ndarray, scale: float, img_type=np.uint8) -> np.ndarray:
    float_img = img.astype(np.float) * scale
    np.clip(float_img, 0, 255, out=float_img)
    return float_img.astype(img_type)


def apply_color_filter(img: np.ndarray) -> np.ndarray:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Keep the red or the bright pixels (all road marks, basically)
    mask = (img[:, :, 2] > 100) | (img_gray > 100)
    mask = mask[:, :, None]
    return img * mask


def brightness(img: np.ndarray, scale_range=(0.3, 1.7)) -> np.ndarray:
    assert isinstance(scale_range, tuple)
    assert len(scale_range) == 2

    scale = scale_range[0] + np.random.random() * (scale_range[1] - scale_range[0])
    return scale_no_overlflow(img, scale)


def spot(img: np.ndarray, kernel_size=51, sigma=10, scale_factor=100.0) -> np.ndarray:
    scale = kernel_size * sigma * scale_factor
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = (kernel.dot(kernel.transpose()) * scale).astype(np.uint8)
    kernel = kernel[:, :, None]

    height, width = img.shape[:2]
    start_height = int(np.random.random() * (height - kernel_size))
    start_width = int(np.random.random() * (width - kernel_size))
    img[start_height:start_height + kernel_size, start_width:start_width + kernel_size, :] += kernel
    return img


def gradient_lighting(img: np.ndarray) -> np.ndarray:
    choices = [True, False]
    hor = random.choice(choices)
    ver = random.choice(choices)
    scale = 0.3 + random.random() * 2.0
    grad = create_gradient(img.shape[:2], hor=hor, ver=ver)
    grad *= scale
    grad = grad[:, :, np.newaxis]
    return img * grad


