#!/usr/bin/env python3

import argparse
import pathlib

import cv2
import numpy as np


def brightness(img: np.ndarray) -> np.ndarray:
    scale = 0.3 + np.random.random() * 1.4
    float_img = img.astype(np.float)
    float_img *= scale
    np.clip(float_img, 0, 255, out=float_img)
    return float_img.astype(np.uint8)


def spot(img: np.ndarray) -> np.ndarray:
    kernel_size = 51
    kernel = cv2.getGaussianKernel(kernel_size, 10)
    kernel = (kernel.dot(kernel.transpose()) * 5e4).astype(np.uint8)
    kernel = kernel[:, :, None]

    height, width = img.shape[:2]
    start_height = int(np.random.random() * (height - kernel_size))
    start_width = int(np.random.random() * (width - kernel_size))
    img[start_height:start_height + kernel_size, start_width:start_width + kernel_size, :] += kernel
    return img


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_dir", required=True)
    parser.add_argument("--tgt_dir", required=True)

    args = parser.parse_args()

    src_dir = pathlib.Path(args.src_dir)
    tgt_dir = pathlib.Path(args.tgt_dir)

    if not src_dir.is_dir():
        raise FileNotFoundError("--src_dir does not exist: {}".format(src_dir))

    tgt_dir.mkdir(exist_ok=True, parents=True)

    for pth in src_dir.iterdir():
        if pth.suffix != ".jpg":
            continue
        img = cv2.imread(pth.as_posix())

        img = brightness(img)
        for _ in range(4):
            if np.random.random() < 0.2:
                img = spot(img)

        tgt_pth = tgt_dir / pth.name
        cv2.imwrite(tgt_pth.as_posix(), img)


if __name__ == "__main__":
    main()
