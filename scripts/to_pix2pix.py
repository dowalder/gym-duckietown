#!/usr/bin/env python3

"""
Creates dataset such that the pix2pix framework can be trained with it.
"""

import random
import argparse
from pathlib import Path
import shutil
from typing import Union, List

import yaml
import cv2
import numpy as np

import src.graphics


def copy_images(src_dirs: List[Path], tgt_dir: Path, filter_color=False) -> None:
    test_dir_a = tgt_dir / "testA"
    train_dir_a = tgt_dir / "trainA"
    test_dir_b = tgt_dir / "testB"
    train_dir_b = tgt_dir / "trainB"

    test_dir_a.mkdir(exist_ok=True, parents=True)
    train_dir_a.mkdir(exist_ok=True)
    test_dir_b.mkdir(exist_ok=True)
    train_dir_b.mkdir(exist_ok=True)

    test_percentage = 0.3
    test_count = 0
    train_count = 0

    for src_dir in src_dirs:
        for path in src_dir.iterdir():
            if path.suffix != ".yaml":
                continue
            seq_info = yaml.load(path.read_text())

            for entry in seq_info:
                if random.random() < test_percentage:
                    img_tgt_a = test_dir_a / "{0:06d}.jpg".format(test_count)
                    img_tgt_b = test_dir_b / "{0:06d}.jpg".format(test_count)
                    test_count += 1
                else:
                    img_tgt_a = train_dir_a / "{0:06d}.jpg".format(train_count)
                    img_tgt_b = train_dir_b / "{0:06d}.jpg".format(train_count)
                    train_count += 1

                img_src_a = src_dir / entry["path"]
                img_src_b = src_dir / entry["only_road_pth"]
                shutil.copy(img_src_a.as_posix(), img_tgt_a.as_posix())

                if filter_color:
                    img = cv2.imread(img_src_b.as_posix())
                    img = src.graphics.apply_color_filter(img)
                    cv2.imwrite(img_tgt_b.as_posix(), img)
                else:
                    shutil.copy(img_src_b.as_posix(), img_tgt_b.as_posix())


def create_fake_dataset_aligned(src_dirs: List[Path], tgt_dir: Path):
    for src_dir in src_dirs:
        for pth in src_dir.iterdir():
            if pth.suffix != ".jpg":
                continue
            img = cv2.imread(pth.as_posix())
            height, width, channels = img.shape

            new_img = np.zeros((height, width * 2, channels))
            new_img[:, :width, :] = img
            img_path = tgt_dir / pth.name
            cv2.imwrite(img_path.as_posix(), new_img)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_dir", required=True)
    parser.add_argument("--tgt_dir", required=True)
    parser.add_argument("--fake_dataset", action="store_true")
    parser.add_argument("--filter_color", action="store_true")

    args = parser.parse_args()

    if "," in args.src_dir:
        src_dirs = [Path(path) for path in args.src_dir.split(",")]
    else:
        src_dirs = [Path(args.src_dir)]

    for path in src_dirs:
        assert path.is_dir(), "Not a directory: {}".format(path)

    tgt_dir = Path(args.tgt_dir)
    tgt_dir.mkdir(exist_ok=True, parents=True)

    if args.fake_dataset:
        create_fake_dataset_aligned(src_dirs, tgt_dir)
    else:
        copy_images(src_dirs, tgt_dir, filter_color=args.filter_color)


if __name__ == "__main__":
    main()
