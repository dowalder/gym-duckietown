#!/usr/bin/env python3

"""
Creates augmented dataset to train pix2pix from only road images.
"""

import argparse
import pathlib
import random
from typing import List

import yaml
import cv2
import numpy as np

import src.graphics
import src.paths


def load_only_road_imgs(paths: List[pathlib.Path]) -> List[pathlib.Path]:
    """
    Assumes structure given by generate_data.py
    :param paths: list of directories containing runs created by generate_data.py
    :return: list of paths pointing to only_road images
    """
    road_imgs = []
    for path in paths:
        for pth in path.iterdir():
            if pth.suffix != ".yaml":
                continue
            info = yaml.load(pth.read_text())
            for entry in info:
                if "only_road_pth" not in entry:
                    raise ValueError("Found invalid sequence info: {}".format(pth))
                road_imgs.append(path / entry["only_road_pth"])

    return road_imgs


def create_images(road_dirs: List[pathlib.Path],
                  background_dirs: List[pathlib.Path],
                  tgt_dir: pathlib.Path,
                  num_imgs=1000,
                  test_percentage=0.3,
                  attach_width=False) -> None:
    a_test = tgt_dir / "testA"
    b_test = tgt_dir / "testB"
    a_train = tgt_dir / "trainA"
    b_train = tgt_dir / "trainB"
    a_test.mkdir(exist_ok=True)
    b_test.mkdir(exist_ok=True)
    a_train.mkdir(exist_ok=True)
    b_train.mkdir(exist_ok=True)

    print("loading road images...")
    road_imgs = load_only_road_imgs(road_dirs)
    print("loading background images...")
    background_imgs = []
    for background_dir in background_dirs:
        background_imgs += [background_dir / img for img in background_dir.glob("*.jpg")]

    for idx in range(num_imgs):
        # ------------- Create Images ------------- #
        road = cv2.imread(random.choice(road_imgs).as_posix())
        height, width = road.shape[:2]
        height_mod = random.randint(0, 100)
        width_mod = random.randint(0, 100)

        background = cv2.imread(random.choice(background_imgs).as_posix())
        background = cv2.resize(background, (width + width_mod, height + height_mod))
        road_filtered_extended = np.zeros((height + height_mod, width + width_mod, 3), dtype=np.uint8)

        scale_width_mod = width_mod if attach_width else random.randint(0, width_mod)
        road = cv2.resize(road, (width + scale_width_mod, height))
        road_filtered = src.graphics.apply_color_filter(road)

        width_start = random.randint(0, width_mod - scale_width_mod)
        width_end = width_start + width + scale_width_mod

        background_box = background[height_mod:, width_start:width_end, :]
        background_box *= (road == 0)
        background_box += road

        road_filtered_extended[height_mod:, width_start:width_end, :] = road_filtered

        # ------------- Data Augmentation ------------- #
        # add gradient
        background = src.graphics.gradient_lighting(background)
        # add spots
        for _ in range(4):
            if np.random.random() < 0.2:
                background = src.graphics.spot(background)
        # add noise
        background = background.astype(np.float)
        background += np.random.randint(-20, 20, background.shape)

        # scale each channel independently
        background *= (np.random.random((1, 1, 3)) * 1.7 + 0.1)
        background = np.clip(background, 0, 255).astype(np.uint8)

        # ------------- Save Images ------------- #
        if random.random() < test_percentage:
            a_path = a_test / "{0:05d}.jpg".format(idx)
            b_path = b_test / "{0:05d}.jpg".format(idx)
        else:
            a_path = a_train / "{0:05d}.jpg".format(idx)
            b_path = b_train / "{0:05d}.jpg".format(idx)

        cv2.imwrite(a_path.as_posix(), background)
        cv2.imwrite(b_path.as_posix(), road_filtered_extended)

        if idx % 500 == 0:
            print("processed images: {}/{}".format(idx, num_imgs))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--road_dirs", required=True)
    parser.add_argument("--background_dirs", required=True)
    parser.add_argument("--tgt_dir", required=True)
    parser.add_argument("--num_imgs", default=1000, type=int)

    args = parser.parse_args()

    road_dirs = src.paths.str_to_list(args.road_dirs, need_exist=True, is_dir=True)
    background_dirs = src.paths.str_to_list(args.background_dirs, need_exist=True, is_dir=True)

    tgt_dir = pathlib.Path(args.tgt_dir)
    tgt_dir.mkdir(exist_ok=True, parents=True)

    create_images(road_dirs, background_dirs, tgt_dir, num_imgs=args.num_imgs, test_percentage=0.05)


if __name__ == "__main__":
    main()
