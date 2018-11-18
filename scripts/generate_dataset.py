#!/usr/bin/env python3

"""
Extracts images from the data stored with generate_data.py or optimal_trajectory_gen.py. Applies several transformation
to all images, if wished.
"""

import pathlib
import random
import argparse
import json
import shutil
from typing import List, Callable

import yaml
import numpy as np
import cv2

from gym_duckietown.envs import SimpleSimEnv
import src.graphics


class Transform:

    def __init__(self):
        self.transforms = []

    def add_transform(self, transform: Callable):
        self.transforms.append(transform)

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __bool__(self):
        return bool(self.transforms)


def write_imgs_from_map(map_name: str, save_dir: pathlib.Path, test_percentage=0.3):
    env = SimpleSimEnv(map_name=map_name)

    file_path = pathlib.Path('experiments/demos_{}.json'.format(map_name))
    if not file_path.is_file():
        raise ValueError("Could not find the file containing the generated trajectories: {}".format(file_path))

    data = json.loads(file_path.read_text())

    demos = data['demos']
    positions = map(lambda d: d['positions'], demos)
    actions = map(lambda d: d['actions'], demos)

    positions = sum(positions, [])
    actions = sum(actions, [])

    test_dir = save_dir / "test"
    train_dir = save_dir / "train"

    test_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    print("Found {} positions to be converted to images...".format(len(positions)))

    for idx, position in enumerate(positions):
        cur_pos = np.array(position[0])
        cur_angle = position[1]
        vels = actions[idx]

        env.cur_pos = cur_pos
        env.cur_angle = cur_angle

        obs = env.render_obs().copy()
        obs = obs[..., ::-1]

        if random.random() < test_percentage:
            img_path = test_dir / "{0:06d}.jpg".format(idx)
            lbl_path = test_dir / "{0:06d}.txt".format(idx)
        else:
            img_path = train_dir / "{0:06d}.jpg".format(idx)
            lbl_path = train_dir / "{0:06d}.txt".format(idx)

        cv2.imwrite(img_path.as_posix(), obs)
        lbl_path.write_text(" ".join(map(str, vels)))


def write_imgs_from_srcdir(src_dir: pathlib.Path, tgt_dir: pathlib.Path, keep_zeros_prob=1.0, only_road=False) -> None:
    test_dir = tgt_dir / "test"
    train_dir = tgt_dir / "train"
    test_dir.mkdir(exist_ok=True)
    train_dir.mkdir(exist_ok=True)

    test_percentage = 0.3
    test_count = 0
    train_count = 0

    for path in src_dir.iterdir():
        if path.suffix != ".yaml":
            continue
        seq_info = yaml.load(path.read_text())

        for entry in seq_info:
            if abs(entry["omega"]) < 0.1 and random.random() > keep_zeros_prob:
                continue
            if random.random() < test_percentage:
                img_tgt = test_dir / "{0:06d}.jpg".format(test_count)
                lbl_tgt = test_dir / "{0:06d}.txt".format(test_count)
                test_count += 1
            else:
                img_tgt = train_dir / "{0:06d}.jpg".format(train_count)
                lbl_tgt = train_dir / "{0:06d}.txt".format(train_count)
                train_count += 1
            img_src = src_dir / entry["only_road_pth"] if only_road else src_dir / entry["path"]
            shutil.copy(img_src.as_posix(), img_tgt.as_posix())
            lbl_tgt.write_text(str(entry["omega"]))


def transform_images(src_dir: pathlib.Path, transform: Callable):
    for pth in src_dir.iterdir():
        if pth.suffix != ".jpg":
            continue
        img = cv2.imread(pth.as_posix())
        img = transform(img)
        cv2.imwrite(pth.as_posix(), img)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--map", help="Name of the map")
    parser.add_argument("--src_dir", help="If specified, the data is assumed to be from sequences")
    parser.add_argument("--tgt_dir", required=True, help="place to store the images")
    parser.add_argument("--flatten_dist", action="store_true", help="if the data distribution should be flattened")
    parser.add_argument("--only_road", action="store_true", help="use the only road images")
    parser.add_argument("--transform_only_road", action="store_true", help="transforms the only road images")
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--overflow", action="store_true")

    args = parser.parse_args()

    if args.transform_only_road:
        args.only_road = True

    if args.flatten_dist:
        keep_zeros_prob = 0.03
    else:
        keep_zeros_prob = 1.0

    if args.src_dir is None:
        if args.map is None:
            raise ValueError("You need to specify either --src_dir or --map")
        if args.only_road:
            raise ValueError("You cant specify both --map and --only_road")
        if args.transform_only_road:
            raise ValueError("You cant specify both --map and --transform_only_road")
        write_imgs_from_map(map_name=args.map, save_dir=pathlib.Path(args.tgt_dir))
    else:
        if args.map is not None:
            raise ValueError("You can't specify both --map and --src_dir")
        write_imgs_from_srcdir(pathlib.Path(args.src_dir),
                               pathlib.Path(args.tgt_dir),
                               keep_zeros_prob=keep_zeros_prob,
                               only_road=args.only_road)

    transform = Transform()
    if args.tranform_only_road:
        transform.add_transform(src.graphics.apply_color_filter)
    if args.invert:
        transform.add_transform(src.graphics.invert)
    if args.overflow:
        transform.add_transform(src.graphics.overflow)

    if transform:
        print("transforming images...")
        transform_images(pathlib.Path(args.tgt_dir) / "train", transform)
        transform_images(pathlib.Path(args.tgt_dir) / "test", transform)


if __name__ == "__main__":
    main()
