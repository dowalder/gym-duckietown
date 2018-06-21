#!/usr/bin/env python3

import sys
import pathlib

import yaml
import cv2

import pyglet
import numpy as np

from gym_duckietown.envs import GeneratorEnv, Parameters


def save_img(path: pathlib.Path, img):
    img = np.flip(img, 0)
    img = img[:, :, [2, 1, 0]]

    cv2.imwrite(path.as_posix(), img)


def main():

    env = GeneratorEnv(map_name="udem1")

    env.render()

    num_sequences = 50
    num_img_per_seq = 100
    save_path = pathlib.Path("/home/dominik/dataspace/images/randomwalk_forward/test")

    for num_seq in range(num_sequences):
        seq_dir = save_path / "seq_{0:05d}".format(num_seq)
        seq_dir.mkdir(exist_ok=True)
        obs = env.reset(perturb_factor=0.0)
        img_path = seq_dir / "img_00000.jpg"

        modifier = 0.5 + np.random.random(2)
        delta_t = 0.033

        save_img(img_path, obs)

        info = []

        for num_img in range(num_img_per_seq):
            img_path = seq_dir / "img_{0:05d}.jpg".format(num_img + 1)

            action = np.random.rand(2) * modifier
            obs, _, _, _ = env.step(action, delta_t)
            save_img(img_path, obs)

            info.append({"path": img_path.as_posix(),
                         "action": action.tolist(),
                         "modifier": modifier.tolist(),
                         "delta_t": delta_t})

        info_path = seq_dir.parent / (seq_dir.stem + ".yaml")
        info_path.write_text(yaml.dump(info))

    sys.exit(0)


if __name__ == "__main__":
    main()
