#!/usr/bin/env python3

import argparse
import abc
import pathlib
import sys
from typing import Any

import yaml
import cv2
import torch
import torchvision

import numpy as np

from gym_duckietown.envs import GeneratorEnv, Parameters


def save_img(path: pathlib.Path, img):
    img = np.flip(img, 0)
    img = img[:, :, [2, 1, 0]]

    cv2.imwrite(path.as_posix(), img)


class Controller(abc.ABC):

    @abc.abstractmethod
    def step(self, img: np.ndarray) -> np.ndarray:
        """
        Controller step that computes the steering command.

        :param img:
        :return: steering angle
        """
        raise NotImplemented


class RealDataController(Controller):

    def __init__(self, pkg_path: pathlib.Path, model_path: pathlib.Path):
        sys.path.append(pkg_path.as_posix())
        import networks

        self.cnn = networks.InitialNet()
        self.cnn.load_state_dict(torch.load(model_path.as_posix()))
        self.v = 0.3
        self.wheel_dist = 0.1
        self.max_v = 1.0

        self._transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((80, 160)),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda img: img.unsqueeze(0))
        ])

    def step(self, img: np.ndarray) -> np.ndarray:
        angle = float(self.cnn(self._transform(img)))
        vel_left = (self.v + 0.5 * angle * self.wheel_dist) * 2.0
        vel_right = (self.v - 0.5 * angle * self.wheel_dist) * 2.0

        # vel_left = min(self.max_v, max(-self.max_v, vel_left))
        # vel_right = min(self.max_v, max(-self.max_v, vel_right))

        return np.array([vel_left, vel_right])


def get_controller(args: Any) -> Controller:
    if args.controller == "real_data":
        controller_args = yaml.load(pathlib.Path(args.args).read_text())
        controller = RealDataController(pathlib.Path(controller_args["pkg_path"]),
                                        pathlib.Path(controller_args["model_path"]))
    else:
        raise ValueError("Unknown controller: {}".format(args.controller))

    return controller


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--controller", "-c", required=True, help="Controller to be used")
    parser.add_argument("--map", default="udem1")
    parser.add_argument("--args", help="yaml file containing arguments dictionary for controller")

    args = parser.parse_args()

    controller = get_controller(args)

    env = GeneratorEnv(map_name=args.map)

    env.render()

    num_sequences = 40
    num_img_per_seq = 100
    save_path = pathlib.Path(
        "/home/dominik/dataspace/images/cnn_controller_lane_following/real_data_learned_caffe_copy")

    for num_seq in range(num_sequences):
        print("running on sequence: {}".format(num_seq))
        seq_dir = save_path / "seq_{0:05d}".format(num_seq)
        seq_dir.mkdir(exist_ok=True)
        obs = env.reset(perturb_factor=0.0)
        img_path = seq_dir / "img_00000.jpg"

        delta_t = 0.1

        save_img(img_path, obs)

        info = []

        for num_img in range(num_img_per_seq):
            img_path = seq_dir / "img_{0:05d}.jpg".format(num_img + 1)

            action = controller.step(obs)
            obs, reward, _, _ = env.step(action, delta_t)
            save_img(img_path, obs)

            info.append({"path": img_path.as_posix(),
                         "action": action.tolist(),
                         "reward": reward,
                         "delta_t": delta_t})

        info_path = seq_dir.parent / (seq_dir.stem + ".yaml")
        info_path.write_text(yaml.dump(info))


if __name__ == "__main__":
    main()
