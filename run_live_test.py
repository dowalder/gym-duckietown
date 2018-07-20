#!/usr/bin/env python3

"""
Runs a test for a defined number of runs of defined length. The output are consecutive images of the controller
navigating the simulation.
"""

import argparse
import abc
import pathlib
import sys
from typing import Any
import collections

import yaml
import cv2
import torch
import torchvision

import numpy as np

from gym_duckietown.envs import GeneratorEnv

DUCKIE_CNN_PATH = "/home/dominik/workspace/duckie_cnn"

sys.path.append(DUCKIE_CNN_PATH)
import networks


def save_img(path: pathlib.Path, img):
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
        raise NotImplemented()


class OmegaController(Controller):
    """
    Loads the standard net that computes omega.
    """

    def __init__(self, model_path: pathlib.Path):

        self.cnn = networks.InitialNet()
        self.cnn.load_state_dict(torch.load(model_path.as_posix()))
        self.v = 0.2
        self.wheel_dist = 0.1
        self.speed_factor = 1.0

        self._transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((80, 160)),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda img: img.unsqueeze(0))
        ])

    def step(self, img: np.ndarray) -> np.ndarray:
        angle = float(self.cnn(self._transform(img)))
        vel_left = (self.v + 0.5 * angle * self.wheel_dist) * self.speed_factor
        vel_right = (self.v - 0.5 * angle * self.wheel_dist) * self.speed_factor

        return np.array([vel_left, vel_right])


class DirectAction(Controller):
    """
    Loads the standard net that computes directly the action (wheel velocities)
    """

    def __init__(self, model_path: pathlib.Path):
        self.cnn = networks.InitialNet()
        self.cnn.load_state_dict(torch.load(model_path.as_posix(), map_location="cpu"))

        self._transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((80, 160)),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda img: img.unsqueeze(0))
        ])

    def step(self, img: np.ndarray):
        with torch.no_grad():
            action = self.cnn(self._transform(img))
        return np.array(action.squeeze())


class DirectActionResnet(Controller):
    """
    Loads the resnet that computes directly the action (wheel velocities)
    """

    def __init__(self, model_path: pathlib.Path):

        self.cnn = networks.ResnetController()
        self.cnn.load_state_dict(torch.load(model_path.as_posix(), map_location="cpu"))

        self._transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((120, 160)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda img: img.unsqueeze(0))
        ])

    def step(self, img: np.ndarray):
        with torch.no_grad():
            action = self.cnn(self._transform(img))
        return np.array(action.squeeze())


class FixDisturbController(Controller):

    def __init__(self, action_model_path: pathlib.Path, disturb_model_path: pathlib.Path):

        self.action_model = OmegaController(action_model_path)
        self.disturb_model = networks.BasicConvRNN()
        self.disturb_model.load_state_dict(torch.load(disturb_model_path.as_posix(), map_location="cpu"))

        self.imgs_per_update = 5
        self.imgs_list = []
        self.action_list = []
        self.modifier = np.ones((2,))

        self.img_size = (120, 160)

        self.imgs = torch.empty(size=(self.imgs_per_update, 3, self.img_size[0], self.img_size[1]), dtype=torch.float)
        self.actions = torch.empty(size=(self.imgs_per_update, 2), dtype=torch.float)

        self.modifier_queue = collections.deque(maxlen=10)

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(self.img_size),
            torchvision.transforms.ToTensor()
        ])

    def step(self, img: np.ndarray):
        with torch.no_grad():
            action = self.action_model.step(img)

        new_action = action * self.modifier
        self.perform_disturbance_update(img, new_action)
        return new_action

    def perform_disturbance_update(self, img: np.ndarray, action: np.ndarray):
        self.imgs_list.append(img)
        self.action_list.append(action)

        if self.imgs_per_update > len(self.imgs_list):
            return

        for idx, img in enumerate(self.imgs_list):
            self.imgs[idx, :, :, :] = self.transform(img)
            self.actions[idx, :] = torch.Tensor(self.action_list[idx])

        self.imgs_list = []
        self.action_list = []

        with torch.no_grad():
            out = self.disturb_model(self.imgs, self.actions)

        self.modifier_queue.append(np.array(out[-1, :].squeeze()))
        self.modifier = np.zeros((2,))
        for mod in self.modifier_queue:
            self.modifier += mod
        self.modifier /= len(self.modifier_queue)
        print(self.modifier)


def get_controller(args: Any) -> Controller:
    """
    Controller factory.
    :param args:
    :return:
    """
    controller_args = yaml.load(pathlib.Path(args.args).read_text())
    if args.controller == "omega":
        controller = OmegaController(pathlib.Path(controller_args["model_path"]))
    elif args.controller == "direct_action":
        controller = DirectAction(pathlib.Path(controller_args["model_path"]))
    elif args.controller == "resnet":
        controller = DirectActionResnet(pathlib.Path(controller_args["model_path"]))
    elif args.controller == "disturb":
        controller = FixDisturbController(pathlib.Path(controller_args["model_path"]),
                                          pathlib.Path("/home/dominik/dataspace/models/rnn_randomwalk_forward/"
                                                       "run5_step_40000.pth"))
    else:
        raise ValueError("Unknown controller: {}".format(args.controller))

    return controller


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--controller", "-c", required=True, help="Controller to be used")
    parser.add_argument("--map", default="udem1")
    parser.add_argument("--args", help="yaml file containing arguments dictionary for controller")
    parser.add_argument("--num_seq", default=3, type=int)
    parser.add_argument("--num_img", default=200, type=int)
    parser.add_argument("--augment_action", default="1,1", type=str)

    args = parser.parse_args()

    controller = get_controller(args)

    env = GeneratorEnv(map_name=args.map)

    env.render()

    num_sequences = args.num_seq
    num_img_per_seq = args.num_img

    action_mult = np.array(list(map(float, args.augment_action.split(","))))
    assert action_mult.shape == (2,), "--augment_action must be two floats separated by a comma, e.g. '0.4,0.6'"

    save_path = pathlib.Path("/home/dominik/dataspace/images/cnn_controller_lane_following/test")

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

            action = controller.step(obs) * action_mult
            obs, reward, _, _ = env.step(action, delta_t)
            save_img(img_path, obs)

            info.append({"path": img_path.as_posix(),
                         "action": action.tolist(),
                         "reward": float(reward),
                         "delta_t": delta_t})

        info_path = seq_dir.parent / (seq_dir.stem + ".yaml")
        info_path.write_text(yaml.dump(info))


if __name__ == "__main__":
    main()
