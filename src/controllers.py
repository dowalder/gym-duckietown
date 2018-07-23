#!/usr/bin/env python3

import abc
import collections
import pathlib

import torch
import torchvision
import numpy as np

import networks


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