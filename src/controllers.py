#!/usr/bin/env python3

import abc
import collections
import pathlib
from typing import Tuple, List, Union

import torch
import torchvision
import numpy as np

import src.networks
import src.params
import src.graphics


def omega_to_wheels(omega, v, wheel_dist=0.1) -> List[float]:
    v_left = v + 0.5 * omega * wheel_dist
    v_right = v - 0.5 * omega * wheel_dist
    return [v_left, v_right]


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

    def __init__(self, params: src.params.Params):

        self.cnn = src.networks.BasicLaneFollower(params)
        self.cnn.load()
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


class DiscreteAction(Controller):

    def __init__(self, params: Union[src.params.TrainParams, src.params.TestParams]):
        low = params.get("action_min", no_raise=False)
        up = params.get("action_max", no_raise=False)
        assert low < up, "low = {}, up = {}".format(low, up)

        num_actions = params.get("num_actions", no_raise=False)
        img_size = params.get("image_size", no_raise=True)
        img_size = (120, 160) if img_size is None else src.graphics.size_from_string(img_size)

        diff = float(up - low)
        self.idx_to_omega = [low + diff / 2.0 / num_actions + i * diff / num_actions for i in range(num_actions)]
        self.cnn = src.networks.ActionEstimator(params)
        self.cnn.load()

        self.softmax = torch.nn.Softmax()

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x.unsqueeze(0))
        ])

        self.last_step = None

    def step(self, img: np.ndarray):
        out = self.transform(img)
        with torch.no_grad():
            out = self.cnn(out)
            out = self.softmax(out)
        idx = np.argmax(out).item()
        omega = self.idx_to_omega[idx]
        action = np.array(omega_to_wheels(omega, 0.2))
        self.last_step = {"omega": omega, "softmax": out.squeeze().tolist()}
        return action


class DirectAction(Controller):
    """
    Loads the standard net that computes directly the action (wheel velocities)
    """

    def __init__(self, model_path: pathlib.Path):
        self.cnn = src.networks.InitialNet()
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

        self.cnn = src.networks.ResnetController()
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
        self.disturb_model = src.networks.BasicConvRNN()
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


class IntentionController(Controller):

    def __init__(self, num_bins: int, limits: Tuple[float, float], checkpoint_path: pathlib.Path, device="cuda:0"):
        assert len(limits) == 2 and limits[0] < limits[1]
        low, up = limits
        diff = float(up )
        self.idx_to_omega = [low + diff / 2.0 / num_bins + i * diff / num_bins for i in range(num_bins)]
        self.cnn = src.networks.DiscreteActionNet(num_bins)
        self.cnn.load_state_dict(torch.load(checkpoint_path.as_posix(), map_location=device))
        self.softmax = torch.nn.Softmax()

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((120, 160)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x.unsqueeze(0))
        ])

        print(self.idx_to_omega)

    def step(self, img: np.ndarray):
        out = self.transform(img)
        with torch.no_grad():
            out = self.cnn(out)
            out = self.softmax(out)
        idx = np.argmax(out).item()
        omega = self.idx_to_omega[idx]

        return np.array(omega_to_wheels(omega, 0.2))
