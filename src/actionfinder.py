#!/usr/bin/env python3

"""
Actionfinder are used as controllers for when ground truth data is available. This means they have access to the
environment.
"""

import abc
import pathlib
import random
from typing import Optional, Dict, Any

import numpy as np

from gym_duckietown.envs import GeneratorEnv
import src.controllers
import src.params


class Result:

    def __init__(self, action: np.ndarray, info: Dict[str, Any]):
        self.action = action
        self.info = info


class Base(abc.ABC):

    @abc.abstractmethod
    def find_action(self, env: GeneratorEnv) -> Optional[Result]:
        pass


class BestDiscrete(Base):
    """
    Finds the correct action from a discrete set by trying every action and choosing the one with the highest reward.
    """

    def __init__(self, params: src.params.DataGen):
        low = params.get("action_min", no_raise=False)
        up = params.get("action_max", no_raise=False)
        assert low < up, "low: {}, up: {}".format(low, up)
        resolution = params.get("action_resolution", no_raise=False)
        assert 0 < resolution
        wheel_dist = params.get("wheel_dist", default=0.1)

        self.possible_omega = np.linspace(low, up, resolution)
        self.left = params.velocitiy + 0.5 * self.possible_omega * wheel_dist
        self.right = params.velocitiy - 0.5 * self.possible_omega * wheel_dist
        self.delta_t = params.delta_t

    def find_action(self, env: GeneratorEnv):
        """
        Finds the most rewarding action.

        :param env: simulation environment to test the actions
        :return:
        """
        best_action = None
        best_omega = None
        best_reward = -1e6
        for i, action in enumerate(zip(self.left, self.right)):
            action = np.array(action)
            reward = env.theoretical_step(action, self.delta_t)
            if reward > best_reward:
                best_action = action
                best_reward = reward
                best_omega = self.possible_omega[i]

        if best_reward <= -20:
            return None
        else:
            return Result(best_action, {"omega": best_omega})


class CNNOmega(src.controllers.OmegaController):

    def __init__(self, params: src.params.DataGen):
        network_name = params.get("network_name", no_raise=False)
        network_conf_file = params.get("network_conf_file", no_raise=False)
        network_params = src.params.TestParams(network_conf_file, network_name)

        super(CNNOmega, self).__init__(network_params)

    def find_action(self, env: GeneratorEnv):
        img = env.render_obs()
        angle = float(self.cnn(self._transform(img)))

        return Result(np.array(src.controllers.omega_to_wheels(angle, self.v, self.wheel_dist)), {"omega": angle})


class CNNDiscrete(src.controllers.DiscreteAction):

    def __init__(self, params: src.params.DataGen):
        network_name = params.get("network_name", no_raise=False)
        network_conf_file = params.get("network_conf_file", no_raise=False)
        network_params = src.params.TestParams(network_conf_file, network_name)

        super(CNNDiscrete, self).__init__(network_params)

    def find_action(self, env: GeneratorEnv):
        img = env.render_obs()
        action = self.step(img)

        return Result(action, self.last_step)


class RandomWalker(Base):

    def __init__(self, params: src.params.DataGen):
        self.v = params.velocitiy
        self.max_omega = params.get("max_omega", default=4.0)
        self.wheel_dist = params.get("wheel_dist", default=0.1)

    def find_action(self, env: GeneratorEnv):
        omega = (random.random() - 0.5) * 2 * self.max_omega
        return Result(np.array(src.controllers.omega_to_wheels(omega, self.v, self.wheel_dist)), {"omega": omega})
