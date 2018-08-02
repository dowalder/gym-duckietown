#!/usr/bin/env python3

import pathlib
import random
from typing import List

import numpy as np

from gym_duckietown.envs import GeneratorEnv
import src.controllers


def omega_to_wheels(omega, v, wheel_dist=0.1) -> List[float]:
    v_left = v + 0.5 * omega * wheel_dist
    v_right = v - 0.5 * omega * wheel_dist
    return [v_left, v_right]


class DiscreteActionFinder:
    """
    Finds the correct action from a discrete set by trying every action and choosing the one with the highest reward.
    """

    def __init__(self, delta_t=0.1, range=(-1.0, 1.0), resolution=201, v=0.5, wheel_dist=0.1):
        assert isinstance(range, tuple), "got {}".format(range.__class__)
        assert len(range) == 2, "got {}".format(len(range))
        assert isinstance(resolution, int), "got {}".format(resolution.__class__)
        assert resolution > 0, "got {}".format(resolution)

        self.possible_omega = np.linspace(range[0], range[1], resolution)
        self.left = v + 0.5 * self.possible_omega * wheel_dist
        self.right = v - 0.5 * self.possible_omega * wheel_dist
        self.delta_t = delta_t

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
            return best_action, best_omega


class CNNActionFinder(src.controllers.OmegaController):

    def __init__(self, model_path: pathlib.Path):
        super(CNNActionFinder, self).__init__(model_path=model_path)

    def find_action(self, env: GeneratorEnv):
        img = env.render_obs()
        angle = float(self.cnn(self._transform(img)))

        return np.array(omega_to_wheels(angle, self.v, self.wheel_dist)), angle


class RandomWalker:

    def __init__(self, v=0.2, max_omega=4.0, wheel_dist=0.1):
        self.max_omega = max_omega
        self.v = v
        self.wheel_dist = wheel_dist

    def find_action(self, env: GeneratorEnv):
        omega = (random.random() - 0.5) * 2 * self.max_omega
        return np.array(omega_to_wheels(omega, self.v, self.wheel_dist)), omega
