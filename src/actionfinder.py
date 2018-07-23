#!/usr/bin/env python3

import pathlib

import numpy as np

from gym_duckietown.envs import GeneratorEnv
import src.controllers


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
        vel_left = (self.v + 0.5 * angle * self.wheel_dist)
        vel_right = (self.v - 0.5 * angle * self.wheel_dist)

        return np.array([vel_left, vel_right]), angle
