#!/usr/bin/env python3

"""
This script generates trajectories from which a lane follower can be learned.
"""

import argparse
import pathlib
import random

import yaml
import cv2

import numpy as np

from gym_duckietown.envs import GeneratorEnv, set_only_road, unset_only_road


def save_img(path: pathlib.Path, img):
    """
    Saves a RGB image wiht opencv. ATTENTION: if the image was loaded with opencv, it most likely is a BGR image and it
    can be stored directly with cv2.imwrite.
    :param path:
    :param img:
    """
    img = img[:, :, [2, 1, 0]]
    cv2.imwrite(path.as_posix(), img)


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


class PurePursuitController:
    """
    NOT WORKING
    """

    def __init__(self, vel=0.4, lookahead=0.1, wheel_dist=0.1):
        self.v = vel
        self.wheel_dist = wheel_dist
        self.lookahead = lookahead

    def step(self, env: GeneratorEnv):
        x_curr, _, z_curr = env.cur_pos
        angle = env.cur_angle

        x_goal, _, z_goal = env.lookahead_point(lookahead_dist=self.lookahead)
        r_goal = -(x_goal - x_curr) * np.sin(angle) + (z_goal - z_curr) * np.cos(angle)
        curvature = 2 * r_goal / (self.lookahead ** 2)
        omega = curvature * self.v

        vel_left = self.v + 0.5 * omega * self.wheel_dist
        vel_right = self.v - 0.5 * omega * self.wheel_dist
        return np.array([vel_left, vel_right]), omega


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--map", default="udem1")
    parser.add_argument("--tgt_dir", "-t", required=True, help="where to store generated images")
    parser.add_argument("--only_road", action="store_true",
                        help="If set, an only road picture is generated for every ordinary image as well")
    parser.add_argument("--mod", action="store_true")
    parser.add_argument("--perturb_factor", default=0.0)
    parser.add_argument("--disturb_chance", default=0.0)

    args = parser.parse_args()

    env = GeneratorEnv(map_name=args.map)
    if args.only_road:
        set_only_road()
        env_only_road = GeneratorEnv(map_name=args.map, domain_rand=False)
        unset_only_road()

    env.render()

    delta_t = 0.1
    num_sequences = 100
    num_img_per_seq = 100
    save_path = pathlib.Path(args.tgt_dir)

    # The faster moving action finder is better to find a starting position (it is more stable to converge), but the
    # slower action finder provides better training data as it does move more smoothly around curves.
    action_finder_fast = DiscreteActionFinder(delta_t=delta_t, range=(-10.0, 10.0), resolution=50, v=0.5)
    action_finder_slow = DiscreteActionFinder(delta_t=delta_t, range=(-4.0, 4.0), resolution=201, v=0.2)

    num_seq = 0
    while num_seq < num_sequences:
        print("running on sequence: {}".format(num_seq))
        seq_dir = save_path / "seq_{0:05d}".format(num_seq)
        seq_dir.mkdir(exist_ok=True)
        obs = env.reset(perturb_factor=float(args.perturb_factor))

        # find a starting position
        for _ in range(50):
            out = action_finder_fast.find_action(env)
            if out is None:
                break
            else:
                action, _ = out
            obs, _, _, _ = env.step(action, delta_t)
        if args.only_road:
            env_only_road.cur_pos = env.cur_pos
            env_only_road.cur_angle = env.cur_angle
            obs_only_road = env_only_road.render_obs(only_road=True)

        info = []

        if args.mod:
            modifier = 0.1 + np.random.random(2) * 2.0
        else:
            modifier = np.ones((2,))

        num_img = 0
        while num_img < num_img_per_seq:
            img_path = seq_dir / "img_{0:05d}.jpg".format(num_img)
            only_road_pth = seq_dir / "only_road_{0:05d}.jpg".format(num_img)

            if random.random() < args.disturb_chance:
                omega = random.choice(np.linspace(-4, 4, 101))
                action = np.array([0.2 + 0.5 * omega * 0.1, 0.2 - 0.5 * omega * 0.1])
                env.step(action, delta_t)
                continue
            else:
                out = action_finder_slow.find_action(env)
                if out is None:
                    info = None
                    break
                else:
                    action, omega = out

            if info is None:
                continue

            # action belongs to the old image
            if args.only_road:
                save_img(only_road_pth, obs_only_road)

            obs, reward, _, _ = env.step(action * modifier, delta_t)
            if args.only_road:
                env_only_road.cur_pos = env.cur_pos
                env_only_road.cur_angle = env.cur_angle
                obs_only_road = env_only_road.render_obs(only_road=True)

            info.append({"path": img_path.relative_to(save_path).as_posix(),
                         "action": action.tolist(),
                         "omega": float(omega),
                         "reward": float(reward),
                         "modifier": modifier.tolist(),
                         "delta_t": delta_t})

            if args.only_road:
                info[-1]["only_road_pth"] = only_road_pth.relative_to(save_path).as_posix()

            num_img += 1

        info_path = seq_dir.parent / (seq_dir.stem + ".yaml")
        info_path.write_text(yaml.dump(info))

        num_seq += 1


if __name__ == "__main__":
    main()
