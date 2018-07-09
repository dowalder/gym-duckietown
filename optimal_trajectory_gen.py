#!/usr/bin/env python3

import argparse
import pathlib
import random

import yaml
import cv2

import numpy as np

from gym_duckietown.envs import GeneratorEnv, Parameters


def save_img(path: pathlib.Path, img):
    img = np.flip(img, 0)
    img = img[:, :, [2, 1, 0]]

    cv2.imwrite(path.as_posix(), img)


class ActionFinder:

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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--map", default="udem1")
    parser.add_argument("--tgt_dir", "-t", required=True, help="where to store generated images")

    args = parser.parse_args()

    env = GeneratorEnv(map_name=args.map)

    env.render()

    delta_t = 0.033
    num_sequences = 3
    num_img_per_seq = 6000
    save_path = pathlib.Path(args.tgt_dir)

    action_finder_fast = ActionFinder(delta_t=delta_t, range=(-10.0, 10.0), resolution=50, v=0.5)
    action_finder_slow = ActionFinder(delta_t=delta_t, range=(-4.0, 4.0), resolution=201, v=0.2)

    num_seq = 0
    while num_seq < num_sequences:
        print("running on sequence: {}".format(num_seq))
        seq_dir = save_path / "seq_{0:05d}".format(num_seq)
        seq_dir.mkdir(exist_ok=True)
        obs = env.reset(perturb_factor=0.0)

        # find a starting position
        for _ in range(50):
            out = action_finder_fast.find_action(env)
            if out is None:
                break
            else:
                action, _ = out
            obs, _, _, _ = env.step(action, delta_t)

        info = []

        disturb_chance = 0.0

        num_img = 0
        while num_img < num_img_per_seq:
            img_path = seq_dir / "img_{0:05d}.jpg".format(num_img)

            if random.random() < disturb_chance:
                omega = random.choice(np.linspace(-10, 10, 101))
                action = np.array([1.0 + 0.5 * omega * 0.1, 1.0 - 0.5 * omega * 0.1])
                env.step(action, delta_t)
                continue
            else:
                out = action_finder_slow.find_action(env)
                if out is None:
                    info = None
                    break
                else:
                    action, omega = out

            save_img(img_path, obs)  # action belongs to the old image
            obs, reward, _, _ = env.step(action, delta_t)

            info.append({"path": img_path.relative_to(save_path).as_posix(),
                         "action": action.tolist(),
                         "omega": float(omega),
                         "reward": float(reward),
                         "delta_t": delta_t})
            num_img += 1

        if info is None:
            continue

        info_path = seq_dir.parent / (seq_dir.stem + ".yaml")
        info_path.write_text(yaml.dump(info))

        num_seq += 1


if __name__ == "__main__":
    main()
