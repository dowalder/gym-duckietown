#!/usr/bin/env python3

"""
This script generates trajectories from which a lane follower can be learned.
"""

import argparse
import copy
import pathlib
import random

import yaml
import cv2
import torch
import numpy as np

from gym_duckietown.envs import GeneratorEnv, set_only_road, unset_only_road

import src.actionfinder
import src.params
import src.options


def save_img(path: pathlib.Path, img):
    """
    Saves a RGB image wiht opencv. ATTENTION: if the image was loaded with opencv, it most likely is a BGR image and it
    can be stored directly with cv2.imwrite.
    :param path:
    :param img:
    """
    img = img[:, :, [2, 1, 0]]
    cv2.imwrite(path.as_posix(), img)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--conf", "-c", required=True, help="name of the configuration file")
    parser.add_argument("--name", "-n", required=True, help="name of the dataset to be generated")

    args = parser.parse_args()
    params = src.params.DataGen(args.conf, args.name)

    env = GeneratorEnv(map_name=params.map)
    if params.only_road:
        set_only_road()
        env_only_road = GeneratorEnv(map_name=params.map, domain_rand=False)
        unset_only_road()

    env.render()

    # The faster moving action finder is better to find a starting position (it is more stable to converge), but the
    # slower action finder provides better training data as it does move more smoothly around curves.
    if params.find_road:
        # Somehow deepcopy cant copy a torch.device
        params.device = str(params.device)
        params_copy = copy.deepcopy(params)
        params.device = torch.device(params.device)

        params_copy.velocitiy = 0.5
        params_copy.additional["action_min"] = -10.0
        params_copy.additional["action_max"] = 10.0
        params_copy.additional["action_resolution"] = 50
        action_finder_fast = src.actionfinder.BestDiscrete(params_copy)

    controller = src.options.choose_actionfinder(params.actionfinder, params)

    num_seq = 0
    while num_seq < params.num_seq:
        print("running on sequence: {}".format(num_seq))
        seq_dir = params.data_path / "seq_{0:05d}".format(num_seq)
        seq_dir.mkdir(exist_ok=True, parents=True)
        obs = env.reset(perturb_factor=float(params.perturb_factor))

        if params.find_road:
            # find a starting position
            for _ in range(10):
                out = action_finder_fast.find_action(env)
                if out is None:
                    break
                obs, _, _, _ = env.step(out.action, params.delta_t)
            if params.only_road:
                env_only_road.cur_pos = env.cur_pos
                env_only_road.cur_angle = env.cur_angle
                obs_only_road = env_only_road.render_obs(only_road=True)

        info = []

        if params.use_modifier:
            modifier = 0.1 + np.random.random(2) * 2.0
        else:
            modifier = np.ones((2,))

        num_img = 0
        while num_img < params.img_per_seq:
            img_path = seq_dir / "img_{0:05d}.jpg".format(num_img)
            only_road_pth = seq_dir / "only_road_{0:05d}.jpg".format(num_img)

            if random.random() < params.disturb_chance:
                omega = random.choice(np.linspace(-4, 4, 101))
                action = np.array([0.2 + 0.5 * omega * 0.1, 0.2 - 0.5 * omega * 0.1])
                env.step(action, params.delta_t)
                continue
            else:
                out = controller.find_action(env)
                if out is None:
                    info = None
                    break

            # action belongs to the old image
            if params.only_road:
                save_img(only_road_pth, obs_only_road)

            save_img(img_path, obs)

            obs, reward, _, _ = env.step(out.action * modifier, params.delta_t)
            if params.only_road:
                env_only_road.cur_pos = env.cur_pos
                env_only_road.cur_angle = env.cur_angle
                obs_only_road = env_only_road.render_obs(only_road=True)

            info.append({"path": img_path.relative_to(params.data_path).as_posix(),
                         "action": out.action.tolist(),
                         "reward": float(reward),
                         "modifier": modifier.tolist(),
                         "delta_t": params.delta_t,
                         **out.info})

            if params.only_road:
                info[-1]["only_road_pth"] = only_road_pth.relative_to(params.data_path).as_posix()

            num_img += 1

        info_path = seq_dir.parent / (seq_dir.stem + ".yaml")
        info_path.write_text(yaml.dump(info))

        num_seq += 1


if __name__ == "__main__":
    main()
