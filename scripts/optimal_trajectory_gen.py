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
import src.actionfinder


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

    parser.add_argument("--map", default="udem1")
    parser.add_argument("--tgt_dir", "-t", required=True, help="where to store generated images")
    parser.add_argument("--only_road", action="store_true",
                        help="If set, an only road picture is generated for every ordinary image as well")
    parser.add_argument("--mod", action="store_true")
    parser.add_argument("--find_road", action="store_true",
                        help="first searches for the road before it starts recording")
    parser.add_argument("--perturb_factor", default=0.0, type=float)
    parser.add_argument("--disturb_chance", default=0.0, type=float)
    parser.add_argument("--delta_t", default=0.1, type=float)
    parser.add_argument("--num_seq", default=20, type=int)
    parser.add_argument("--num_imgs", default=50, type=int)
    parser.add_argument("--model_path",
                        help="if specified, a CNN controller is used instead of the optimal action finder")

    args = parser.parse_args()

    env = GeneratorEnv(map_name=args.map)
    if args.only_road:
        set_only_road()
        env_only_road = GeneratorEnv(map_name=args.map, domain_rand=False)
        unset_only_road()

    env.render()

    delta_t = args.delta_t
    save_path = pathlib.Path(args.tgt_dir)

    # The faster moving action finder is better to find a starting position (it is more stable to converge), but the
    # slower action finder provides better training data as it does move more smoothly around curves.
    action_finder_fast = src.actionfinder.DiscreteActionFinder(delta_t=delta_t, range=(-10.0, 10.0), resolution=50, v=0.5)
    if args.model_path is not None:
        controller = src.actionfinder.CNNActionFinder(model_path=pathlib.Path(args.model_path))
    else:
        controller = src.actionfinder.DiscreteActionFinder(delta_t=delta_t, range=(-4.0, 4.0), resolution=201, v=0.2)

    num_seq = 0
    while num_seq < args.num_seq:
        print("running on sequence: {}".format(num_seq))
        seq_dir = save_path / "seq_{0:05d}".format(num_seq)
        seq_dir.mkdir(exist_ok=True)
        obs = env.reset(perturb_factor=float(args.perturb_factor))

        if args.find_road:
            # find a starting position
            for _ in range(10):
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
        while num_img < args.num_imgs:
            img_path = seq_dir / "img_{0:05d}.jpg".format(num_img)
            only_road_pth = seq_dir / "only_road_{0:05d}.jpg".format(num_img)

            if random.random() < args.disturb_chance:
                omega = random.choice(np.linspace(-4, 4, 101))
                action = np.array([0.2 + 0.5 * omega * 0.1, 0.2 - 0.5 * omega * 0.1])
                env.step(action, delta_t)
                continue
            else:
                out = controller.find_action(env)
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

            save_img(img_path, obs)

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
