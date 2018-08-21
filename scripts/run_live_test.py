#!/usr/bin/env python3

"""
Runs a test for a defined number of runs of defined length. The output are consecutive images of the controller
navigating the simulation.
"""

import argparse
import pathlib
from typing import Any

import yaml
import cv2
import numpy as np

from gym_duckietown.envs import GeneratorEnv, unset_only_road, set_only_road
import src.controllers
import src.graphics
import src.options
import src.params


def save_img(path: pathlib.Path, img):
    img = img[:, :, [2, 1, 0]]

    cv2.imwrite(path.as_posix(), img)


# def get_controller(args: Any) -> src.controllers.Controller:
#     """
#     Controller factory.
#     :param args:
#     :return:
#     """
#     controller_args = yaml.load(pathlib.Path(args.args).read_text())
#     if args.controller == "omega":
#         controller = src.controllers.OmegaController(pathlib.Path(controller_args["model_path"]))
#     elif args.controller == "direct_action":
#         controller = src.controllers.DirectAction(pathlib.Path(controller_args["model_path"]))
#     elif args.controller == "resnet":
#         controller = src.controllers.DirectActionResnet(pathlib.Path(controller_args["model_path"]))
#     elif args.controller == "disturb":
#         controller = src.controllers.FixDisturbController(pathlib.Path(controller_args["model_path"]),
#                                                           pathlib.Path(
#                                                               "/home/dominik/dataspace/models/rnn_randomwalk_forward/"
#                                                               "run5_step_40000.pth"))
#     elif args.controller == "intention":
#         controller = src.controllers.DiscreteAction(11, (-4.0, 4.0), pathlib.Path(controller_args["model_path"]))
#     else:
#         raise ValueError("Unknown controller: {}".format(args.controller))
#
#     return controller


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--conf", "-c", required=True, help="Configuration file to use")
    parser.add_argument("--name", "-n", default="default_name", help="name of the experiment to use")
    parser.add_argument("--map", default="udem1")
    parser.add_argument("--num_seq", default=3, type=int)
    parser.add_argument("--num_img", default=200, type=int)
    parser.add_argument("--augment_action", default="1,1", type=str)
    parser.add_argument("--only_road", action="store_true")

    args = parser.parse_args()
    params = src.params.TestParams(args.conf, args.name)

    controller = src.options.choose_controller(params.get("controller", no_raise=False), params)

    if args.only_road:
        set_only_road()
    env = GeneratorEnv(map_name=args.map)

    env.render()

    num_sequences = args.num_seq
    num_img_per_seq = args.num_img

    action_mult = np.array(list(map(float, args.augment_action.split(","))))
    assert action_mult.shape == (2,), "--augment_action must be two floats separated by a comma, e.g. '0.4,0.6'"

    for num_seq in range(num_sequences):
        print("running on sequence: {}".format(num_seq))
        seq_dir = params.result_path / "seq_{0:05d}".format(num_seq)
        seq_dir.mkdir(exist_ok=True)
        obs = env.reset(perturb_factor=0.0)
        img_path = seq_dir / "img_00000.jpg"

        delta_t = 0.1

        save_img(img_path, obs)

        info = []

        for num_img in range(num_img_per_seq):
            img_path = seq_dir / "img_{0:05d}.jpg".format(num_img + 1)

            action = controller.step(obs) * action_mult
            obs, reward, _, _ = env.step(action, delta_t, only_road=args.only_road)
            if args.only_road:
                obs = src.graphics.apply_color_filter(obs)
            save_img(img_path, obs)

            info.append({"path": img_path.as_posix(),
                         "action": action.tolist(),
                         "reward": float(reward),
                         "delta_t": delta_t})

        info_path = seq_dir.parent / (seq_dir.stem + ".yaml")
        info_path.write_text(yaml.dump(info))


if __name__ == "__main__":
    main()
