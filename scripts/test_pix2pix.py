#!/usr/bin/env python3

import time
import argparse
import pathlib

import yaml

import PIL.Image
import torch
import torchvision
import numpy as np
import cv2

import src.params
import src.networks
import src.neural_style_transformer
import src.datasetv2

import sys

sys.path.insert(0, '/home/dominik/workspace/pix2pix')

import models.networks as pix2pix_networks

intrinsics = None


def _init_intrinsics():
    global intrinsics
    calibration = yaml.load(pathlib.Path("/home/dominik/workspace/duckietown/catkin_ws/src/00-infrastructure/duckietown"
                                         "/config/baseline/calibration/camera_intrinsic/ferrari.yaml").read_text())

    intrinsics = {
        'K': np.array(calibration['camera_matrix']['data']).reshape(3, 3),
        'D': np.array(calibration['distortion_coefficients']['data']).reshape(1, 5),
        'R': np.array(calibration['rectification_matrix']['data']).reshape(3, 3),
        'P': np.array(calibration['projection_matrix']['data']).reshape((3, 4)),
        'distortion_model': calibration['distortion_model'],
    }


def rectify(image):
    img_shape = image.shape
    height, width = 480, 640
    image_resized = cv2.resize(image, (width, height))
    rectified_image = np.zeros(np.shape(image_resized))
    mapx = np.ndarray(shape=(height, width, 1), dtype='float32')
    mapy = np.ndarray(shape=(height, width, 1), dtype='float32')
    mapx, mapy = cv2.initUndistortRectifyMap(intrinsics['K'],
                                             intrinsics['D'],
                                             intrinsics['R'],
                                             intrinsics['P'],
                                             (width, height), cv2.CV_32FC1, mapx, mapy)
    cv2.remap(image_resized, mapx, mapy, cv2.INTER_CUBIC, rectified_image)
    return cv2.resize(rectified_image, img_shape[:2])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--save_dir", help="where and if to store images with printed on result")
    parser.add_argument("--rectify", action="store_true")

    args = parser.parse_args()

    print("Loading network...")
    params = src.params.TrainParams("conf_files/basic_lanefollower.yaml",
                                   "base_lanefollower_gray_noise_onlymarks")
    net = src.networks.BasicLaneFollower(params)
    net.load()

    transform_net = pix2pix_networks.UnetGenerator(3, 3, 8)
    transform_net.cuda()
    state_dict = torch.load("/home/dominik/dataspace/models/pix2pix/randbackgradscale_aug/latest_net_G.pth", #"/home/dominik/dataspace/models/pix2pix/randbackgradscale/latest_net_G.pth",
                            map_location="cuda:0")
    for key in list(state_dict.keys()):
        if "num_batches_tracked" in key:
            del state_dict[key]
    transform_net.load_state_dict(state_dict)

    transform_pix = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        torchvision.transforms.Lambda(lambda x: x.unsqueeze(0).cuda())
    ])

    transform_cnn = torchvision.transforms.Compose([
        torchvision.transforms.Resize((120, 160)),
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x[:, 40:, :]),
        torchvision.transforms.Lambda(lambda x: x.unsqueeze(0).to(params.device))
    ])

    dataset = src.datasetv2.SingleImage(data_dir=pathlib.Path(args.data_dir), params=params)
    dataset.transform = transform_pix

    gt = []
    pred = []
    save_idx = 0

    with torch.no_grad():
        for img, cmd in dataset:
            # img = transform_pix(img_orig)
            img = transform_net(img)
            img = (img + 1) / 2.0 * 255.0
            img = img.clamp(0, 255).cpu().squeeze(0).numpy()
            img = img.transpose(1, 2, 0).astype("uint8")[..., [2, 1, 0]]
            img = PIL.Image.fromarray(img)
            img_tmp = img.copy()

            if args.rectify:
                img = rectify(np.array(img))
                img = PIL.Image.fromarray(img)
            img = transform_cnn(img)
            img = img.to(params.device)
            cmd_pred = net(img).cpu()
            gt.append(cmd.item())
            pred.append(cmd_pred.item())

            if args.save_dir:
                save_path = pathlib.Path(args.save_dir)
                assert save_path.is_dir()
                save_path /= "img_{}.jpg".format(save_idx)
                img = np.array(img_tmp)
                cv2.putText(img, "gt: {:1.2f}".format(gt[-1]), (2, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
                cv2.putText(img, "pe: {:1.2f}".format(pred[-1]), (2, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
                cv2.imwrite(save_path.as_posix(), img)
                save_idx += 1

    gt = np.array(gt)
    pred = np.array(pred)

    print("gt mean", np.mean(gt))
    print("pred mean", np.mean(pred))
    print("gt abs mean:", np.mean(np.abs(gt)))
    print("pred abs mean:", np.mean(np.abs(pred)))
    print(np.mean(np.abs(gt - pred)))


if __name__ == "__main__":
    main()
