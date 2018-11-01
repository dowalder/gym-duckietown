#!/usr/bin/env python3

import time
import argparse
import io
import pathlib

import zmq
import PIL.Image
import yaml

import torch
import torchvision
import numpy as np
import cv2

import src.params
import src.networks
import src.neural_style_transformer

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


# intrinsics = {
#     "K": np.array([307.7379294605756, 0, 329.692367951685, 0, 314.9827773443905, 244.4605588877848, 0, 0, 1
#                    ]).reshape(3, 3),
#     "D": np.array([-0.2565888993516047, 0.04481160508242147, -0.00505275149956019, 0.001308569367976665, 0
#                    ]).reshape(1, 5),
#     "R": np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0
#                    ]).reshape(3, 3),
#     "P": np.array([210.1107940673828, 0, 327.2577820024981, 0, 0, 253.8408660888672, 239.9969353923052, 0, 0, 0, 1, 0
#                    ]).reshape(3, 4)
# }


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

    parser.add_argument("--mode", choices=["img_to_cmd", "img_to_img", "neural_img_to_cmd", "delay"], required=True)
    parser.add_argument("--rectify", action="store_true")
    parser.add_argument("--use_pix2pix", action="store_true")
    parser.add_argument("--delay_ms", default=20)

    args = parser.parse_args()

    context = zmq.Context()

    img_in_topic = "ipc:///home/dominik/tmp/image.zeromq"
    img_out_topic = "ipc:///home/dominik/tmp/image_corrected.zeromq"
    cmd_topic = "ipc:///home/dominik/tmp/command.zeromq"

    image_sub = context.socket(zmq.SUB)
    image_sub.connect(img_in_topic)
    image_sub.setsockopt_string(zmq.SUBSCRIBE, "")

    cmd_pub = context.socket(zmq.PUB)
    cmd_pub.bind(cmd_topic)

    img_pub = context.socket(zmq.PUB)
    img_pub.bind(img_out_topic)

    print("Loading network...")
    params = src.params.TestParams("conf_files/basic_lanefollower.yaml",
                                   "domain_rand_onlymarks_80_160_color")
    net = src.networks.BasicLaneFollower(params)
    net.load()

    if args.use_pix2pix:
        import sys
        sys.path.insert(0, '/home/dominik/workspace/pix2pix')

        import models.networks as pix2pix_networks
        transform_net = pix2pix_networks.UnetGenerator(3, 3, 8)
        transform_net.cuda()
        state_dict = torch.load("/home/dominik/dataspace/models/pix2pix/randbackgradscale_aug/latest_net_G.pth", #"/home/dominik/dataspace/models/pix2pix/randbackgradscale/latest_net_G.pth",
                                map_location="cuda:0")
        for k in list(state_dict.keys()):
            if "num_batches_tracked" in k:
                del state_dict[k]
        transform_net.load_state_dict(state_dict)


        transform_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            torchvision.transforms.Lambda(lambda x: x.unsqueeze(0).cuda())
        ])
    else:
        transform_net = src.neural_style_transformer.TransformerNet()

        transform_net.load(
            "/home/dominik/dataspace/models/neural_style/style3/epoch_2_Wed_Aug_22_22:38:39_2018_100000.0_10000000000.0.model")
        transform_net.to("cuda:0")

        transform_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x.mul(255).unsqueeze(0).to(params.device))
        ])

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((120, 160)),
        # torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x[:, 40:, :]),
        torchvision.transforms.Lambda(lambda x: x.unsqueeze(0).to(params.device))
    ])

    path = pathlib.Path("/home/dominik/tmp/out/")
    idx = 0

    print("Waiting for messages...")
    while True:
        message = image_sub.recv()
        print("Received message")
        if args.mode == "delay":
            time.sleep(args.delay_ms / 1e3)
            img_pub.send(message)
        with torch.no_grad():
            t = time.time()
            img = PIL.Image.open(io.BytesIO(message))

            idx += 1
            # img.save((path / "{}_original.jpg".format(idx)).as_posix())

            if args.mode in ["neural_img_to_cmd", "img_to_img"]:
                img = transform_transform(img)
                img = transform_net(img)
                if args.use_pix2pix:
                    img = (img + 1) / 2.0 * 255.0
                img = img.clamp(0, 255).cpu().squeeze(0).numpy()
                img = img.transpose(1, 2, 0).astype("uint8")
                img = PIL.Image.fromarray(img)

            if args.mode in ["neural_img_to_cmd", "img_to_cmd"]:
                if args.rectify:
                    img = rectify(np.array(img))
                    img = PIL.Image.fromarray(img)
                img = transform(img)
                img = img.to(params.device)
                cmd = net(img).cpu().item()
                cmd_pub.send_string(str(cmd))

            if args.mode == "img_to_img":
                jpg_bytes = io.BytesIO()
                img.save(jpg_bytes, format="JPEG")
                # img.save((path / "{}_augmented.jpg".format(idx)))

                img_pub.send(jpg_bytes.getvalue())


if __name__ == "__main__":
    main()
