#!/usr/bin/env python3

import argparse
import io
import zmq
import PIL.Image
import re

import torch
import torchvision

import src.params
import src.networks
import src.neural_style_transformer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--use_neural_style", default=False)
    parser.add_argument("--on_rudolf", default=False)

    args = parser.parse_args()

    context = zmq.Context()

    if args.on_rudolf:
        sub_topic = "tcp://localhost:54321"
        pub_topic = "ipc://localhost:12345"
    else:
        sub_topic = "ipc:///home/dominik/tmp/image.zeromq"
        pub_topic = "ipc:///home/dominik/tmp/command.zeromq"

    subscriber = context.socket(zmq.SUB)
    if args.on_rudolf:
        subscriber.bind(sub_topic)
    else:
        subscriber.connect(sub_topic)
        
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

    publisher = context.socket(zmq.PUB)
    publisher.bind(pub_topic)

    print("Loading network...")
    params = src.params.TestParams("/home/dominik/workspace/gym-duckietown/conf_files/basic_lanefollower.yaml",
                                   "domain_rand_gray_80_160")
    net = src.networks.BasicLaneFollower(params)
    net.load()

    transform_net = None
    if args.use_neural_style:
        transform_net = src.neural_style_transformer.TransformerNet()

        state_dict = torch.load(args.model)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        transform_net.load_state_dict(state_dict)
        transform_net.to("cuda:0")
        transform_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x.mul(255))
        ])

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((80, 160)),
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.unsqueeze(0))
    ])

    print("Waiting for messages...")
    while True:
        message = subscriber.recv()
        print("Received message")
        img = PIL.Image.open(io.BytesIO(message))
        img = transform(img)
        cmd = net(img).item()
        publisher.send_string(str(cmd))


if __name__ == "__main__":
    main()
