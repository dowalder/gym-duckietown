#!/usr/bin/env python3

import argparse
import time
from pathlib import Path
from typing import Dict, Any

import yaml
import torch
import torch.utils.data

import src.params
import src.networks
import src.datasetv2 as dataset


def load_network(params: src.params.Params):
    if params.network == "BasicLaneFollower":
        return src.networks.BasicLaneFollower(params)
    else:
        raise ValueError("Unknown network: {}".format(params.network))


def load_datasets(params: src.params.Params):
    if params.data_set == "singleimage":
        train_loader = torch.utils.data.DataLoader(dataset.SingleImage(params.train_path, params),
                                                   batch_size=params.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset.SingleImage(params.test_path, params),
                                                  batch_size=params.batch_size, shuffle=True)
        return train_loader, test_loader
    else:
        raise ValueError("Unknown dataset: {}".format(params.data_set))


def train(net: src.networks.BaseModel, train_set, test_set):
    running_loss = 0
    for epoch in range(net.params.num_epochs):
        t = time.time()

        for samples, labels in train_set:
            running_loss += net.training_step(samples, labels)

            if net.iteration % net.params.train_interval == 0:
                net.say("[{}][{}]: train: {}".format(
                    epoch, net.iteration, running_loss / net.params.train_interval))
                running_loss = 0

            if net.iteration % net.params.test_interval == 0:
                error = []
                for test_samples, test_labels in test_set:
                    error.append(net.testing_step(test_samples, test_labels))
                error = sum(error) / len(error)
                net.say("\ttest: {}".format(error))

            if net.iteration % net.params.save_interval == 0:
                net.save(verbose=True)

        net.say("epoch {} finished. took {:4.2f}s".format(epoch, time.time() - t))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--conf", "-c", required=True, type=str, help="conf file containing options")
    parser.add_argument("--name", "-n", required=True, type=str, help="give a name")

    args = parser.parse_args()
    params = src.params.Params(args.conf, args.name)

    net = load_network(params)
    print(net)
    net.init()
    net.say(50 * "=")
    net.say("Starting training named {}:\nnetwork: {}\ndataset: {}\n\n{}\n\n".format(
        args.name, params.network, params.train_path, str(net)))
    net.say("moving network to device {}".format(params.device))
    net.to(params.device)
    net.say("loading data sets")
    train_set, test_set = load_datasets(params)

    net.say("starting training")
    train(net, train_set, test_set)


if __name__ == "__main__":
    main()
