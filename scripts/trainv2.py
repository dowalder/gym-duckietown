#!/usr/bin/env python3

"""
General training scripts. For a choice of datasets, networks, etc checkout out the source files in ./src
"""

import argparse
import time
import pathlib

import torch
import torch.utils.data

import src.datasetv2
import src.params
import src.networks
import src.options


def load_dataloader(dataset_path: pathlib.Path, params: src.params.TrainParams):
    dataset = src.datasetv2.choose_dataset(params.data_set, dataset_path, params)
    collate_fn = dataset.collate_fn()
    if collate_fn is None:
        return torch.utils.data.DataLoader(dataset, batch_size=params.batch_size, shuffle=True)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=params.batch_size, shuffle=True, collate_fn=collate_fn)


def load_datasets(params: src.params.TrainParams):
    train_loader = load_dataloader(params.train_path, params)
    test_loader = load_dataloader(params.test_path, params)
    return train_loader, test_loader


def train(net: src.networks.BaseModel, train_set, test_set):
    running_loss = 0
    for epoch in range(net.params.num_epochs):
        t = time.time()

        for data in train_set:
            running_loss += net.training_step(data)

            if net.iteration % net.params.train_interval == 0:
                net.say("[{}][{}]: train: {}".format(
                    epoch, net.iteration, running_loss / net.params.train_interval))
                running_loss = 0

            if net.iteration % net.params.test_interval == 0:
                error = []
                for test_data in test_set:
                    error.append(net.testing_step(test_data))
                error = sum(error) / len(error)
                net.say("\ttest: {}".format(error))

            if net.iteration % net.params.save_interval == 0:
                net.save(verbose=True)

        net.say("finished epoch {}. took {:4.2f}s".format(epoch + 1, time.time() - t))
        net.say("-" * 50)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--conf", "-c", required=True, type=str, help="conf file containing options")
    parser.add_argument("--name", "-n", required=True, type=str, help="give a name")

    args = parser.parse_args()
    params = src.params.TrainParams(args.conf, args.name)

    net = src.networks.choose_network(params.network, params)
    net.init_training()
    net.say(50 * "=")
    net.say("Starting training named {}".format(args.name))
    net.say("network: {}".format(params.network))
    net.say("dataset: {}, path: {}".format(params.data_set, params.train_path))
    net.say("\n{}\n".format(str(net)))
    net.say("moving network to device {}".format(params.device))
    net.to(params.device)
    net.say("loading data sets")
    train_set, test_set = load_datasets(params)

    net.say("starting training")
    train(net, train_set, test_set)


if __name__ == "__main__":
    main()
