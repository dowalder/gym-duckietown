#!/usr/bin/env python3
from typing import Dict, Callable, Any

import torch.nn
import torch.optim

import src.networks
import src.datasetv2 as dataset
import src.losses
import src.controllers


def _choose(option: str, options: Dict[str, Callable], name: str,  *args, **kwargs) -> Any:

    for key, val in options.items():
        if key == option:
            return val(*args, **kwargs)
    raise ValueError("No option named {} in {}. Options are: {}".format(option, name, "|".join(sorted(options.keys()))))


def choose_network(network: str, *args, **kwargs):
    options = {
        "BasicLaneFollower": src.networks.BasicLaneFollower,
        "ActionEstimator": src.networks.ActionEstimator,
    }
    return _choose(network, options, "Networks", *args, **kwargs)


def choose_dataset(data_set: str, *args, **kwargs) -> dataset.Base:
    options = {
        "SingleImage": dataset.SingleImage,
    }
    return _choose(data_set, options, "DataSets", *args, **kwargs)


def choose_optimizer(optim: str, *args, **kwargs) -> Any:
    options = {
        "Adam": torch.optim.Adam,
    }
    return _choose(optim, options, "Optimizers", *args, **kwargs)


def choose_criterion(criterion: str, *args, **kwargs) -> Any:
    options = {
        "MSE": torch.nn.MSELoss,
        "CrossEntropy": torch.nn.CrossEntropyLoss,
        "LabelsToCrossEntropy": src.losses.LabelsToCrossEntropy,
        "DistributionLoss": src.losses.MySpecialMSELoss,
    }
    return _choose(criterion, options, "Criterions", *args, **kwargs)


def choose_controller(controller: str, *args, **kwargs):
    options = {
        "Omega": src.controllers.OmegaController,
        "DiscreteAction": src.controllers.DiscreteAction,
    }
    return _choose(controller, options, "Controllers", *args, **kwargs)
