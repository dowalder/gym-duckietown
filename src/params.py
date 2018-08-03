#!/usr/bin/env python3
from pathlib import Path
from typing import Dict, Any

import yaml
import torch


class Params:

    def __init__(self, conf_file: Path, name: str):
        conf = yaml.load(conf_file.read_text())
        check_conf(conf)

        self.train_path = Path(conf["data_path"]) / "train"
        self.test_path = Path(conf["data_path"]) / "test"
        self.model_path = Path(conf["model_path"]) / name
        assert self.train_path.is_dir(), "Not a directory: {}".format(self.train_path)
        assert self.test_path.is_dir(), "Not a directory: {}".format(self.test_path)

        self.test_interval = conf["test_interval"]
        self.train_interval = conf["train_interval"]
        self.save_interval = conf["save_interval"]
        self.num_epochs = conf["num_epochs"]

        self.name = name
        self.device = torch.device(conf["device"])
        self.network = conf["network"]
        self.cont = conf["continue"]
        self.data_in_memory = conf["data_in_memory"]

        self._additional = conf["additional"]
        self._conf_file = conf_file

    def get(self, param_name: str) -> Any:
        if param_name not in self._additional:
            raise ValueError(
                "There is no '{}' loaded. Please add it to the conf file {}".format(param_name, self._conf_file))
        return self._additional[param_name]


def check_conf(conf: Dict[str, Any]):
    required_fields = {
        "data_path": str,
        "model_path": str,
        "device": str,
        "test_interval": int,
        "train_interval": int,
        "save_interval": int,
        "network": str,
        "num_epochs": int,
        "continue": bool,
        "data_in_memory": bool,
        "additional": dict
    }
    for key, val in required_fields.items():
        assert key in conf, "Missing key: {}".format(key)
        assert isinstance(conf[key], val), "Expected {} to be {}, but got {}".format(key, val, conf[key].__class__)
