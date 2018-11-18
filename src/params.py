#!/usr/bin/env python3

"""
Different parameter classes for testing, training and data generation.
"""

from pathlib import Path
from typing import Dict, Any

import yaml
import torch


class Params:
    
    def __init__(self, conf_file: str, name: str, mode: str):
        conf = yaml.load(Path(conf_file).read_text())
        check_conf(conf, mode)

        self.model_path = Path(conf["model_path"]) / name
        if not conf["model_path"] == "none":
            self.model_path.mkdir(exist_ok=True)

        self.name = name
        self.device = torch.device(conf["device"])
        self.network = conf["network"]

        self.additional = conf["additional"]
        self._conf_file = Path(conf_file)
        self._conf = conf
    
    def get(self, param_name: str, no_raise=True, default=None) -> Any:
        if param_name not in self.additional:
            if no_raise:
                return default
            raise ValueError(
                "There is no '{}' loaded. Please add it to the conf file {}".format(param_name, self._conf_file))
        return self.additional[param_name]

    def conf_file_text(self):
        return self._conf_file.read_text()
    
    
class TestParams(Params):

    def __init__(self, conf_file: str, name: str):
        super(TestParams, self).__init__(conf_file, name, "testing")
        self.result_path = Path(self._conf["result_path"])


class TrainParams(Params):

    def __init__(self, conf_file: str, name: str):
        super(TrainParams, self).__init__(conf_file, name, "training")

        self.train_path = Path(self._conf["data_path"]) / "train"
        self.test_path = Path(self._conf["data_path"]) / "test"
        assert self.train_path.is_dir(), "Not a directory: {}".format(self.train_path)
        assert self.test_path.is_dir(), "Not a directory: {}".format(self.test_path)
        
        self.test_interval = self._conf["test_interval"]
        self.train_interval = self._conf["train_interval"]
        self.save_interval = self._conf["save_interval"]
        self.num_epochs = self._conf["num_epochs"]
        
        self.cont = self._conf["continue"]
        self.data_set = self._conf["data_set"]
        self.batch_size = self._conf["batch_size"]
        self.data_in_memory = self._conf["data_in_memory"]
        
        self.optimizer = self._conf["optimizer"]["name"]
        self.optimizer_params = self._conf["optimizer"]["params"] if "params" in self._conf["optimizer"] else {}
        assert isinstance(self.optimizer_params, dict)

        self.criterion = self._conf["criterion"]["name"]
        self.criterion_params = self._conf["criterion"]["params"] if "params" in self._conf["criterion"] else {}
        assert isinstance(self.criterion_params, dict)
        

class DataGen(Params):

    def __init__(self, conf_file: str, name: str):
        super(DataGen, self).__init__(conf_file, name, "datagen")

        self.data_path = Path(self._conf["data_path"]) / name
        self.data_path.mkdir(exist_ok=True, parents=True)

        self.map = self._conf["map"]
        self.only_road = self._conf["only_road"]
        self.use_modifier = self._conf["use_modifier"]
        self.find_road = self._conf["find_road"]
        self.perturb_factor = self._conf["perturb_factor"]
        self.disturb_chance = self._conf["disturb_chance"]
        self.delta_t = self._conf["delta_t"]
        self.velocitiy = self._conf["velocity"]
        self.num_seq = self._conf["num_sequences"]
        self.img_per_seq = self._conf["imgs_per_sequence"]
        self.actionfinder = self._conf["action_finder"]


def check_conf(conf: Dict[str, Any], mode):
    general = {
        "model_path": str,
        "device": str,
        "network": str,
        "additional": dict,
    }
    if mode == "training":
        general.update({
            "test_interval": int,
            "train_interval": int,
            "save_interval": int,
            "num_epochs": int,
            "continue": bool,
            "data_in_memory": bool,
            "data_set": str,
            "optimizer": dict,
            "criterion": dict,
            "batch_size": int,
            "data_path": str,
        })
    elif mode == "testing":
        general.update({
            "result_path": str,
        })
    elif mode == "datagen":
        general.update({
            "map": str,
            "data_path": str,
            "only_road": bool,
            "use_modifier": bool,
            "find_road": bool,
            "perturb_factor": float,
            "disturb_chance": float,
            "delta_t": float,
            "velocity": float,
            "num_sequences": int,
            "imgs_per_sequence": int,
            "action_finder": str,
        })
    else:
        raise ValueError("Invalid mode: {}".format(mode))

    for key, val in general.items():
        assert key in conf, "Missing key: {}".format(key)
        assert isinstance(conf[key], val), "Expected {} to be {}, but got {}".format(key, val, conf[key].__class__)
