#!/usr/bin/env python3
from pathlib import Path
from typing import Dict, Any

import yaml
import torch


class Params:
    
    def __init__(self, conf_file: str, name: str, train: bool):
        conf = yaml.load(Path(conf_file).read_text())
        check_conf(conf, train)

        self.model_path = Path(conf["model_path"]) / name
        self.model_path.mkdir(exist_ok=True)

        self.name = name
        self.device = torch.device(conf["device"])
        self.network = conf["network"]

        self._additional = conf["additional"]
        self._conf_file = Path(conf_file)
        self._conf = conf
    
    def get(self, param_name: str, no_raise=True, default=None) -> Any:
        if param_name not in self._additional:
            if no_raise:
                return default
            raise ValueError(
                "There is no '{}' loaded. Please add it to the conf file {}".format(param_name, self._conf_file))
        return self._additional[param_name]

    def conf_file_text(self):
        return self._conf_file.read_text()
    
    
class TestParams(Params):

    def __init__(self, conf_file: str, name: str):
        super(TestParams, self).__init__(conf_file, name, False)
        self.result_path = Path(self._conf["result_path"])


class TrainParams(Params):

    def __init__(self, conf_file: str, name: str):
        super(TrainParams, self).__init__(conf_file, name, True)

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
        

def check_conf(conf: Dict[str, Any], train=True):
    general = {
        "model_path": str,
        "device": str,
        "network": str,
        "additional": dict,
    }
    training = {
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
    }
    testing = {
        "result_path": str,
    } 
    general.update(training) if train else general.update(testing)
    for key, val in general.items():
        assert key in conf, "Missing key: {}".format(key)
        assert isinstance(conf[key], val), "Expected {} to be {}, but got {}".format(key, val, conf[key].__class__)
