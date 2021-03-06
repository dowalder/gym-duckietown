#!/usr/bin/env python3

"""
Implementation of different torch datasets
"""

import pathlib
import PIL.Image
import yaml
import collections
from typing import Union, Tuple, Optional, Callable, Dict, List

import torch
import torch.utils.data
import torchvision.transforms as transforms
import PIL.Image
import numpy as np
import cv2

import src.params
import src.graphics
from src.options import choose


def choose_dataset(data_set: str, *args, **kwargs):
    options = {
        "SingleImage": SingleImage,
        "Sequence": Sequence
    }
    return choose(data_set, options, "DataSets", *args, **kwargs)


def get_transform(params: src.params.Params):

    crop = params.get("crop", default=False)
    normalize = params.get("normalize", default=False)
    color = params.get("color", no_raise=True)
    size = params.get("image_size", no_raise=True)
    size = src.graphics.size_from_string(size) if size is not None else (120, 160)

    transf = [transforms.Grayscale()] if color == "gray" else []
    if crop:
        transf.append(CropTransform(size))
    else:
        transf.append(transforms.Resize(size))

    transf.append(transforms.ToTensor())
    if normalize:
        transf.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    return transf


class Base(torch.utils.data.Dataset):

    def __init__(self, data_dir: pathlib.Path, params: Optional[src.params.TrainParams]=None):
        self.data_in_memory = []
        self.labels = []
        self.data_dir = data_dir
        self._len = None  # type: Optional[int]
        self.params = params
        self._init_data()

        if self._len is None:
            raise NotImplemented("The member _len has not been set for {}".format(self.__class__))

        if self.params.data_in_memory:
            for idx in range(self._len):
                self.data_in_memory.append(self._load_item(idx))

    def _init_data(self):
        raise NotImplemented()

    def _load_item(self, item):
        raise NotImplemented()

    def __len__(self):
        return self._len

    def __getitem__(self, item):
        if self.params.data_in_memory:
            return self.data_in_memory[item]
        return self._load_item(item)

    @staticmethod
    def collate_fn():
        return None


class CropTransform:

    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, img):
        width, height = img.size
        idx_from_top = height - self.size[0]
        idx_from_left = (width - self.size[1]) // 2
        return transforms.functional.crop(img, idx_from_top, idx_from_left, *self.size)


class SingleImage(Base):

    def _init_data(self):
        self.images = list(self.data_dir.glob("*.jpg")) + list(self.data_dir.glob("*.png"))
        self._len = len(self.images)
        self.labels = []
        for image in self.images:
            lbl_file = image.parent / "{}.txt".format(image.stem)
            if not lbl_file.is_file():
                raise IOError("Could not find the label file {}".format(lbl_file))
            self.labels.append(list(map(float, lbl_file.read_text().strip().split())))

        self.transform = transforms.Compose(get_transform(self.params))

    def _load_item(self, item):
        img = PIL.Image.open(self.images[item].as_posix())
        if img is None:
            raise IOError("Could not read the image {}".format(self.images[item]))
        return self.transform(img).to(self.params.device), torch.Tensor(self.labels[item]).to(self.params.device)


class Sequence(Base):

    def _init_data(self):
        color = self.params.get("color", no_raise=True)
        self.grayscale = color == "gray" if color is not None else False
        self.datapoints = self.params.get("datapoints", no_raise=False)

        self.seq_len = self.params.get("sequence_length", no_raise=False)

        size = self.params.get("image_size", no_raise=True)
        self.img_size = src.graphics.size_from_string(size) if size is not None else (120, 160)

        transf = [transforms.Grayscale()] if self.grayscale else []
        self.transform = transforms.Compose(transf + [
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
        ])

        self._sequences = []
        for path in self.data_dir.iterdir():
            if path.suffix == ".yaml":
                data = yaml.load(path.read_text())
                if len(data) < self.seq_len:
                    print("WARNING: seq_len is too long for this data sequence: {}".format(path))
                    continue
                for i in range(0, len(data), self.seq_len):
                    self._sequences.append(data[i:i + self.seq_len])

        if not self._sequences:
            raise ValueError("Empty dataset: {}".format(self.data_dir))
        self._len = len(self._sequences)

    def _load_item(self, item):
        seq = self._sequences[item]
        datapoint = {}
        if "imgs" in self.datapoints:
            channels = 1 if self.grayscale else 3
            imgs = torch.empty(size=(self.seq_len, channels, self.img_size[0], self.img_size[1]),
                               dtype=torch.float,
                               device=self.params.device)
            for idx, info in enumerate(seq):
                img = PIL.Image.open((self.data_dir / info["path"]).as_posix())
                if img is None:
                    raise RuntimeError("Could not find the image at: {}".format(info["path"]))

                imgs[idx, :, :, :] = self.transform(img)
            datapoint["imgs"] = imgs

        if "actions" in self.datapoints:
            actions = torch.empty(size=(self.seq_len, 2), dtype=torch.float, device=self.params.device)
            for idx, info in enumerate(seq):
                actions[idx, :] = torch.Tensor(info["action"])
            datapoint["actions"] = actions

        if "modifiers" in self.datapoints:
            modifiers = torch.empty(size=(self.seq_len, 2), dtype=torch.float, device=self.params.device)
            for idx, info in enumerate(seq):
                modifiers[idx, :] = torch.Tensor(info["modifier"])
            datapoint["modifiers"] = modifiers

        if "softmax" in self.datapoints:
            num_actions = self.params.get("num_actions", no_raise=False)
            softmax = torch.empty(size=(self.seq_len, num_actions), dtype=torch.float, device=self.params.device)
            for idx, info in enumerate(seq):
                softmax[idx, :] = torch.Tensor(info["softmax"])
            datapoint["softmax"] = softmax

        return datapoint
