#!/usr/bin/env python3
import os
import pathlib
import PIL.Image
import yaml
import csv
from typing import Union, Tuple, Optional

import torch
import torch.utils.data
import torchvision.transforms as transforms
import PIL.Image

import numpy as np
import cv2

# ============= V1 ===================

class DataSet(torch.utils.data.Dataset):
    """
    A simple data set to use when you have a single folder containing the images and for every image a .txt file in the
    same directory with the same name containing a single line with space separated values as labels.
    """

    def __init__(self, data_dir):
        self.images = [os.path.join(data_dir, img_file) for img_file in os.listdir(data_dir)
                       if (img_file.endswith(".jpg") or img_file.endswith(".png"))]
        self.labels = []
        for image in self.images:
            stem, _ = os.path.splitext(os.path.basename(image))
            lbl_file = os.path.join(os.path.dirname(image), "{}.txt".format(stem))
            if not os.path.isfile(lbl_file):
                raise IOError("Could not find the label file {}".format(lbl_file))
            with open(lbl_file, "r") as fid:
                self.labels.append(list(map(float, fid.read().strip().split(" "))))

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((80, 160)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = PIL.Image.open(self.images[item])
        if img is None:
            raise IOError("Could not read the image {}".format(self.images[item]))
        return torch.Tensor(self.labels[item]), self.transform(img)


class ColorDataSet(DataSet):

    def __init__(self, data_dir):
        super(ColorDataSet, self).__init__(data_dir)

        self.transform = transforms.Compose([
            transforms.Resize((120, 160)),
            transforms.ToTensor()
        ])


class AurelDataSet(DataSet):

    def __init__(self, data_csv: pathlib.Path):
        if not data_csv.is_file():
            ValueError("{} is not a valid file".format(data_csv))

        self.images = []
        self.labels = []
        with data_csv.open() as fid:
            reader = csv.reader(fid)
            for row in reader:
                try:
                    label = float(row[0])
                except ValueError:
                    continue
                self.labels.append([label])
                self.images.append(row[1])

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((80, 160)),
            transforms.ToTensor(),
        ])


class NImagesDataSet(DataSet):
    """
    And extended data set, that returns n images concatenated as one instead of a single one.
    """

    def __init__(self, data_dir, n=1):
        super(NImagesDataSet, self).__init__(data_dir)

        self.n = n
        self.images.sort()

        if len(self) < self.n:
            raise RuntimeError("Found {} images in {}, but require at least {}.".format(len(self), data_dir, self.n))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((80 * self.n, 160)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images) - self.n

    def __getitem__(self, item):
        imgs = [cv2.imread(path) for path in self.images[item:item + self.n]]
        img = np.concatenate(tuple(imgs))
        return torch.Tensor(self.labels[item + self.n]), self.transform(img)


class FutureLabelDataSet(NImagesDataSet):

    def __len__(self):
        # for every entry, we need a future label, therefor we cannot return a value for the last image/label pair
        return len(self.images) - self.n - 1

    def __getitem__(self, item):
        imgs = [cv2.imread(path) for path in self.images[item:item + self.n]]
        img = np.concatenate(tuple(imgs))
        return torch.Tensor(self.labels[item + self.n] + self.labels[item + self.n + 1]), self.transform(img)


class RNNDataSet(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir: pathlib.Path,
                 seq_length: Union[int, Tuple[int, int]],
                 device="cpu",
                 img_size=(120, 160),
                 grayscale=False,
                 in_memory=False):
        self.length = seq_length
        self.sequences = []
        self.dir = data_dir
        self.img_size = img_size
        for path in data_dir.iterdir():
            if path.suffix == ".yaml":
                self.sequences.append(yaml.load(path.read_text()))

        transforms_list = [transforms.Resize(img_size)]
        if grayscale:
            transforms_list.append(transforms.Grayscale())
        transforms_list.append(transforms.ToTensor())
        self.transform = transforms.Compose(transforms_list)
        self.device = torch.device(device)
        self.grayscale = grayscale

        self.imgs = []
        self.actions = []
        self.lbls = []

        self.in_memory = False
        if in_memory:
            for idx in range(len(self.sequences)):
                img, action, lbl = self.__getitem__(idx)
                self.imgs.append(img)
                self.actions.append(action)
                self.lbls.append(lbl)
            self.in_memory = True

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):
        if self.in_memory:
            return self.imgs[item], self.actions[item], self.lbls[item]

        seq = self.sequences[item]
        length = np.random.randint(self.length[0], self.length[1]) if isinstance(self.length, tuple) else self.length
        if length >= len(seq):
            length = len(seq) - 1

        start_idx = np.random.randint(0, len(seq) - length)

        channels = 1 if self.grayscale else 3
        imgs = torch.empty(
            size=(length, channels, self.img_size[0], self.img_size[1]), dtype=torch.float, device=self.device)
        actions = torch.empty(size=(length, 2), dtype=torch.float, device=self.device)
        lbls = torch.empty(size=(length, 2), dtype=torch.float, device=self.device)

        for idx, img_info in enumerate(seq[start_idx:start_idx + length]):
            img = PIL.Image.open((self.dir / img_info["path"]).as_posix())
            if img is None:
                raise RuntimeError("Could not find the image at: {}".format(img_info["path"]))

            imgs[idx, :, :, :] = self.transform(img)
            actions[idx, :] = torch.Tensor(img_info["action"])
            lbls[idx, :] = torch.Tensor(img_info["modifier"])

        return imgs, actions, lbls


if __name__ == "__main__":
    raise NotImplemented("This module cannot be run as an executable")
