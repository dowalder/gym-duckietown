#!/usr/bin/env python3

import pathlib
import PIL.Image
import yaml
import csv
from typing import Union, Tuple, Optional, Callable

import torch
import torch.utils.data
import torchvision.transforms as transforms
import PIL.Image
import numpy as np
import cv2

import src.params


def size_from_string(size: str) -> Tuple[int, int]:
    new_size = tuple(map(int, size.strip().split(",")))
    assert len(new_size) == 2, "Expected H,W got {}".format(size)
    return new_size


class Base(torch.utils.data.Dataset):

    def __init__(self, data_dir: pathlib.Path, params: Optional[src.params.Params]=None):
        self.data_in_memory = []
        self.labels = []
        self.data_dir = data_dir
        self.transform = None  # type: Optional[Callable]
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

        color = self.params.get("color", no_raise=True)
        size = self.params.get("image_size", no_raise=True)
        size = size_from_string(size) if size is not None else (120, 160)

        transf = [transforms.Grayscale()] if color == "gray" else []
        self.transform = transforms.Compose(transf + [
            transforms.Resize(size),
            transforms.ToTensor(),
        ])

    def _load_item(self, item):
        img = PIL.Image.open(self.images[item].as_posix())
        if img is None:
            raise IOError("Could not read the image {}".format(self.images[item]))
        return self.transform(img).to(self.params.device), torch.Tensor(self.labels[item]).to(self.params.device)


# class AurelDataSet(DataSet):
#
#     def __init__(self, data_csv: pathlib.Path):
#         if not data_csv.is_file():
#             ValueError("{} is not a valid file".format(data_csv))
#
#         self.images = []
#         self.labels = []
#         with data_csv.open() as fid:
#             reader = csv.reader(fid)
#             for row in reader:
#                 try:
#                     label = float(row[0])
#                 except ValueError:
#                     continue
#                 self.labels.append([label])
#                 self.images.append(row[1])
#
#         self.transform = transforms.Compose([
#             transforms.Grayscale(),
#             transforms.Resize((80, 160)),
#             transforms.ToTensor(),
#         ])
#
#
# class NImagesDataSet(DataSet):
#     """
#     And extended data set, that returns n images concatenated as one instead of a single one.
#     """
#
#     def __init__(self, data_dir, n=1):
#         super(NImagesDataSet, self).__init__(data_dir)
#
#         self.n = n
#         self.images.sort()
#
#         if len(self) < self.n:
#             raise RuntimeError("Found {} images in {}, but require at least {}.".format(len(self), data_dir, self.n))
#
#         self.transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Grayscale(),
#             transforms.Resize((80 * self.n, 160)),
#             transforms.ToTensor(),
#         ])
#
#     def __len__(self):
#         return len(self.images) - self.n
#
#     def __getitem__(self, item):
#         imgs = [cv2.imread(path) for path in self.images[item:item + self.n]]
#         img = np.concatenate(tuple(imgs))
#         return torch.Tensor(self.labels[item + self.n]), self.transform(img)
#
#
# class FutureLabelDataSet(NImagesDataSet):
#
#     def __len__(self):
#         # for every entry, we need a future label, therefor we cannot return a value for the last image/label pair
#         return len(self.images) - self.n - 1
#
#     def __getitem__(self, item):
#         imgs = [cv2.imread(path) for path in self.images[item:item + self.n]]
#         img = np.concatenate(tuple(imgs))
#         return torch.Tensor(self.labels[item + self.n] + self.labels[item + self.n + 1]), self.transform(img)
#
#
# class RNNDataSet(torch.utils.data.Dataset):
#
#     def __init__(self,
#                  data_dir: pathlib.Path,
#                  seq_length: Union[int, Tuple[int, int]],
#                  device="cpu",
#                  img_size=(120, 160),
#                  grayscale=False,
#                  in_memory=False):
#         self.length = seq_length
#         self.sequences = []
#         self.dir = data_dir
#         self.img_size = img_size
#         for path in data_dir.iterdir():
#             if path.suffix == ".yaml":
#                 self.sequences.append(yaml.load(path.read_text()))
#
#         transforms_list = [transforms.Resize(img_size)]
#         if grayscale:
#             transforms_list.append(transforms.Grayscale())
#         transforms_list.append(transforms.ToTensor())
#         self.transform = transforms.Compose(transforms_list)
#         self.device = torch.device(device)
#         self.grayscale = grayscale
#
#         self.imgs = []
#         self.actions = []
#         self.lbls = []
#
#         self.in_memory = False
#         if in_memory:
#             for idx in range(len(self.sequences)):
#                 img, action, lbl = self.__getitem__(idx)
#                 self.imgs.append(img)
#                 self.actions.append(action)
#                 self.lbls.append(lbl)
#             self.in_memory = True
#
#     def __len__(self):
#         return len(self.sequences)
#
#     def __getitem__(self, item):
#         if self.in_memory:
#             return self.imgs[item], self.actions[item], self.lbls[item]
#
#         seq = self.sequences[item]
#         length = np.random.randint(self.length[0], self.length[1]) if isinstance(self.length, tuple) else self.length
#         if length >= len(seq):
#             length = len(seq) - 1
#
#         start_idx = np.random.randint(0, len(seq) - length)
#
#         channels = 1 if self.grayscale else 3
#         imgs = torch.empty(
#             size=(length, channels, self.img_size[0], self.img_size[1]), dtype=torch.float, device=self.device)
#         actions = torch.empty(size=(length, 2), dtype=torch.float, device=self.device)
#         lbls = torch.empty(size=(length, 2), dtype=torch.float, device=self.device)
#
#         for idx, img_info in enumerate(seq[start_idx:start_idx + length]):
#             img = PIL.Image.open((self.dir / img_info["path"]).as_posix())
#             if img is None:
#                 raise RuntimeError("Could not find the image at: {}".format(img_info["path"]))
#
#             imgs[idx, :, :, :] = self.transform(img)
#             actions[idx, :] = torch.Tensor(img_info["action"])
#             lbls[idx, :] = torch.Tensor(img_info["modifier"])
#
#         return imgs, actions, lbls
#
#
# if __name__ == "__main__":
#     raise NotImplemented("This module cannot be run as an executable")
