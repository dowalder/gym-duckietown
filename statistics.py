#!/usr/bin/env python3

import argparse
import pathlib
from typing import Tuple

import yaml


class Statistiker:

    def __init__(self, range: Tuple[float, float], num_bins: int, compute_average=True):
        assert isinstance(range, tuple), "got {}".format(range.__class__)
        assert len(range) == 2
        assert range[0] < range[1]

        self.histo = [0] * num_bins
        self.lower = float(range[0])
        self.upper = float(range[1])
        self.num_bins = num_bins
        self.num = 0.0
        self.sum = None
        self.comp_avg = compute_average

    def _in_range(self, val: float) -> bool:
        return self.lower <= val <= self.upper

    def reset_avg(self):
        self.num = 0.0
        self.sum = None

    def add(self, val: float):
        assert self._in_range(val)
        idx = self.num_bins - 1 if val == self.upper else int((val - self.lower) / (self.upper - self.lower) * self.num_bins)
        self.histo[idx] += 1

        if self.comp_avg:
            if self.sum is None:
                self.sum = val
            else:
                self.sum += val
            self.num += 1.0

    def avg(self):
        if not self.comp_avg:
            raise RuntimeError("computing average is turned off")
        return self.sum / self.num

    def peak(self, print_pretty: True) -> int:
        max_idcs = []
        max_val = 0
        for idx, val in enumerate(self.histo):
            if val == max_val:
                max_idcs.append(idx)
            if val > max_val:
                max_val = val
                max_idcs = [idx]

        if print_pretty:
            print("The following bins are peak bins with the value {}:".format(max_val))
            for idx in max_idcs:
                lower = self.lower + (self.upper - self.lower) * idx / float(self.num_bins)
                upper = self.lower + (self.upper - self.lower) * (idx + 1) / float(self.num_bins)
                print("\tbin_nr: {} contains: {} <= x < {}".format(idx, lower, upper))
        return max_val


def apply_statistik(statistiker: Statistiker, dir: pathlib.Path) -> Statistiker:
    if not dir.is_dir():
        raise ValueError("dir must be an existing directory")

    for pth in dir.iterdir():
        if pth.suffix != ".yaml":
            continue

        data = yaml.load(pth.read_text())
        for entry in data:
            statistiker.add(entry["omega"])
    return statistiker


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_dir", "-s", required=True)

    args = parser.parse_args()

    statistiker = Statistiker((-4, 4), 100)

    statistiker = apply_statistik(statistiker, pathlib.Path(args.src_dir))

    statistiker.peak(True)
    print(statistiker.histo)
    print("avg: {}".format(statistiker.avg()))


if __name__ == "__main__":
    main()