#!/usr/bin/env python3

"""
Changes continuous actions into discrete ones. Assumes actions to be stored in .txt files containing nothing else than
a single number.
"""

import argparse
from pathlib import Path
from typing import Tuple


class Converter:
    
    def __init__(self, num_bins: int, limits: Tuple[float, float]):
        self.bins = []
        for i in range(num_bins):
            low = limits[0]
            up = limits[1]
            self.bins.append(low + i * float(up - low) / num_bins)
    
    def which_bin(self, number: float):
        for i in range(2, len(self.bins)):
            if number < self.bins[i]:
                return i - 1
        return len(self.bins) - 1


def convert_to_discrete(src_dir: Path, num_bins: int, limits: Tuple[float, float]):
    converter = Converter(num_bins, limits)
    for path in src_dir.iterdir():
        if path.suffix != ".txt":
            continue

        try:
            labels = list(map(float, path.read_text().split()))
        except Exception:
            print("Could not read file: {}".format(path))
            raise
        labels_new = [str(converter.which_bin(label)) for label in labels]
        path.write_text(" ".join(labels_new))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_dir", required=True)
    parser.add_argument("--num_bins", required=True, type=int)
    # The parentheses are need because argparse does not allow argument values to start with a minus sign.
    parser.add_argument("--limits", default="(-4,4)", type=str)

    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    assert src_dir.is_dir(), "Not a directory: {}".format(src_dir)

    try:
        assert args.limits[0] == "(" and args.limits[-1] == ")"
        limits = tuple(map(float, args.limits[1:-1].split(",")))
        assert len(limits) == 2 and limits[0] < limits[1]
    except Exception:
        print("ERROR: --limits must be in the form '(n1,n2)' where n1 and n2 are real numbers and n1 < n2'")
        raise

    convert_to_discrete(src_dir, args.num_bins, limits)


if __name__ == "__main__":
    main()
