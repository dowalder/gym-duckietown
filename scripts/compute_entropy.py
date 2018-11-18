#!/usr/bin/env python3

"""
Computes the average entropy of images contained in a directory. Can recursively search for images in directory
tree
"""

import argparse
import pathlib

import numpy as np
import cv2
import scipy.stats


def compute_entropy(labels, base=None):
    _, counts = np.unique(labels, return_counts=True)
    return scipy.stats.entropy(counts, base=base)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_dir", required=True, help="path of dataset to compute entropy from")
    parser.add_argument("--recursive", "-r", action="store_true", help="perform recursive search for images")
    parser.add_argument("--verbose", "-v", action="store_true", help="print information for every directory")

    args = parser.parse_args()

    src_dir = pathlib.Path(args.src_dir)
    entropy = []

    def entropy_of_dir(path: pathlib.Path):
        tmp_entropy = []
        for fyle in path.iterdir():
            if fyle.is_dir():
                if args.recursive:
                    entropy_of_dir(fyle)
                continue
            if fyle.suffix not in [".jpg", ".png"]:
                continue
            img = cv2.imread(fyle.as_posix())
            for i in range(3):
                # np.ravel() is the same as np.flatten(), but without copying (making 1-D array from 2-D)
                tmp_entropy.append(compute_entropy(img[:, :, i].ravel()))

        entropy.extend(tmp_entropy)
        if args.verbose:
            result = str(sum(tmp_entropy) / len(tmp_entropy)) if tmp_entropy else "no images found"
            print("Finished: {}: {}".format(path, result))

    entropy_of_dir(src_dir)

    print("mean:", np.mean(entropy))
    print("std: ", np.std(entropy))


if __name__ == "__main__":
    main()
