#!/usr/bin/env python3

import argparse
import pathlib

import cv2
import numpy as np

import src.graphics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_dir", required=True)
    parser.add_argument("--tgt_dir", required=True)

    args = parser.parse_args()

    src_dir = pathlib.Path(args.src_dir)
    tgt_dir = pathlib.Path(args.tgt_dir)

    if not src_dir.is_dir():
        raise FileNotFoundError("--src_dir does not exist: {}".format(src_dir))

    tgt_dir.mkdir(exist_ok=True, parents=True)

    for pth in src_dir.iterdir():
        if pth.suffix != ".jpg":
            continue
        img = cv2.imread(pth.as_posix())

        img = src.graphics.gradient_lighting(img)
        for _ in range(4):
            if np.random.random() < 0.2:
                img = src.graphics.spot(img)

        tgt_pth = tgt_dir / pth.name
        cv2.imwrite(tgt_pth.as_posix(), img)


if __name__ == "__main__":
    main()
