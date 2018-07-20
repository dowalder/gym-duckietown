#!/usr/bin/env python3

import argparse
import pathlib

import cv2


def create_images(road_dir: pathlib.Path, background_dir: pathlib.Path, tgt_dir: pathlib.Path, num_imgs=1000) -> None:
    a_dir = tgt_dir / "A"
    b_dir = tgt_dir / "B"





def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--road_dir", required=True)
    parser.add_argument("--background_dir", required=True)
    parser.add_argument("--tgt_dir", required=True)
    parser.add_argument("--num_imgs", default=1000)

    args = parser.parse_args()

    road_dir = pathlib.Path(args.road_dir)
    background_dir = pathlib.Path(args.background_dir)
    tgt_dir = pathlib.Path(args.tgt_dir)

    assert road_dir.is_dir(), "Not a directory: {}".format(road_dir)
    assert background_dir.is_dir(), "Not a directory: {}".format(background_dir)
    tgt_dir.mkdir(exist_ok=True, parents=True)




if __name__ == "__main__":
    main()