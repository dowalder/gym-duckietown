#!/usr/bin/env python3

import pathlib
import shutil

import yaml
import cv2
import numpy as np

import src.graphics


def write_sim_files(src_dir: pathlib.Path, tgt_dir: pathlib.Path):
    for folder in src_dir.iterdir():
        if not folder.is_dir():
            continue
        img_tgt_dir  = tgt_dir / folder.name
        or_tgt_dir = tgt_dir / "{}_onlyroad".format(folder.name)
        om_tgt_dir = tgt_dir / "{}_onlymarks".format(folder.name)
        img_tgt_dir.mkdir(exist_ok=True)
        or_tgt_dir.mkdir(exist_ok=True)
        om_tgt_dir.mkdir(exist_ok=True)

        seq_dir = folder / "2048images"
        info_yaml = seq_dir / "seq_00000.yaml"  # type: pathlib.Path
        if not info_yaml.is_file():
            print("WARNING: found invalid directory at {} (could not find {})".format(folder, info_yaml))
            continue
        seq = yaml.load(info_yaml.read_text())
        if len(seq) < 2048:
            print("WARNING: not enough entries found in {}. Ignoring this dataset.".format(info_yaml))
        for idx, item in enumerate(seq):
            if idx == 2048:
                break
            img_path = seq_dir / item["path"]
            onlyroad_path = seq_dir / item["only_road_pth"]
            shutil.copy(img_path.as_posix(), (img_tgt_dir / img_path.name).as_posix())
            shutil.copy(onlyroad_path.as_posix(), (or_tgt_dir / img_path.name).as_posix())

            img = cv2.imread(onlyroad_path.as_posix())
            img = src.graphics.apply_color_filter(img)
            cv2.imwrite((om_tgt_dir / img_path.name).as_posix(), img)


def write_real_files(src_dir: pathlib.Path, tgt_dir: pathlib.Path):
    tgt_dir.mkdir(exist_ok=True)
    idx = 0
    files = [fil for fil in src_dir.iterdir()]
    files.sort(key=lambda x: x.as_posix())
    for fil in files:
        if fil.suffix != ".jpg":
            continue
        img = cv2.imread(fil.as_posix())
        img = cv2.resize(img, (160, 120))
        cv2.imwrite((tgt_dir / fil.name).as_posix(), img)
        idx += 1
        if idx == 2047:
            break


def main():
    sim_dir = pathlib.Path("/home/dominik/dataspace/images/mxm_sim")
    real_dir = pathlib.Path("/home/dominik/dataspace/images/real/20180108135529_a313")
    tgt_dir = pathlib.Path("/home/dominik/tmp/fid")

    write_real_files(real_dir, tgt_dir / "real")
    write_sim_files(sim_dir, tgt_dir)


if __name__ == "__main__":
    main()
