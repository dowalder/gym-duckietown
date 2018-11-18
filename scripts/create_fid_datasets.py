#!/usr/bin/env python3

"""
Aggregates images extracted from runs from logs (see analzye_logs package) and sorts them after colors into fixed size
datasets.
"""

import pathlib
import argparse
import shutil


class FidDirectory:

    def __init__(self, path: pathlib.Path, restrict_size=-1):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        self.restrict_size = restrict_size

        tmp_path = "-1"
        for img_path in self.path.iterdir():
            if not img_path.suffix == ".jpg":
                continue
            if img_path.stem > tmp_path:
                tmp_path = img_path.stem
        self.counter = int(tmp_path) + 1

        if self.full():
            print("FidDirectory: already full at {}: {}".format(self.restrict_size, self.path))

    def full(self):
        return 0 < self.restrict_size <= self.counter

    def add(self, source_file: pathlib.Path):
        if source_file.suffix != ".jpg":
            return
        if self.full():
            return

        new_file = self.path / "{0:06d}.jpg".format(self.counter)
        shutil.copy(source_file.as_posix(), new_file.as_posix())
        self.counter += 1

        if self.full():
            print("FidDirectory: reached maximum size at {}: {}".format(self.restrict_size, self.path))

    def clear(self):
        for path in self.path.iterdir():
            path.unlink()
        self.counter = 0


def translate_source_dir(name: str) -> str:
    source_encoding = name.split("-")[0]
    return {
        "b_b": "blue",
        "b_r": "red",
        "b_g": "green",
        "b_w": "b_white",
        "m_w": "m_white",
        "l_w": "l_white",
    }[source_encoding]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src", required=True, help="source directory containing runs")
    parser.add_argument("--tgt", required=True, help="target directory to store datasets")
    parser.add_argument("--reset", action="store_true", help="reset all directories before copying new files")

    args = parser.parse_args()

    src_dir = pathlib.Path(args.src)
    assert src_dir.is_dir()
    tgt_dir = pathlib.Path(args.tgt)

    fid_dirs = {
        "blue": FidDirectory(tgt_dir / "blue", 2048),
        "red": FidDirectory(tgt_dir / "red", 2048),
        "green": FidDirectory(tgt_dir / "green", 2048),
        "m_white": FidDirectory(tgt_dir / "m_white", 2048),
        "l_white": FidDirectory(tgt_dir / "l_white", 2048),
        "b_white": FidDirectory(tgt_dir / "b_white", 2048),
    }

    if args.reset:
        for fid_dir in fid_dirs.values():
            fid_dir.clear()

    for path in src_dir.iterdir():
        tgt_dir_name = translate_source_dir(path.stem)
        for img_path in (path / "original").iterdir():
            fid_dirs[tgt_dir_name].add(img_path)


if __name__ == "__main__":
    main()
