#!/usr/bin/env python3
"""
Creates a video of two image series next to each other. The src_left directory can be searched recursively for images,
where a video is created for every folder found that contains images. The src_right directory is expected to have the
same relative directory structure.
"""

import argparse
import pathlib

import cv2
import numpy as np


def check_mirrored_directories(src_path: pathlib.Path, tgt_path: pathlib.Path, check_only=None, recursive=False):
    def check_folder(path: pathlib.Path):
        for fyle in path.iterdir():
            if fyle.is_dir():
                if recursive:
                    check_folder(fyle)
                continue
            if check_only and fyle.suffix not in check_only:
                continue
            mirrored_file = tgt_path / fyle.relative_to(src_path)
            if not mirrored_file.exists():
                raise FileNotFoundError("src_path contains file that does not exist in tgt_path: {}".format(fyle))

    check_folder(src_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_left", required=True, help="directory containing images for the left side")
    parser.add_argument("--src_right", required=True, help="directory containing images for the right side")
    parser.add_argument("--tgt", required=True, help="target directory to store video files in")
    parser.add_argument("--fps", default=20, help="fps of the videos")
    parser.add_argument("--recursive", action="store_true",
                        help="search recursively for directories containing images and create a video from them")
    parser.add_argument("--height", default=120)
    parser.add_argument("--width", default=160)

    args = parser.parse_args()

    src_left = pathlib.Path(args.src_left)
    assert src_left.is_dir()
    src_right = pathlib.Path(args.src_right)
    assert src_right.is_dir()
    tgt_dir = pathlib.Path(args.tgt)
    tgt_dir.mkdir(exist_ok=True)

    check_mirrored_directories(src_left, src_right, check_only=[".png", ".jpg"], recursive=True)
    width, height = (args.width, args.height)

    def make_video(path_left: pathlib.Path):
        imgs_left = []
        imgs_right = []
        for fyle in path_left.iterdir():
            if fyle.is_dir():
                if args.recursive:
                    make_video(fyle)
                continue
            if fyle.suffix not in [".jpg", ".png"]:
                continue
            imgs_left.append(fyle)
            imgs_right.append(src_right / fyle.relative_to(src_left))

        if not imgs_left:
            return
        current_dir = tgt_dir / path_left.relative_to(src_left)
        current_dir.parent.mkdir(exist_ok=True, parents=True)
        video_filename = (current_dir.parent / "{}.mp4".format(current_dir.name))
        video = cv2.VideoWriter(video_filename.as_posix(),
                                cv2.VideoWriter_fourcc(*"MP4V"),
                                int(args.fps),
                                (width * 2, height))

        imgs_left.sort()
        imgs_right.sort()
        for path_left, path_right in zip(imgs_left, imgs_right):
            img_left = cv2.resize(cv2.imread(path_left.as_posix()), (width, height))
            img_right = cv2.resize(cv2.imread(path_right.as_posix()), (width, height))
            new_img = np.empty((height, width * 2, 3), np.uint8)
            new_img[:, 0:width, :] = img_left
            new_img[:, width:, :] = img_right
            video.write(new_img)
        video.release()

        print("Finished video at: {}".format(video_filename))

    make_video(src_left)


if __name__ == "__main__":
    main()
