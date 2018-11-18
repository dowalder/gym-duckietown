"""
Creates one dataset out of two, where the dataset is expected to consist of image1.jpg as images and image1.txt as
corresponding labels.
"""

import pathlib
import random


def main():
    src1 = pathlib.Path("/home/dominik/workspace/duckietown_imitation_learning/test_images")
    src2 = pathlib.Path("/home/dominik/workspace/duckietown_imitation_learning/train_images")
    tgt_test = pathlib.Path("/home/dominik/dataspace/images/images_mixed_train_test/test")
    tgt_train = pathlib.Path("/home/dominik/dataspace/images/images_mixed_train_test/train")

    test_ratio = 0.4
    count = 0
    for directory in [src1, src2]:
        for img in directory.iterdir():
            if img.suffix != ".jpg":
                continue
            txt = img.parent / "{}.txt".format(img.stem)
            if not txt.is_file():
                raise RuntimeError("Hey, no txt file for this image {}. UNACCEPTABLE!!!".format(img))

            if random.uniform(0.0, 1.0) < test_ratio:
                tgt_txt = tgt_test / "{0:05d}.txt".format(count)
                tgt_img = tgt_test / "{0:05d}.jpg".format(count)
            else:
                tgt_txt = tgt_train / "{0:05d}.txt".format(count)
                tgt_img = tgt_train / "{0:05d}.jpg".format(count)

            tgt_txt.write_bytes(txt.read_bytes())
            tgt_img.write_bytes(img.read_bytes())
            count += 1


if __name__ == "__main__":
    main()
