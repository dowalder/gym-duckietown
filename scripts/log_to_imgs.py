#!/usr/bin/env python
import argparse
import os

import rosbag
import cv2
import numpy as np
import cv_bridge
import collections


def get_img_messages(path, img_topic, compressed=False):
    """
    """
    assert isinstance(path, str)
    assert isinstance(img_topic, str)

    bag = rosbag.Bag(path)

    _, topics = bag.get_type_and_topic_info()

    if img_topic not in topics:
        raise ValueError("Could not find the requested topic (%s) in the bag %s" % (img_topic, path))

    imgs = []
    bridge = cv_bridge.CvBridge()
    for msg in bag.read_messages():
        topic = msg.topic
        if topic != img_topic:
            continue
        if compressed:
            img = bridge.compressed_imgmsg_to_cv2(msg.message)
        else:
            img = bridge.imgmsg_to_cv2(msg.message)
        imgs.append(img)
    bag.close()

    return imgs


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bag", required=True)
    parser.add_argument("--tgt_dir", required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--compressed", action="store_true")

    args = parser.parse_args()

    imgs = get_img_messages(args.bag, args.topic, args.compressed)

    for idx, img in enumerate(imgs):
        path = os.path.join(args.tgt_dir, "{0:05d}.jpg".format(idx))
        cv2.imwrite(path, img)


if __name__ == "__main__":
    main()
