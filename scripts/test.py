#!/usr/bin/env python3

import pathlib
import random

import matplotlib.pyplot as plt
import torch

import networks
import dataset


def main():
    model_path = pathlib.Path("/home/dominik/dataspace/models/rnn_randomwalk_forwad/run5_step_40000.pth")
    data_path = pathlib.Path("/home/dominik/dataspace/images/randomwalk_forward/test")
    result_path = pathlib.Path("something")

    net = networks.BasicConvRNN("cpu")
    net.load_state_dict(torch.load(model_path.as_posix()))

    test_set = dataset.RNNDataSet(data_dir=data_path, seq_length=40)

    output = []

    with torch.no_grad():
        net.init_hidden()
        idx = random.randint(0, len(test_set) - 1)
        print(idx)
        imgs, actions, lbls = test_set[idx]
        out = net(imgs, actions)
        out = out.squeeze()
        diff = (lbls - out)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    t = range(len(diff[:, 0]))
    ax.plot(t, diff[:, 0].tolist())
    ax.plot(t, diff[:, 1].tolist())
    ax.set_xlabel("t")
    ax.set_ylabel("error")
    ax.set_title("Calibration Estimation")
    ax.grid()

    plt.show()


if __name__ == "__main__":
    main()
