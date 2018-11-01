#!/usr/bin/env python

import argparse
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas


def plot1(args):
    ns = [
        ("bright_white", (1, 1, 0.1, 1)),
        ("mid_white", (0, 0, 0, 0)),
        ("low_white", (0, 0, 0, 0)),
        ("red", (0, 0, 0, 0)),
        ("blue", (0, 0, 0, 0)),
        ("green", (0, 0, 0, 0)),
    ]
    cont = [
        ("bright_white", (0.95, 0.7, 0.2, 0.6)),
        ("mid_white", (0.75, 0.4, 0.273, 0.818)),
        ("low_white", (0, 0, 0, 0)),
        ("red", (0.7, 0.5, 0, 0.2)),
        ("blue", (0, 0, 0, 0)),
        ("green", (0.9, 0.4, 0, 0)),
    ]

    fig, ax_list = plt.subplots(1, 4)

    names = [pt[0] for pt in ns]

    for idx, name in enumerate(["straight", "s", "curvy", "curvy_"]):
        axes = ax_list[idx]
        data_ns = [pt[1][idx] for pt in ns]
        data_cont = [pt[1][idx] for pt in cont]
        axes.plot(data_ns, names, "o-")
        axes.plot(data_cont, names, "x-")
        axes.set_title(name)
        axes.set_xlim(0, 1)
        axes.grid(True)

    for i in range(1, len(ax_list)):
        ax_list[i].set_yticklabels([])
    ax_list[-1].legend(["original", "neural_style"])
    fig.show()


def frame_statistics(frame: pandas.DataFrame) -> pandas.DataFrame:
    times = {"s": [], "curvy": [], "straight": []}
    quantiles = {"s": None, "curvy": None, "straight": None}
    maxes = {"s": None, "curvy": None, "straight": None}
    mines = {"s": None, "curvy": None, "straight": None}
    means = {"s": None, "curvy": None, "straight": None}

    for row in frame.itertuples():
        if not getattr(row, "success"):
            continue
        times[getattr(row, "run")].append(getattr(row, "time"))
    for k, v in times.items():
        v.sort()
        quantiles[k] = v[int(len(v) * 0.75)]
        maxes[k] = v[-1]
        mines[k] = v[0]
        means[k] = sum(v) / len(v)

    frame = frame.assign(success_rate=pandas.Series())

    def add_success_rate(series: pandas.Series) -> pandas.Series:
        if series.success == 1:
            series.success_rate = 1.0
        else:
            series.success_rate = min(series.time / quantiles[series.run], 1.0)
        return series

    frame = frame.apply(add_success_rate, axis=1, result_type="broadcast")

    print("quantiles: ", quantiles)
    print("maxes: ", maxes)
    print("mines: ", mines)
    print("means: ", means)
    return frame


def plot2(args):
    # def create_success_measure(frame: pandas.DataFrame) -> pandas.DataFrame:
    data = pandas.read_csv("/home/dominik/data/ETH/MasterThesis/data/runs.csv")
    data = frame_statistics(data)

    only_s = data[data.run == "s"]
    only_curvy = data[data.run == "curvy"]
    only_straight = data[data.run == "straight"]

    # f, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(7, 7))
    # sns.distplot(only_s[only_s.success == 1].time, ax=axes[0], kde=False)
    # sns.distplot(only_curvy[only_curvy.success == 1].time, ax=axes[1], kde=False)
    # sns.distplot(only_straight[only_straight.success == 1].time, ax=axes[2], kde=False)

    colors = {"blue": "b", "green": "g", "red": "r", "b_white": "0.8", "m_white": "0.5", "l_white": "0.1"}

    sns.catplot(x="transformer", y="success_rate", hue="light", data=data[data.run == "straight"], kind="bar",
                legend_out=False, palette=colors, legend=False)
    sns.catplot(x="transformer", y="success_rate", hue="light", data=data[data.run == "s"], kind="bar",
                legend_out=False, palette=colors, legend=False)
    sns.catplot(x="transformer", y="success_rate", hue="light", data=data[data.run == "curvy"], kind="bar",
                legend_out=False, palette=colors, legend=False)
    sns.catplot(x="transformer", y="success_rate", hue="light", data=data, kind="bar", legend_out=False, palette=colors,
                legend=False)

    # grid = sns.FacetGrid(data=data, row="transformer", hue="light", palette=colors)
    # grid.map(sns.barplot, "run", "success_rate")

    plt.show()


def plot3(args):
    data = pandas.read_csv("/home/dominik/data/ETH/MasterThesis/data/fid_all.txt")

    sns.barplot(x="transformer", y="fid", data=data)
    plt.show()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--which", "-w", required=True, type=int)
    parser.add_argument("--path", "-p", default=".")

    args = parser.parse_args()

    if args.which == 1:
        img = plot1(args)
    elif args.which == 2:
        img = plot2(args)
    elif args.which == 3:
        img = plot3(args)
    else:
        raise ValueError("invalid argument: --which {}".format(args.which))


if __name__ == "__main__":
    main()
