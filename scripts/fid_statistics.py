#!/usr/bin/env python3

import pathlib
import numpy as np


map_light_to_index = {
    "blue": 0,
    "green": 1,
    "red": 2,
    "b_white": 3,
    "m_white": 4,
    "l_white": 5,
}


def print_latex_table(mat: np.ndarray):
    assert mat.shape == (6, 6)
    for i in range(6):
        txt = []
        for j in range(6):
            txt.append(" & " + str(mat[i, j]))
        print("".join(txt))


def make_fid_stats(path: pathlib.Path) -> np.ndarray:
    txt = path.read_text()
    txt = txt.strip().split("\n")
    mat = np.zeros((6, 6), dtype=np.float)
    for line in txt:
        light1, light2, fid = line.split(",")
        idx1 = map_light_to_index[light1]
        idx2 = map_light_to_index[light2]
        mat[idx1, idx2] = fid
        mat[idx2, idx1] = fid
    print("-"*50)
    print(path)
    # print_latex_table(mat)
    mat = np.tril(mat)
    array = mat.flatten()
    array = array[array.nonzero()]
    print(array)
    print("\tmean:\t", np.mean(array))
    print("\tstd:\t", np.std(array))

    return np.tril(mat)


def main():
    fid_dir = pathlib.Path("/home/dominik/data/ETH/MasterThesis/data")
    make_fid_stats(fid_dir / "fid_original.txt")
    make_fid_stats(fid_dir / "fid_style3.txt")
    make_fid_stats(fid_dir / "fid_pix2pix.txt")
    make_fid_stats(fid_dir / "fid_20_sib_cropped.txt")



if __name__ == "__main__":
    main()
