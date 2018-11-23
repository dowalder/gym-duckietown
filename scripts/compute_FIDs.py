#!/usr/bin/env python3

"""
Computes the FID between different datasets. You need a script that takes paths to two dataset can computes the FID from
it as an executable. Example: https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py
"""

import subprocess
import threading

NUM_THREADS = 8


def main():
    comparisons = []

    sets = [
        ["original/blue", "original/red", "original/green", "original/b_white", "original/m_white", "original/l_white"],
        ["20_sib_cropped/blue", "20_sib_cropped/red", "20_sib_cropped/green", "20_sib_cropped/b_white", "20_sib_cropped/m_white", "20_sib_cropped/l_white"],
        ["style3/blue", "style3/red", "style3/green", "style3/b_white", "style3/m_white", "style3/l_white"],
        ["pix2pix/blue", "pix2pix/red", "pix2pix/green", "pix2pix/b_white", "pix2pix/m_white", "pix2pix/l_white"]
    ]
    for set in sets:
        for idx_s in range(len(set)):
            for idx_t in range(idx_s + 1, len(set)):
                comparisons.append((set[idx_s], set[idx_t]))

    datapath = "/home/dwalder/dataspace/fid/"
    execuatable = "./fid_score.py"

    def get_comparison():
        if comparisons:
            return comparisons.pop()
        else:
            return None

    lock = threading.Lock()

    def compute_fid():
        while True:
            lock.acquire(blocking=True)
            try:
                comp = get_comparison()
                if comp is None:
                    break
                set1, set2 = comp
            finally:
                lock.release()
            ret = subprocess.run([execuatable, datapath + set1, datapath + set2], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, universal_newlines=True)
            msg = ""
            if ret.returncode:
                msg = "{} exited with code {}:\n{}".format(ret.args, ret.returncode, ret.stdout)

            out = str(ret.stdout).split()
            try:
                fid = float(out[-1][:4])
            except ValueError as e:
                msg = "Could not read out FID score for {} vs {}:\nout: {}\nerr: {}".format(set1, set2, ret.stdout, e)

            if not msg:
                msg = "{} vs {}: {}".format(set1, set2, fid)

            lock.acquire(blocking=True)
            try:
                print(msg)
            finally:
                lock.release()

    threads = []
    for _ in range(NUM_THREADS):
        thread = threading.Thread(target=compute_fid)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
