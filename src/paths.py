#!/usr/bin/env python3

"""
Functions for path manipulation
"""

from pathlib import Path
from typing import List


def str_to_list(string: str, need_exist=False, create=False, sep=",", is_dir=False) -> List[Path]:
    """
    Creates a list of pathlib.Paths from a comma (or else) separated list.
    """
    paths = [Path(path) for path in string.split(sep)]
    if create:
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)
    if need_exist:
        for path in paths:
            if is_dir:
                assert path.is_dir(), "Not a directory: {}".format(path)
            else:
                assert path.exists(), "File not found: {}".format(path)
    return paths
