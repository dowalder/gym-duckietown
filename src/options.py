#!/usr/bin/env python3

"""
Utility to load choose between different options.
"""

from typing import Dict, Callable, Any


def choose(option: str, options: Dict[str, Callable], name: str, *args, **kwargs) -> Any:

    for key, val in options.items():
        if key == option:
            return val(*args, **kwargs)
    raise ValueError("No option named {} in {}. Options are: {}".format(option, name, "|".join(sorted(options.keys()))))
