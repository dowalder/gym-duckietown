#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Dict, Any

import yaml
import torch



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--conf", "-c", required=True, type=str, help="conf file containing options")
    parser.add_argument("--name", "-n", required=True, type=str, help="give a name")



if __name__ == "__main__":
    main()
