import os
import sys
import argparse

import numpy as np


def main(opts):
    pass


def parse_opts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-o', '--output', help='')
    opts = parser.parse_args()
    return opts


if __init__=="__main__":
    opts = parse_opts(sys.argv[1:])
    main(opts)
