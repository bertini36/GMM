# -*- coding: UTF-8 -*-

"""
create a new dataset getting the first D dimensions of a input dataset
"""

import argparse
import pickle as pkl

import numpy as np

"""
Parameters:
    * input: Input path (pkl)
    * output: Output path (pkl)
    * d: Number of dimensions

Execution:
    python reduce_dimensions.py
        -input mallorca/mallorca_linear_int1000_ppca98.pkl
        -output mallorca/mallorca_linear_int1000_ppca4.pkl
        -d 4
"""

parser = argparse.ArgumentParser(description='Data sampler')
parser.add_argument('-input', metavar='input', type=str, default='')
parser.add_argument('-output', metavar='output', type=str, default='')
parser.add_argument('-d', metavar='d', type=int, default='')
args = parser.parse_args()


def main():

    # Get data
    with open('{}'.format(args.input), 'r') as inputfile:
        data = pkl.load(inputfile)
        xn = data['xn']
    N, D = xn.shape

    with open(args.output, 'w') as output:
        pkl.dump({'xn': np.array(xn[:, 0:args.d])}, output)


if __name__ == '__main__': main()
