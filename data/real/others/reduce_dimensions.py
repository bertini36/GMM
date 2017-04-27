# -*- coding: UTF-8 -*-

"""
Create a new dataset getting the first D dimensions of a input dataset
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
    python reduce_dimensions.py -input mallorca_nnint50_pca50.pkl -d 10
                                -output generated/mallorca_nnint50_pca10.pkl
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

    with open(args.output, 'w') as output:
        pkl.dump({'xn': np.array(xn[:, 0:args.d])}, output)


if __name__ == '__main__': main()
