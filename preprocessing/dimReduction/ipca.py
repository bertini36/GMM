# -*- coding: UTF-8 -*-

"""
Dimensionality reduction with incremental principal component analysis
"""

import argparse
import csv
import pickle as pkl
import sys

import numpy as np
from sklearn.decomposition import IncrementalPCA

from .common import format_track

"""
Parameters:
    * input: Input path (CSV with ; delimiter)
    * output: Output path (PKL file)
    * c: PCA number of principal components

Execution:
    python ipca.py -input mallorca_nnint50.csv 
                   -output generated/mallorca_nnint50_pca50.pkl -c 50
"""

parser = argparse.ArgumentParser(description='Incremental PCA')
parser.add_argument('-input', metavar='input', type=str, default='')
parser.add_argument('-output', metavar='output', type=str, default='')
parser.add_argument('-c', metavar='c', type=int, default=50)
args = parser.parse_args()


def main():
    try:
        if not('.csv' in args.input): raise Exception('input_format')
        if not('.pkl' in args.output): raise Exception('output_format')

        with open(args.input, 'rb') as input:
            reader = csv.reader(input, delimiter=';')
            reader.next()
            n = 0
            xn = []
            for track in reader:
                print('Track {}'.format(n))
                track = format_track(track[0])
                xn.append(track)
                n += 1

            print('Doing Incremental PCA...')
            pca = IncrementalPCA(n_components=args.c, batch_size=500)
            xn_new = pca.fit_transform(xn)

            with open(args.output, 'w') as output:
                pkl.dump({'xn': np.array(xn_new)}, output)

    except IOError:
        print('File not found!')
    except Exception as e:
        if e.args[0] == 'input_format': print('Input must be a CSV file')
        elif e.args[0] == 'output_format': print('Output must be a PKL file')
        else:
            print('Unexpected error: {}'.format(sys.exc_info()[0]))
            raise


if __name__ == '__main__': main()
