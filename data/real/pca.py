# -*- coding: UTF-8 -*-

"""
Principal Component Analysis
"""

import argparse
import csv
import pickle as pkl
import sys

import numpy as np
from sklearn.decomposition import PCA

"""
Parameters:
    * input: Input path (CSV with ; delimiter)
    * output: Output path (PKL file)
    * c: PCA number of principal components

Execution:
    python pca.py -input porto_int50.csv -output porto_pca.pkl -c 50
"""

parser = argparse.ArgumentParser(description='PCA')
parser.add_argument('-input', metavar='input', type=str, default='')
parser.add_argument('-output', metavar='output', type=str, default='')
parser.add_argument('-c', metavar='c', type=int, default=50)
args = parser.parse_args()

INPUT = args.input
OUTPUT = args.output
USE_N_COMPONENTS = True
N_COMPONENTS = args.c


def format_track(track):
    """
    Format track from String to coordinates list
    :param track: Track as a string
    :return: Track as a Python list of coordinates
    """
    new_track = []
    for point in track.split('[[')[1].split(']]')[0].split('], ['):
        aux = [float(n) for n in point.split(', ')]
        new_track.append(aux[0])
        new_track.append(aux[1])
    return new_track


def main():
    try:
        if not('.csv' in INPUT): raise Exception('input_format')
        if not('.pkl' in OUTPUT): raise Exception('output_format')

        with open(INPUT, 'rb') as input:
            reader = csv.reader(input, delimiter=';')
            reader.next()
            n = 0
            xn = []
            for track in reader:
                print('Track {}'.format(n))
                track = format_track(track[0])
                xn.append(track)
                n += 1

            print('Length xn: {}'.format(len(xn)))
            print('Length xn[0]: {}'.format(len(xn[0])))

            if USE_N_COMPONENTS:
                print('Doing PCA...')
                pca = PCA(n_components=N_COMPONENTS)
            else:
                print('Doing PCA with MLE...')
                pca = PCA(n_components='mle', svd_solver='full')
            xn_new = pca.fit_transform(xn)

            print('Explained variance: {}'
                  .format(pca.explained_variance_ratio_))
            print('Number of components: {}'
                  .format(len(pca.explained_variance_ratio_)))

            with open(OUTPUT, 'w') as output:
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
