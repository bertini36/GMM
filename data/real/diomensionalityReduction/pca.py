# -*- coding: UTF-8 -*-

"""
Dimensionality reduction with principal component analysis
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
    * savePCA: Save PCA object

Execution:
    python pca.py -input mallorca_nnint50.csv 
                  -output generated/mallorca_nnint50_pca50.pkl -c 50
"""

parser = argparse.ArgumentParser(description='Principal Component Analysis')
parser.add_argument('-input', metavar='input', type=str, default='')
parser.add_argument('-output', metavar='output', type=str, default='')
parser.add_argument('-c', metavar='c', type=int)
parser.add_argument('-savePCA', dest='savePCA', action='store_true')
parser.set_defaults(savePCA=False)
args = parser.parse_args()

SAVE_PCA = args.savePCA


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

            print('Length xn: {}'.format(len(xn)))
            print('Length xn[0]: {}'.format(len(xn[0])))

            if args.c is not None:
                print('Doing PCA...')
                pca = PCA(n_components=args.c)
            else:
                print('Doing PCA with MLE...')
                pca = PCA(n_components='mle', svd_solver='full')
            xn_new = pca.fit_transform(xn)

            print('Explained variance: {}'
                  .format(pca.explained_variance_ratio_))
            print('Number of components: {}'
                  .format(len(pca.explained_variance_ratio_)))

            with open(args.output, 'w') as output:
                pkl.dump({'xn': np.array(xn_new)}, output)

            if SAVE_PCA:
                with open('pca.pkl', 'w') as output:
                    pkl.dump({'pca': pca}, output)

    except IOError:
        print('File not found!')
    except Exception as e:
        if e.args[0] == 'input_format': print('Input must be a CSV file')
        elif e.args[0] == 'output_format': print('Output must be a PKL file')
        else:
            print('Unexpected error: {}'.format(sys.exc_info()[0]))
            raise


if __name__ == '__main__': main()
