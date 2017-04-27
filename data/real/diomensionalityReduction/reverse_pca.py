# -*- coding: UTF-8 -*-

"""
Converts input data in real data using a previous PCA object
"""

import argparse
import csv
import pickle as pkl
import sys

"""
Parameters:
    * pca: Path to pkl that contains a previous PCA (pkl)
    * data: Set of points to convert to the real space (pkl)
    * output: Real points (csv)

Execution:
    python reverse_pca.py -pca pca.pkl -data new_data.pkl
                          -output generated/transformed_new_data.csv
"""

parser = argparse.ArgumentParser(description='Reverse PCA')
parser.add_argument('-pca', metavar='pca', type=str, default='pca.pkl')
parser.add_argument('-data', metavar='data', type=str, default='new_data.pkl')
parser.add_argument('-output', metavar='output',
                    type=str, default='transformed_new_data.csv')
args = parser.parse_args()


def main():
    try:
        if not ('.pkl' in args.pca): raise Exception('input_format')
        if not ('.pkl' in args.data): raise Exception('input_format')
        if not ('.csv' in args.output): raise Exception('output_format')

        # Get PCA
        with open(args.pca, 'r') as input:
            data = pkl.load(input)
            pca = data['pca']

        # Read data
        with open(args.data, 'r') as input:
            data = pkl.load(input)
            xn = data['xn']

        # Transform to real data
        xn = pca.inverse_transform(xn)

        with open(args.output, 'wb') as output:
            writer = csv.writer(output, delimiter=';',
                                escapechar='', quoting=csv.QUOTE_NONE)
            writer.writerow(['Points'])
            for track in range(len(xn)):
                point = 0
                new_track = []
                for _ in range(len(xn[0]) / 2):
                    new_track.append([xn[track, point], xn[track, point + 1]])
                    point += 2
                writer.writerow([new_track])

    except IOError:
        print('File not found!')
    except Exception as e:
        if e.args[0] == 'input_format': print('Input pca must be a pkl file')
        elif e.args[0] == 'output_format': print('Output must be a csv file')
        else:
            print('Unexpected error: {}'.format(sys.exc_info()[0]))
            raise


if __name__ == '__main__': main()
