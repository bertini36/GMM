# -*- coding: UTF-8 -*-

"""
2D NN interpolation
"""

import argparse
import csv
import pickle as pkl
import sys

import numpy as np

"""
Parameters:
    * input: Input path (CSV with ; delimiter)
    * output: Output path (CSV with ; delimiter)
    * n: Interpolation number of points

Execution:
    python nn_interpolation.py -input porto.csv -output porto_int50.csv -n 50
"""

parser = argparse.ArgumentParser(description='KNN interpolation')
parser.add_argument('-input', metavar='input', type=str, default='')
parser.add_argument('-output', metavar='output', type=str, default='')
parser.add_argument('-n', metavar='n', type=int, default=50)
parser.add_argument('--generatePKL', dest='generatePKL', action='store_true')
parser.add_argument('--no-generatePKL', dest='generatePKL', action='store_false')
parser.set_defaults(generatePKL=False)
args = parser.parse_args()

INPUT = args.input
OUTPUT = args.output
N = args.n
GENERATE_PKL = args.generatePKL

csv.field_size_limit(sys.maxsize)


def format_track(track):
    """
    Format track from String to points list
    :param track: Track as a string
    :return: Track as a Python list of points
    """
    new_track = []
    for point in track.split('[[')[1].split(']]')[0].split('], ['):
        new_track.append([float(n) for n in point.split(', ')])
    return new_track


def nn_interpolation(track, N):
    """
    NN interpolation of a track
    :param track: Track as a Python list of points
    :param N: Number of points of the interpolate track
    :return: Interpolate track
    """
    track_len = len(track)
    dists = [(track_len / float(N)) * i for i in range(N)]
    indices = []
    for dist in dists:
        aux = []
        for i in range(track_len):
            aux.append(abs(dist - i))
        indices.append(np.argmin(aux))
    return [track[i] for i in indices]


def main():
    try:
        if not('.csv' in INPUT): raise Exception('input_format')
        if not('.csv' in OUTPUT): raise Exception('output_format')

        with open(INPUT, 'rb') as input, open(OUTPUT, 'wb') as output:
            reader = csv.reader(input, delimiter=';')
            writer = csv.writer(output, delimiter=';',
                                escapechar='', quoting=csv.QUOTE_NONE)
            writer.writerow(reader.next())
            n = 0
            xn = []
            for track in reader:
                print('Track {}'.format(n))
                track = format_track(track[0])
                new_track = nn_interpolation(track, N)
                if new_track is not None: writer.writerow([new_track])
                if GENERATE_PKL:
                    aux = []
                    for p in new_track:
                        aux.append(p[0])
                        aux.append(p[1])
                    xn.append(aux)
                n += 1

        if GENERATE_PKL:
            with open('{}.pkl'.format(OUTPUT.split('.')[0]), 'w') as output:
                pkl.dump({'xn': np.array(xn)}, output)

    except IOError:
        print('File not found!')
    except Exception as e:
        if e.args[0] == 'input_format': print('Input must be a CSV file')
        elif e.args[0] == 'output_format': print('Output must be a PKL file')
        else:
            print('Unexpected error: {}'.format(sys.exc_info()[0]))
            raise


if __name__ == '__main__': main()
