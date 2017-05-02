# -*- coding: UTF-8 -*-

"""
Average time of the Porto tracks dataset
"""

import argparse
import csv
import sys

import numpy as np

"""
Parameters:
    * input: Input path (CSV with ; delimiter)
    
Execution:
    python average.py -input mallorca.csv
"""

parser = argparse.ArgumentParser(description='Average')
parser.add_argument('-input', metavar='input', type=str, default='')
args = parser.parse_args()

csv.field_size_limit(sys.maxsize)


def format_track(track):
    """
    Format track from String to coordinates list
    :param track: Track as a string
    :return: Track as a Python list of coordinates
    """
    new_track = []
    for point in track.split('[[')[1].split(']]')[0].split('], ['):
        new_track.append([float(n) for n in point.split(', ')])
    return new_track


def main():
    try:
        if not ('.csv' in args.input): raise Exception('input_format')

        with open(args.input, 'rb') as input:
            reader = csv.reader(input, delimiter=';')
            reader.next()

            n_points = []
            for track in reader:
                n_points.append(len(format_track(track[0])))

            av_points = np.mean(n_points)
            print('n_points: {}'.format(n_points))
            print('Average points: {}'.format(av_points))
            print('Average time: {} min'.format((av_points * 15) / 60))

    except IOError:
        print('File not found!')
    except Exception as e:
        if e.args[0] == 'input_format': print('Input must be a CSV file')
        else:
            print('Unexpected error: {}'.format(sys.exc_info()[0]))
            raise

if __name__ == '__main__': main()
