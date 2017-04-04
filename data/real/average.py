# -*- coding: UTF-8 -*-

"""
Average time of the Porto tracks dataset
"""

import csv
import sys

import numpy as np

csv.field_size_limit(sys.maxsize)
DATASET = '../../data/real/mallorca/mallorca.csv'


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
    with open(DATASET, 'rb') as input:
        reader = csv.reader(input, delimiter=';')
        reader.next()

        n_points = []
        for track in reader:
            n_points.append(len(format_track(track[0])))

        av_points = np.mean(n_points)
        print('Average points: {}'.format(av_points))
        print('Average time: {} min'.format((av_points * 15) / 60))

if __name__ == '__main__': main()
