# -*- coding: UTF-8 -*-

"""
KNN interpolation
"""

import csv
import sys
import numpy as np
from scipy.spatial import distance

DATA_DIRECTORY = 'mallorca/mallorca.csv'
N = 50

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


def track_distance(track):
    """
    Function that calculate the track length
    :param track: Track as a Python list of points
    :return: Scalar distance
    """
    dist = 0
    for i in range(len(track) - 1):
        dist += distance.euclidean(track[i], track[i + 1])
    return dist


def knn_interpolation(track, N):
    """
    KNN interpolation of a track
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
    with open(DATA_DIRECTORY, 'rb') as input, \
            open('mallorca/mallorca_int.csv', 'wb') as output:
        reader = csv.reader(input, delimiter=';')
        writer = csv.writer(output, delimiter=';',
                            escapechar='', quoting=csv.QUOTE_NONE)
        writer.writerow(reader.next())
        n = 0
        for track in reader:
            print('Track {}'.format(n))
            track = format_track(track[0])
            new_track = knn_interpolation(track, N)
            if new_track is not None: writer.writerow([new_track])
            n += 1


if __name__ == '__main__': main()
