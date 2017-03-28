# -*- coding: UTF-8 -*-

"""
Principal Componen Analysis
"""

import csv
import pickle as pkl

import numpy as np
from sklearn.decomposition import PCA

DATA_DIRECTORY = 'mallorca/mallorca_int.csv'
N_COMPONENTS = 50


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
    with open(DATA_DIRECTORY, 'rb') as input:
        reader = csv.reader(input, delimiter=';')
        reader.next()
        n = 0
        xn = []
        for track in reader:
            print('Track {}'.format(n))
            track = format_track(track[0])
            xn.append(track)
            n += 1
        print('Lenght xn[0]: {}'.format(len(xn[0])))
        pca = PCA(n_components=N_COMPONENTS)
        xn_new = pca.fit_transform(xn)
        print('Lenght xn_new[0]: {}'.format(len(xn_new[0])))
        with open('mallorca/mallorca_pca.pkl', 'w') as output:
            pkl.dump({'xn': np.array(xn_new)}, output)


if __name__ == '__main__': main()
