# -*- coding: UTF-8 -*-

"""
Probabilistic Principal Component Analysis
with automatic relevance determination
"""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import pickle as pkl
import sys

import edward as ed
import numpy as np
import tensorflow as tf
from edward.models import Normal

"""
Parameters:
    * input: Input path (CSV with ; delimiter)
    * output: Output path (PKL file)

Execution:
    python ppca.py -input mallorca_nnint50.csv 
                   -output generated/mallorca_nnint50_ppca.pkl -k 100
"""

parser = argparse.ArgumentParser(description='PCA')
parser.add_argument('-input', metavar='input', type=str, default='')
parser.add_argument('-output', metavar='output', type=str, default='')
parser.add_argument('-k', metavar='k', type=int)
args = parser.parse_args()

K = args.k
N_ITERS = 10000
N_SAMPLES = 10


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

            # DATA
            reader = csv.reader(input, delimiter=';')
            reader.next()
            n = 0
            xn = []
            for track in reader:
                print('Track {}'.format(n))
                track = format_track(track[0])
                xn.append(track)
                n += 1
            xn = np.asarray(xn)         # N x D
            xn = xn.T                   # D x N

            D = len(xn)
            N = len(xn[0])

            # MODEL
            ds = tf.contrib.distributions
            sigma = ed.models.Gamma(1.0, 1.0)

            alpha = ed.models.Gamma(tf.ones([K]), tf.ones([K]))
            w = Normal(mu=tf.zeros([D, K]),
                       sigma=tf.reshape(tf.tile(alpha, [D]), [D, K]))
            z = Normal(mu=tf.zeros([K, N]), sigma=tf.ones([K, N]))
            mu = Normal(mu=tf.zeros([D]), sigma=tf.ones([D]))
            x = Normal(mu=tf.matmul(w, z)
                          + tf.transpose(tf.reshape(tf.tile(mu, [N]), [N, D])),
                       sigma=sigma * tf.ones([D, N]))

            # INFERENCE
            qalpha = ed.models.TransformedDistribution(
                distribution=ed.models.NormalWithSoftplusSigma(
                    mu=tf.Variable(tf.random_normal([K])),
                    sigma=tf.Variable(tf.random_normal([K]))),
                bijector=ds.bijector.Exp(),
                name='qalpha')

            qw = Normal(mu=tf.Variable(tf.random_normal([D, K])),
                        sigma=tf.nn.softplus(
                            tf.Variable(tf.random_normal([D, K]))))
            qz = Normal(mu=tf.Variable(tf.random_normal([K, N])),
                        sigma=tf.nn.softplus(
                            tf.Variable(tf.random_normal([K, N]))))

            data_mean = np.mean(xn, axis=1).astype(np.float32, copy=False)

            qmu = Normal(mu=tf.Variable(data_mean + tf.random_normal([D])),
                         sigma=tf.nn.softplus(
                             tf.Variable(tf.random_normal([D]))))

            qsigma = ed.models.TransformedDistribution(
                distribution=ed.models.NormalWithSoftplusSigma(
                    mu=tf.Variable(0.0), sigma=tf.Variable(1.0)),
                bijector=ds.bijector.Exp(), name='qsigma')

            inference = ed.KLqp({alpha: qalpha, w: qw, z: qz,
                                 mu: qmu, sigma: qsigma}, data={x: xn})
            inference.run(n_iter=N_ITERS, n_samples=N_SAMPLES)

            alphas = tf.exp(qalpha.distribution.mean()).eval()
            alphas.sort()
            mean_alphas = np.mean(alphas)
            print('Alphas: {}'.format(alphas))

            points = qz.eval()
            xn_new = []
            for i in range(len(alphas)):
                if alphas[i] > (mean_alphas * 1.2):
                    xn_new.append(points[i])
            xn_new = np.asarray(xn_new).T

            print('New points: {}'.format(xn_new))
            print('Number of points: {}'.format(len(xn_new)))
            print('Point dimensions: {}'.format(len(xn_new[0])))

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
