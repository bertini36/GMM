# -*- coding: UTF-8 -*-

"""
Probabilistic Principal Component Analysis
with automatic relevance determination
"""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import sys

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from edward.models import Normal

"""
Parameters:
    * input: Input path (CSV with ; delimiter)
    * output: Output path (PKL file)

Execution:
    python ppca.py -input porto_int50.csv -output porto_pca.pkl
"""

parser = argparse.ArgumentParser(description='PCA')
parser.add_argument('-input', metavar='input', type=str, default='')
parser.add_argument('-output', metavar='output', type=str, default='')
parser.add_argument('-k', metavar='k', type=int)
args = parser.parse_args()

INPUT = args.input
OUTPUT = args.output
K = args.k
N_ITERS = 10000
N_SAMPLES = 10


def build_toy_dataset(N, D, K, sigma=1):
    x_train = np.zeros([D, N])
    w = np.zeros([D, K])
    for k in range(K):
        w[k, k] = 1.0 / (k+1)
        w[k+1, k] = -1.0 / (k+1)
    z = np.random.normal(0.0, 1.0, size=(K, N))
    mean = np.dot(w, z)
    shift = np.zeros([D])
    shift[1] = 10
    for d in range(D):
        for n in range(N):
            x_train[d, n] = np.random.normal(mean[d, n], sigma) + shift[d]
    return x_train.astype(np.float32, copy=False)


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

            print('Inferred principal axes (columns):')
            print('Mean: {}'.format(qw.mean().eval()))
            print('Variance: {}'.format(qw.variance().eval()))

            print('Inferred center:')
            print('Mean: {}'.format(qmu.mean().eval()))
            print('Variance: {}'.format(qmu.variance().eval()))

            print('Length new points: {}'.format(len(qz.eval()[0])))
            print('Dimensions new points: {}'.format(len(qz.eval())))
            print('New points: ')
            print(qz.eval())

            alphas = tf.exp(qalpha.distribution.mean()).eval()
            print('Alphas: {}'.format(alphas))
            alphas.sort()
            plt.plot(range(alphas.size), alphas)
            plt.show()

            plt.hist(qalpha.sample(1000).eval(), bins=30)
            plt.show()

            # TODO: Establecer umbral (0.1) alphas y coger solo las
            #       rows/dimensiones de qz que superen ese umbral y
            #       construir el PKL tras invertir la matriz

    except IOError:
        print('File not found!')
    except Exception as e:
        if e.args[0] == 'input_format':
            print('Input must be a CSV file')
        elif e.args[0] == 'output_format':
            print('Output must be a PKL file')
        else:
            print('Unexpected error: {}'.format(sys.exc_info()[0]))
            raise


if __name__ == '__main__': main()