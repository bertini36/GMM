# -*- coding: UTF-8 -*-

"""
Gradient Ascent Variational Inference
process to approximate a Mixture of Gaussians (GMM)
Tensorflow implementation
[DOING]
"""

from __future__ import absolute_import

import argparse
import pickle as pkl
import sys
from time import time

import numpy as np
import tensorflow as tf

from utils import multilgamma

"""
Parameters:
    * maxIter: Max number of iterations
    * dataset: Dataset path
    * k: Number of clusters
    * verbose: Printing time, intermediate variational parameters, plots, ...
    * randomInit: Init assignations randomly or with Kmeans
    * exportAssignments: If true generate a csv with the cluster assignments

Execution:
    python gmm_gavi.py
        -dataset ../../data/real/mallorca/mallorca_pca30.pkl
        -k 2 --verbose --no-randomInit --exportAssignments
"""

parser = argparse.ArgumentParser(description='CAVI in mixture of gaussians')
parser.add_argument('-maxIter', metavar='maxIter', type=int, default=100)
parser.add_argument('-dataset', metavar='dataset', type=str, default='')
parser.add_argument('-k', metavar='k', type=int, default=2)
parser.add_argument('-verbose', dest='verbose', action='store_true')
parser.set_defaults(verbose=False)
parser.add_argument('-randomInit', dest='randomInit', action='store_true')
parser.set_defaults(randomInit=False)
parser.add_argument('-exportAssignments',
                    dest='exportAssignments', action='store_true')
parser.set_defaults(exportAssignments=False)
args = parser.parse_args()

K = args.k
VERBOSE = args.verbose
THRESHOLD = 1e-6
LR = 0.01

sess = tf.Session()


def elbo(lambda_phi, lambda_pi, lambda_beta, lambda_nu,
         lambda_w, alpha_o, beta_o, nu_o, w_o, N, D):
    """
    ELBO computation
    """
    DOS = tf.constant(2., dtype=tf.float64)
    N_t = tf.cast(N, dtype=tf.float64)
    D_t = tf.cast(D, dtype=tf.float64)

    lb = tf.lgamma(tf.reduce_sum(alpha_o))
    lb = tf.subtract(lb, tf.reduce_sum(tf.lgamma(alpha_o)))
    lb = tf.subtract(lb, tf.lgamma(tf.reduce_sum(lambda_pi)))
    lb = tf.add(lb, tf.reduce_sum(tf.lgamma(lambda_pi)))
    lb = tf.subtract(lb,
                     tf.multiply(N_t, tf.multiply(tf.divide(D_t, DOS), tf.log(
                         tf.multiply(DOS, np.pi)))))
    for k in range(K):
        lb = tf.add(lb, tf.add(
            tf.div(tf.multiply(-nu_o, tf.multiply(D_t, tf.log(DOS))), DOS),
            tf.div(tf.multiply(lambda_nu[k], tf.multiply(D_t, tf.log(DOS))),
                   DOS)
        ))
        lb = tf.add(lb, tf.add(
            -multilgamma(tf.div(nu_o, DOS), D, D_t),
            multilgamma(tf.div(lambda_nu[k], DOS), D, D_t)
        ))
        lb = tf.add(lb, tf.subtract(
            tf.multiply(tf.div(D_t, DOS), tf.log(tf.abs(beta_o))),
            tf.multiply(tf.div(D_t, DOS), tf.log(tf.abs(lambda_beta[k])))
        ))
        lb = tf.add(lb, tf.subtract(
            tf.multiply(tf.div(nu_o, DOS), tf.log(tf.matrix_determinant(w_o))),
            tf.multiply(tf.div(lambda_nu[k], DOS),
                        tf.log(tf.matrix_determinant(lambda_w[k, :, :])))
        ))
        lb = tf.subtract(lb, tf.reduce_sum(tf.multiply(
            tf.transpose(tf.log(lambda_phi[:, k])),
            lambda_phi[:, k]
        )))
    return lb


def main():
    try:
        if not ('.pkl' in args.dataset): raise Exception('input_format')

        # Get data
        with open('{}'.format(args.dataset), 'r') as inputfile:
            data = pkl.load(inputfile)
            xn = data['xn']
        N, D = xn.shape

        if VERBOSE: init_time = time()

        # Priors
        alpha_o = np.array([1.0] * K)
        nu_o = np.array([float(D)])
        w_o = np.array([[2, 1], [3, 2]])
        m_o = np.array([0.0] * D)
        beta_o = np.array([0.7])

        # Variational parameters intialization
        lambda_phi = np.random.dirichlet(alpha_o, N)
        lambda_pi = np.zeros(shape=K)
        lambda_beta = np.zeros(shape=K)
        lambda_nu = np.zeros(shape=K)
        lambda_m = np.zeros(shape=(K, D))
        lambda_w = np.zeros(shape=(K, D, D))

        lambda_phi = tf.Variable(lambda_phi, dtype=tf.float64)
        lambda_pi = tf.Variable(lambda_pi, dtype=tf.float64)
        lambda_beta = tf.Variable(lambda_beta, dtype=tf.float64)
        lambda_nu = tf.Variable(lambda_nu, dtype=tf.float64)
        lambda_m = tf.Variable(lambda_m, dtype=tf.float64)
        lambda_w = tf.Variable(lambda_w, dtype=tf.float64)

        alpha_o = tf.convert_to_tensor(alpha_o, dtype=tf.float64)
        nu_o = tf.convert_to_tensor(nu_o, dtype=tf.float64)
        w_o = tf.convert_to_tensor(w_o, dtype=tf.float64)
        m_o = tf.convert_to_tensor(m_o, dtype=tf.float64)
        beta_o = tf.convert_to_tensor(beta_o, dtype=tf.float64)

        init = tf.global_variables_initializer()
        sess.run(init)

    except IOError:
        print('File not found!')
    except Exception as e:
        if e.args[0] == 'input_format': print('Input must be a pkl file')
        elif e.args[0] == 'degrees_of_freedom':
            print('Degrees of freedom can not be smaller than D!')
        else:
            print('Unexpected error: {}'.format(sys.exc_info()[0]))
            raise


if __name__ == '__main__': main()
