# -*- coding: UTF-8 -*-

"""
Gradient Ascent Variational Inference
process to approximate a Mixture of Gaussians (GMM)
Tensorflow implementation
"""

from __future__ import absolute_import

import argparse
import os
import pickle as pkl
import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import dirichlet_expectation, log_, log_beta_function, multilgamma

from common import init_kmeans, generate_random_positive_matrix
from viz import plot_iteration

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

parser = argparse.ArgumentParser(description='GAVI in mixture of gaussians')
parser.add_argument('-maxIter', metavar='maxIter', type=int, default=1000000)
parser.add_argument('-dataset', metavar='dataset', type=str,
                    default='../../data/synthetic/2D/k2/data_k2_100.pkl')
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
LR = 0.5
THRESHOLD = 1e-6

sess = tf.Session()

# Get data
with open('{}'.format(args.dataset), 'r') as inputfile:
    data = pkl.load(inputfile)
    xn = data['xn']
N, D = xn.shape

if VERBOSE: init_time = time()

# Priors
alpha_o = np.array([1.0] * K)
nu_o = np.array([float(D)])
w_o = generate_random_positive_matrix(D)
m_o = np.array([0.0] * D)
beta_o = np.array([0.7])

# Variational parameters intialization
lambda_phi_var = np.random.dirichlet(alpha_o, N) \
    if args.randomInit else init_kmeans(xn, N, K)
lambda_pi_var = np.zeros(shape=K)
lambda_beta_var = np.zeros(shape=K)
lambda_nu_var = np.zeros(shape=K)
lambda_m_var = np.zeros(shape=(K, D))
lambda_w_var = np.array([np.copy(w_o) for _ in range(K)])

lambda_phi_var = tf.Variable(lambda_phi_var, dtype=tf.float64)
lambda_pi_var = tf.Variable(lambda_pi_var, dtype=tf.float64)
lambda_beta_var = tf.Variable(lambda_beta_var, dtype=tf.float64)
lambda_nu_var = tf.Variable(lambda_nu_var, dtype=tf.float64)
lambda_m = tf.Variable(lambda_m_var, dtype=tf.float64)
lambda_w_var = tf.Variable(lambda_w_var, dtype=tf.float64)

# Maintain numerical stability
lambda_pi = tf.nn.softplus(lambda_pi_var)
lambda_beta = tf.nn.softplus(lambda_beta_var)
lambda_phi = tf.nn.softmax(lambda_phi_var)
lambda_nu = tf.nn.softplus(lambda_nu_var)
lambda_w = tf.convert_to_tensor([tf.matmul(lambda_w_var[k],
                                           tf.transpose(lambda_w_var[k]))
                                 for k in range(K)])

alpha_o = tf.convert_to_tensor(alpha_o, dtype=tf.float64)
nu_o = tf.convert_to_tensor(nu_o, dtype=tf.float64)
w_o = tf.convert_to_tensor(w_o, dtype=tf.float64)
m_o = tf.convert_to_tensor(m_o, dtype=tf.float64)
beta_o = tf.convert_to_tensor(beta_o, dtype=tf.float64)

# Lower Bound definition
e3 = tf.convert_to_tensor(0., dtype=tf.float64)
e2 = tf.convert_to_tensor(0., dtype=tf.float64)
h2 = tf.convert_to_tensor(0., dtype=tf.float64)
e1 = tf.add(-log_beta_function(alpha_o),
            tf.reduce_sum(tf.multiply(
                tf.subtract(alpha_o, tf.ones(K, dtype=tf.float64)),
                dirichlet_expectation(lambda_pi))))
h1 = tf.subtract(log_beta_function(lambda_pi),
                 tf.reduce_sum(tf.multiply(
                     tf.subtract(lambda_pi, tf.ones(K, dtype=tf.float64)),
                     dirichlet_expectation(lambda_pi))))
logdet = tf.log(tf.convert_to_tensor([
    tf.matrix_determinant(lambda_w[k, :, :]) for k in xrange(K)]))
logDeltak = tf.add(tf.digamma(tf.div(lambda_nu, 2.)),
                   tf.add(tf.digamma(tf.div(tf.subtract(lambda_nu,
                                                        tf.cast(1.,
                                                                dtype=tf.float64)),
                                            tf.cast(2., dtype=tf.float64))),
                          tf.add(tf.multiply(tf.cast(2., dtype=tf.float64),
                                             tf.cast(tf.log(2.),
                                                     dtype=tf.float64)),
                                 logdet)))
for n in range(N):
    e2 = tf.add(e2, tf.reduce_sum(
        tf.multiply(lambda_phi[n, :], dirichlet_expectation(lambda_pi))))
    h2 = tf.add(h2, -tf.reduce_sum(
        tf.multiply(lambda_phi[n, :], log_(lambda_phi[n, :]))))
    product = tf.convert_to_tensor([tf.reduce_sum(tf.matmul(
        tf.matmul(tf.reshape(tf.subtract(xn[n, :], lambda_m[k, :]), [1, 2]),
                  lambda_w[k, :, :]),
        tf.reshape(tf.transpose(tf.subtract(xn[n, :], lambda_m[k, :])),
                   [2, 1]))) for k in range(K)])
    aux = tf.transpose(tf.subtract(
        logDeltak, tf.add(tf.multiply(tf.cast(2., dtype=tf.float64),
                                      tf.cast(tf.log(2. * np.pi),
                                              dtype=tf.float64)),
                          tf.add(tf.multiply(lambda_nu, product),
                                 tf.div(tf.cast(2., dtype=tf.float64),
                                        lambda_beta)))))
    e3 = tf.add(e3, tf.reduce_sum(
        tf.multiply(tf.cast(1 / 2., dtype=tf.float64),
                    tf.multiply(lambda_phi[n, :], aux))))
product = tf.convert_to_tensor([tf.reduce_sum(tf.matmul(
    tf.matmul(tf.reshape(tf.subtract(lambda_m[k, :], m_o), [1, 2]),
              lambda_w[k, :, :]),
    tf.reshape(tf.transpose(tf.subtract(lambda_m[k, :], m_o)), [2, 1]))) for
    k in range(K)])
traces = tf.convert_to_tensor([tf.trace(tf.matmul(
    tf.matrix_inverse(w_o), lambda_w[k, :, :])) for k in range(K)])
h4 = tf.reduce_sum(
    tf.add(tf.cast(1., dtype=tf.float64),
           tf.subtract(tf.log(tf.cast(2., dtype=tf.float64) * np.pi),
                       tf.multiply(tf.cast(1. / 2., dtype=tf.float64),
                                   tf.add(tf.cast(tf.log(lambda_beta),
                                                  dtype=tf.float64),
                                          logdet)))))
aux = tf.add(tf.multiply(tf.cast(1. / 2., dtype=tf.float64), tf.log(
    tf.cast(tf.constant(np.pi), dtype=tf.float64))),
             tf.add(tf.lgamma(
                 tf.div(lambda_nu, tf.cast(2., dtype=tf.float64))),
                    tf.lgamma(tf.div(tf.subtract(lambda_nu, tf.cast(1.,
                                                                    dtype=tf.float64)),
                                     tf.cast(2., dtype=tf.float64)))))
logB = tf.add(
    tf.multiply(tf.div(lambda_nu, tf.cast(2., dtype=tf.float64)), logdet),
    tf.add(tf.multiply(lambda_nu, tf.log(tf.cast(2., dtype=tf.float64))),
           aux))
h5 = tf.reduce_sum(tf.subtract(tf.add(logB, lambda_nu),
                               tf.multiply(tf.div(tf.subtract(
                                   lambda_nu,
                                   tf.cast(3., dtype=tf.float64)),
                                   tf.cast(2., dtype=tf.float64)),
                                   logDeltak)))
aux = tf.add(tf.multiply(tf.cast(2., dtype=tf.float64),
                         tf.log(tf.cast(2., dtype=tf.float64) * np.pi)),
             tf.add(tf.multiply(beta_o, tf.multiply(lambda_nu, product)),
                    tf.multiply(tf.cast(2., dtype=tf.float64),
                                tf.div(beta_o, lambda_beta))))
e4 = tf.reduce_sum(tf.multiply(tf.cast(1. / 2., dtype=tf.float64),
                               tf.subtract(
                                   tf.add(tf.log(beta_o), logDeltak), aux)))
logB = tf.add(
    tf.multiply(tf.div(nu_o, tf.cast(2., dtype=tf.float64)),
                tf.log(tf.matrix_determinant(w_o))),
    tf.add(tf.multiply(nu_o, tf.cast(tf.log(2.), dtype=tf.float64)),
           tf.add(tf.multiply(tf.cast(1. / 2., dtype=tf.float64),
                              tf.cast(tf.log(np.pi), dtype=tf.float64)),
                  tf.add(tf.lgamma(
                      tf.div(nu_o, tf.cast(2., dtype=tf.float64))),
                         tf.lgamma(tf.div(tf.subtract(nu_o, tf.cast(1.,
                                                                    dtype=tf.float64)),
                                          tf.cast(2.,
                                                  dtype=tf.float64)))))))
e5 = tf.reduce_sum(tf.add(-logB, tf.subtract(
    tf.multiply(tf.div(tf.subtract(nu_o, tf.cast(3., dtype=tf.float64)),
                       tf.cast(2., dtype=tf.float64)), logDeltak),
    tf.multiply(tf.div(lambda_nu, tf.cast(2., dtype=tf.float64)), traces))))
LB = e1 + e2 + e3 + e4 + e5 + h1 + h2 + h4 + h5

# Optimizer definition
optimizer = tf.train.AdamOptimizer(learning_rate=LR)
grads_and_vars = optimizer.compute_gradients(
    -LB, var_list=[lambda_pi_var, lambda_phi_var, lambda_m,
                   lambda_beta_var, lambda_nu_var, lambda_w_var])
train = optimizer.apply_gradients(grads_and_vars)

# Summaries definition
tf.summary.histogram('lambda_pi', lambda_pi)
tf.summary.histogram('lambda_phi', lambda_phi)
tf.summary.histogram('lambda_m', lambda_m)
tf.summary.histogram('lambda_beta', lambda_beta)
tf.summary.histogram('lambda_nu', lambda_nu)
tf.summary.histogram('lambda_w', lambda_w)
merged = tf.summary.merge_all()
file_writer = tf.summary.FileWriter('/tmp/tensorboard/', tf.get_default_graph())


def main():

    # Plot configs
    if VERBOSE:
        plt.ion()
        fig = plt.figure(figsize=(10, 10))
        ax_spatial = fig.add_subplot(1, 1, 1)
        circs = []
        sctZ = None

    # Inference
    init = tf.global_variables_initializer()
    sess.run(init)
    lbs = []
    n_iters = 0
    for _ in range(args.maxIter):

        # ELBO computation
        _, mer, lb, pi_out, phi_out, m_out, beta_out, nu_out, w_out = sess.run(
            [train, merged, LB, lambda_pi, lambda_phi,
             lambda_m, lambda_beta, lambda_nu, lambda_w])
        lbs.append(lb)

        if VERBOSE:
            print('\n******* ITERATION {} *******'.format(n_iters))
            print('lambda_pi: {}'.format(pi_out))
            print('lambda_phi: {}'.format(phi_out[0:9, :]))
            print('lambda_m: {}'.format(m_out))
            print('lambda_beta: {}'.format(beta_out))
            print('lambda_nu: {}'.format(nu_out))
            print('lambda_w: {}'.format(w_out))
            print('ELBO: {}'.format(lb))
            covs = [(w_out[k, :, :] / (nu_out[k] - D - 1)) for k in range(K)]
            ax_spatial, circs, sctZ = plot_iteration(ax_spatial, circs,
                                                     sctZ, m_out,
                                                     covs, xn,
                                                     n_iters, K)

        # Break condition
        if n_iters > 0 and abs(lb - lbs[n_iters - 1]) < THRESHOLD:
            plt.savefig('generated/plot.png')
            break

        n_iters += 1
        file_writer.add_summary(mer, n_iters)

    if VERBOSE:
        print('\n******* RESULTS *******')
        for k in range(K):
            print('Mu k{}: {}'.format(k, m_out[k, :]))
        final_time = time()
        exec_time = final_time - init_time
        print('Time: {} seconds'.format(exec_time))
        print('Iterations: {}'.format(n_iters))
        print('ELBOs: {}'.format(lbs[len(lbs)-10:len(lbs)]))

if __name__ == '__main__': main()
