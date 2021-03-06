# -*- coding: UTF-8 -*-

"""
Gradient Ascent Variational Inference
process to approximate a Mixture of Gaussians (GMM)
"""

from __future__ import absolute_import

import argparse
import csv
import os
import pickle as pkl
import sys
from time import time

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import det, inv
import tensorflow as tf
from scipy.special import psi

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
    * exportVariationalParameters: If true generate a pkl of a dictionary with
                                   the variational parameters inferred
    * exportELBOs: If true generates a pkl wirh the ELBOs list
    
Execution:
    python gmm_gavi.py -dataset data_k2_1000.pkl -k 2 -verbose randomInit
"""

parser = argparse.ArgumentParser(description='GAVI in mixture of gaussians')
parser.add_argument('-maxIter', metavar='maxIter', type=int, default=300)
parser.add_argument('-dataset', metavar='dataset', type=str,
                    default='../../data/synthetic/2D/k2/data_k2_1000.pkl')
parser.add_argument('-k', metavar='k', type=int, default=2)
parser.add_argument('-verbose', dest='verbose', action='store_true')
parser.set_defaults(verbose=False)
parser.add_argument('-randomInit', dest='randomInit', action='store_true')
parser.set_defaults(randomInit=False)
parser.add_argument('-exportAssignments',
                    dest='exportAssignments', action='store_true')
parser.add_argument('-exportVariationalParameters',
                    dest='exportVariationalParameters', action='store_true')
parser.set_defaults(exportVariationalParameters=False)
parser.add_argument('-exportELBOs', dest='exportELBOs', action='store_true')
parser.set_defaults(exportELBOs=False)
args = parser.parse_args()

K = args.k
VERBOSE = args.verbose
INITIAL_LR = 0.1
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
lambda_nu_var = np.zeros(shape=K) + D
lambda_m_var = np.random.uniform(np.min(xn[:, 0]), np.max(xn[:, 0]), (K, D))
lambda_w_var = np.array([np.copy(w_o) for _ in range(K)])

lambda_phi = tf.Variable(lambda_phi_var, trainable=False, dtype=tf.float64)
lambda_pi_var = tf.Variable(lambda_pi_var, dtype=tf.float64)
lambda_beta_var = tf.Variable(lambda_beta_var, dtype=tf.float64)
lambda_nu_var = tf.Variable(lambda_nu_var, dtype=tf.float64)
lambda_m = tf.Variable(lambda_m_var, dtype=tf.float64)
lambda_w_var = tf.Variable(lambda_w_var, dtype=tf.float64)

# Maintain numerical stability
lambda_pi = tf.nn.softplus(lambda_pi_var)
lambda_beta = tf.nn.softplus(lambda_beta_var)
lambda_nu = tf.add(tf.nn.softplus(lambda_nu_var), tf.cast(D, dtype=tf.float64))

# Semidefinite positive matrices definition with Cholesky descomposition
mats = []
for k in range(K):
    aux1 = tf.matrix_set_diag(tf.matrix_band_part(lambda_w_var[k], -1, 0),
                              tf.nn.softplus(tf.diag_part(lambda_w_var[k])))
    mats.append(tf.matmul(aux1, aux1, transpose_b=True))
lambda_w = tf.convert_to_tensor(mats)

alpha_o = tf.convert_to_tensor(alpha_o, dtype=tf.float64)
nu_o = tf.convert_to_tensor(nu_o, dtype=tf.float64)
w_o = tf.convert_to_tensor(w_o, dtype=tf.float64)
m_o = tf.convert_to_tensor(m_o, dtype=tf.float64)
beta_o = tf.convert_to_tensor(beta_o, dtype=tf.float64)

# Evidence Lower Bound definition
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
                   tf.add(tf.digamma(tf.div(tf.subtract(
                       lambda_nu, tf.cast(1., dtype=tf.float64)),
                       tf.cast(2., dtype=tf.float64))),
                       tf.add(tf.multiply(tf.cast(2., dtype=tf.float64),
                                          tf.cast(tf.log(2.),
                                                  dtype=tf.float64)), logdet)))
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
                    tf.lgamma(tf.div(
                        tf.subtract(lambda_nu, tf.cast(1., dtype=tf.float64)),
                        tf.cast(2., dtype=tf.float64)))))
logB = tf.add(
    tf.multiply(tf.div(lambda_nu, tf.cast(2., dtype=tf.float64)), logdet),
    tf.add(tf.multiply(lambda_nu, tf.log(tf.cast(2., dtype=tf.float64))), aux))
h5 = tf.reduce_sum(tf.subtract(tf.add(logB, lambda_nu),
                               tf.multiply(tf.div(tf.subtract(
                                   lambda_nu,
                                   tf.cast(3., dtype=tf.float64)),
                                   tf.cast(2., dtype=tf.float64)), logDeltak)))
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
                         tf.lgamma(tf.div(tf.subtract(
                             nu_o, tf.cast(1., dtype=tf.float64)),
                                          tf.cast(2., dtype=tf.float64)))))))
e5 = tf.reduce_sum(tf.add(-logB, tf.subtract(
    tf.multiply(tf.div(tf.subtract(nu_o, tf.cast(3., dtype=tf.float64)),
                       tf.cast(2., dtype=tf.float64)), logDeltak),
    tf.multiply(tf.div(lambda_nu, tf.cast(2., dtype=tf.float64)), traces))))
LB = e1 + e2 + e3 + e4 + e5 + h1 + h2 + h4 + h5

# Optimizer definition
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(INITIAL_LR, global_step,
                                           100, 0.96, staircase=True)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
grads_and_vars = optimizer.compute_gradients(
    -LB, var_list=[lambda_pi_var, lambda_m,
                   lambda_beta_var, lambda_nu_var, lambda_w_var])
train = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

# Summaries definition
tf.summary.histogram('lambda_pi', lambda_pi)
tf.summary.histogram('lambda_m', lambda_m)
tf.summary.histogram('lambda_beta', lambda_beta)
tf.summary.histogram('lambda_nu', lambda_nu)
tf.summary.histogram('lambda_w', lambda_w)
merged = tf.summary.merge_all()
file_writer = tf.summary.FileWriter('/tmp/tensorboard/', tf.get_default_graph())


def dirichlet_expectation_k(alpha, k):
    """
    Dirichlet expectation computation
    \Psi(\alpha_{k}) - \Psi(\sum_{i=1}^{K}(\alpha_{i}))
    """
    return psi(alpha[k] + np.finfo(np.float32).eps) - psi(np.sum(alpha))


def softmax(x):
    """
    Softmax computation
    e^{x} / sum_{i=1}^{K}(e^x_{i})
    """
    e_x = np.exp(x - np.max(x))
    return (e_x + np.finfo(np.float32).eps) / \
           (e_x.sum(axis=0) + np.finfo(np.float32).eps)


def update_lambda_phi(lambda_phi, lambda_pi, lambda_m,
                      lambda_nu, lambda_w, lambda_beta, xn, N, K, D):
    """
    Update lambda_phi
    softmax[dirichlet_expectation(lambda_pi) +
            lambda_m * lambda_nu * lambda_w^{-1} * x_{n} -
            1/2 * lambda_nu * lambda_w^{-1} * x_{n} * x_{n}.T -
            1/2 * lambda_beta^{-1} -
            lambda_nu * lambda_m.T * lambda_w^{-1} * lambda_m +
            D/2 * log(2) +
            1/2 * sum_{i=1}^{D}(\Psi(lambda_nu/2 + (1-i)/2)) -
            1/2 log(|lambda_w|)]
    """
    for n in range(N):
        for k in range(K):
            inv_lambda_w = inv(lambda_w[k, :, :])
            lambda_phi[n, k] = dirichlet_expectation_k(lambda_pi, k)
            lambda_phi[n, k] += np.dot(lambda_m[k, :], np.dot(
                lambda_nu[k] * inv_lambda_w, xn[n, :]))
            lambda_phi[n, k] -= np.trace(
                np.dot((1 / 2.) * lambda_nu[k] * inv_lambda_w,
                       np.outer(xn[n, :], xn[n, :])))
            lambda_phi[n, k] -= (D / 2.) * (1 / lambda_beta[k])
            lambda_phi[n, k] -= (1. / 2.) * np.dot(
                np.dot(lambda_nu[k] * lambda_m[k, :].T, inv_lambda_w),
                lambda_m[k, :])
            lambda_phi[n, k] += (D / 2.) * np.log(2.)
            lambda_phi[n, k] += (1 / 2.) * np.sum(
                [psi((lambda_nu[k] / 2.) + ((1 - i) / 2.)) for i in range(D)])
            lambda_phi[n, k] -= (1 / 2.) * np.log(det(lambda_w[k, :, :]))
        lambda_phi[n, :] = softmax(lambda_phi[n, :])
    return lambda_phi


def main():

    # Plot configs
    if VERBOSE:
        plt.ion()
        plt.style.use('seaborn-darkgrid')
        fig = plt.figure(figsize=(10, 10))
        ax_spatial = fig.add_subplot(1, 1, 1)
        circs = []
        sctZ = None

    # Inference
    init = tf.global_variables_initializer()
    sess.run(init)
    lbs = []
    n_iters = 0

    phi_out = sess.run(lambda_phi)
    pi_out = sess.run(lambda_pi)
    m_out = sess.run(lambda_m)
    nu_out = sess.run(lambda_nu)
    w_out = sess.run(lambda_w)
    beta_out = sess.run(lambda_beta)

    for _ in range(args.maxIter):

        # Update local variational parameter lambda_phi
        new_lambda_phi = update_lambda_phi(phi_out, pi_out, m_out, nu_out,
                                           w_out, beta_out, xn, N, K, D)
        sess.run(lambda_phi.assign(new_lambda_phi))

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
            covs = []
            for k in range(K):
                w_out[k, 0, 0] = 1.0 / w_out[k, 0, 0]
                w_out[k, 1, 1] = 1.0 / w_out[k, 1, 1]
                covs.append(w_out[k, :, :] / (nu_out[k] - D - 1))
            ax_spatial, circs, sctZ = plot_iteration(ax_spatial, circs,
                                                     sctZ, m_out,
                                                     covs, xn,
                                                     n_iters, K)

            # Break condition
            improve = lb - lbs[n_iters - 1] if n_iters > 0 else lb
            if VERBOSE: print('Improve: {}'.format(improve))
            if n_iters > 0 and 0 <= improve < THRESHOLD: break

        n_iters += 1
        file_writer.add_summary(mer, n_iters)

    zn = np.array([np.argmax(phi_out[n, :]) for n in xrange(N)])

    if VERBOSE:
        print('\n******* RESULTS *******')
        for k in range(K):
            print('Mu k{}: {}'.format(k, m_out[k, :]))
        final_time = time()
        exec_time = final_time - init_time
        print('Time: {} seconds'.format(exec_time))
        print('Iterations: {}'.format(n_iters))
        print('ELBOs: {}'.format(lbs[len(lbs)-10:len(lbs)]))
        if D == 2: plt.savefig('generated/gavi_plot.png')
        if D == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xn[:, 0], xn[:, 1], xn[:, 2],
                       c=zn, cmap=cm.gist_rainbow, s=5)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.show()
        plt.gcf().clear()
        plt.plot(np.arange(len(lbs)), lbs)
        plt.ylabel('ELBO')
        plt.xlabel('Iterations')
        plt.savefig('generated/gavi_elbos.png')

    if args.exportAssignments:
        with open('generated/gavi_assignments.csv', 'wb') as output:
            writer = csv.writer(output, delimiter=';', quotechar='',
                                escapechar='\\', quoting=csv.QUOTE_NONE)
            writer.writerow(['zn'])
            for i in range(len(zn)):
                writer.writerow([zn[i]])

    if args.exportVariationalParameters:
        with open('generated/gavi_variational_parameters.pkl', 'w') as output:
            pkl.dump({'lambda_pi': pi_out, 'lambda_m': m_out,
                      'lambda_beta': beta_out, 'lambda_nu': nu_out,
                      'lambda_w': w_out, 'K': K, 'D': D}, output)

    if args.exportELBOs:
        with open('generated/gavi_elbos.pkl', 'w') as output:
            pkl.dump({'elbos': lbs, 'iter_time': exec_time/n_iters}, output)

if __name__ == '__main__': main()
