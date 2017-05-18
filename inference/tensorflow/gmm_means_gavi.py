# -*- coding: UTF-8 -*-

"""
Gradient Ascent Variational Inference process to approximate a mixture
of gaussians with common variance for all classes
"""

from __future__ import absolute_import

import argparse
import math
import os
import pickle as pkl
import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import dirichlet_expectation, log_beta_function

from viz import plot_iteration

"""
Parameters:
    * maxIter: Max number of iterations
    * dataset: Dataset path
    * k: Number of clusters
    * verbose: Printing time, intermediate variational parameters, plots, ...
    
Execution:
    python gmm_means_gavi.py -dataset data_k4_1000.pkl -k 2 -verbose 
"""

parser = argparse.ArgumentParser(description='GAVI in mixture of gaussians')
parser.add_argument('-maxIter', metavar='maxIter', type=int, default=500)
parser.add_argument('-dataset', metavar='dataset', type=str,
                    default='../../data/synthetic/2D/k2/data_k4_1000.pkl')
parser.add_argument('-k', metavar='k', type=int, default=2)
parser.add_argument('-verbose', dest='verbose', action='store_true')
parser.set_defaults(verbose=False)
args = parser.parse_args()

K = args.k
VERBOSE = args.verbose
LR = 0.01
THRESHOLD = 1e-6

sess = tf.Session()


# Get data
with open('{}'.format(args.dataset), 'r') as inputfile:
    data = pkl.load(inputfile)
    xn = data['xn']
    xn_tf = tf.convert_to_tensor(xn, dtype=tf.float64)
N, D = xn.shape

if VERBOSE: init_time = time()

# Model hyperparameters
alpha_o_aux = [1.0] * K
m_o_aux = np.array([0.0, 0.0])
beta_o_aux = 0.01
delta_o_aux = np.zeros((D, D), long)
np.fill_diagonal(delta_o_aux, 1)

# Priors (TF castings)
alpha_o = tf.convert_to_tensor([alpha_o_aux], dtype=tf.float64)
m_o = tf.convert_to_tensor([list(m_o_aux)], dtype=tf.float64)
beta_o = tf.convert_to_tensor(beta_o_aux, dtype=tf.float64)
delta_o = tf.convert_to_tensor(delta_o_aux, dtype=tf.float64)

# Initializations
lambda_phi_aux = np.random.dirichlet(alpha_o_aux, N)
lambda_pi_aux = alpha_o_aux + np.sum(lambda_phi_aux, axis=0)
lambda_beta_aux = beta_o_aux + np.sum(lambda_phi_aux, axis=0)
lambda_m_aux = np.tile(1. / lambda_beta_aux, (2, 1)).T * \
                  (beta_o_aux * m_o_aux + np.dot(lambda_phi_aux.T, xn))

# Variational parameters
lambda_phi_var = tf.Variable(lambda_phi_aux, dtype=tf.float64)
lambda_pi_var = tf.Variable(lambda_pi_aux, dtype=tf.float64)
lambda_beta_var = tf.Variable(lambda_beta_aux, dtype=tf.float64)
lambda_m = tf.Variable(lambda_m_aux, dtype=tf.float64)

# Maintain numerical stability
lambda_pi = tf.nn.softplus(lambda_pi_var)
lambda_beta = tf.nn.softplus(lambda_beta_var)
lambda_phi = tf.nn.softmax(lambda_phi_var)

# Reshapes
lambda_mu_beta_res = tf.reshape(lambda_beta, [K, 1])

# Lower Bound definition
LB = log_beta_function(lambda_pi)
LB = tf.subtract(LB, log_beta_function(alpha_o))
LB = tf.add(LB, tf.matmul(tf.subtract(alpha_o, lambda_pi),
                          tf.reshape(dirichlet_expectation(lambda_pi),
                                     [K, 1])))
LB = tf.add(LB, tf.multiply(tf.cast(K / 2., tf.float64),
                            tf.log(tf.matrix_determinant(
                                tf.multiply(beta_o, delta_o)))))
LB = tf.add(LB, tf.cast(K * (D / 2.), tf.float64))
for k in range(K):
    a1 = tf.subtract(lambda_m[k, :], m_o)
    a2 = tf.matmul(delta_o, tf.transpose(tf.subtract(lambda_m[k, :], m_o)))
    a3 = tf.multiply(tf.div(beta_o, 2.), tf.matmul(a1, a2))
    a4 = tf.div(tf.multiply(tf.cast(D, tf.float64), beta_o),
                tf.multiply(tf.cast(2., tf.float64), lambda_mu_beta_res[k]))
    a5 = tf.multiply(tf.cast(1 / 2., tf.float64), tf.log(
        tf.multiply(tf.pow(lambda_mu_beta_res[k], 2),
                    tf.matrix_determinant(delta_o))))
    a6 = tf.add(a3, tf.add(a4, a5))
    LB = tf.subtract(LB, a6)
    b1 = tf.transpose(lambda_phi[:, k])
    b2 = dirichlet_expectation(lambda_pi)[k]
    b3 = tf.log(lambda_phi[:, k])
    b4 = tf.multiply(tf.cast(1 / 2., tf.float64), tf.log(
        tf.div(tf.matrix_determinant(delta_o),
               tf.multiply(tf.cast(2., tf.float64), math.pi))))
    b5 = tf.subtract(xn_tf, lambda_m[k, :])
    b6 = tf.matmul(delta_o, tf.transpose(tf.subtract(xn_tf, lambda_m[k, :])))
    b7 = tf.multiply(tf.cast(1 / 2., tf.float64),
                     tf.stack([tf.matmul(b5, b6)[i, i] for i in range(N)]))
    b8 = tf.div(tf.cast(D, tf.float64),
                tf.multiply(tf.cast(2., tf.float64), lambda_beta[k]))
    b9 = tf.subtract(tf.subtract(tf.add(tf.subtract(b2, b3), b4), b7), b8)
    b1 = tf.reshape(b1, [1, N])
    b9 = tf.reshape(b9, [N, 1])
    LB = tf.add(LB, tf.reshape(tf.matmul(b1, b9), [1]))

# Optimizer definition
optimizer = tf.train.AdamOptimizer(learning_rate=LR)
grads_and_vars = optimizer.compute_gradients(-LB,  var_list=[lambda_phi_var,
                                                             lambda_pi_var,
                                                             lambda_beta_var,
                                                             lambda_m])
train = optimizer.apply_gradients(grads_and_vars)

# Summaries definition
tf.summary.histogram('lambda_phi', lambda_phi)
tf.summary.histogram('lambda_pi', lambda_pi)
tf.summary.histogram('lambda_m', lambda_m)
tf.summary.histogram('lambda_beta', lambda_beta)
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
        _, mer, lb, m_out, beta_out, pi_out, phi_out = sess.run(
            [train, merged, LB, lambda_m, lambda_beta, lambda_pi, lambda_phi])
        lbs.append(lb[0][0])

        if VERBOSE:
            print('\n******* ITERATION {} *******'.format(n_iters))
            print('lambda_pi: {}'.format(pi_out))
            print('lambda_beta: {}'.format(beta_out))
            print('lambda_m: {}'.format(m_out))
            print('lambda_phi: {}'.format(phi_out[0:9, :]))
            print('ELBO: {}'.format(lb))
            ax_spatial, circs, sctZ = plot_iteration(ax_spatial, circs, sctZ,
                                                     sess.run(lambda_m),
                                                     sess.run(delta_o),
                                                     xn, n_iters, K)

        # Break condition
        improve = lb - lbs[n_iters - 1]
        if VERBOSE: print('Improve: {}'.format(improve))
        if (n_iters == (args.maxIter - 1)) \
                or (n_iters > 0 and 0 < improve < THRESHOLD):
            if VERBOSE and D == 2: plt.savefig('generated/plot.png')
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
        print('ELBOs: {}'.format(lbs))


if __name__ == '__main__': main()
