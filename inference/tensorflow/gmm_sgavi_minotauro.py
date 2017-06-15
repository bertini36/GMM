# -*- coding: UTF-8 -*-

"""
Sthocastic Gradient Ascent Variational Inference
process to approximate a Mixture of Gaussians (GMM)
"""

from __future__ import absolute_import

import argparse
import csv
import pickle as pkl
from time import time

import numpy as np
import tensorflow as tf
from numpy.linalg import det, inv
from scipy import random
from scipy.special import psi
from sklearn.cluster import KMeans

tf.logging.set_verbosity(tf.logging.INFO)

"""
Parameters:
    * maxIter: Max number of iterations
    * dataset: Dataset path
    * k: Number of clusters
    * bs: Batch size
    * verbose: Printing time, intermediate variational parameters, plots, ...
    * randomInit: Init assignations randomly or with Kmeans
    * exportAssignments: If true generate a csv with the cluster assignments
    * exportVariationalParameters: If true generates a pkl of a dictionary with
                                   the variational parameters inferred
    * exportELBOs: If true generates a pkl wirh the ELBOs list
    * optimizer: Gradient optimizer

Execution:
    python gmm_sgavi_minotauro.py -dataset data_k2_1000.pkl 
                                  -k 2 -verbose -bs 100 
"""

parser = argparse.ArgumentParser(description='Sthocastic GAVI in'
                                             ' mixture of gaussians')
parser.add_argument('-maxIter', metavar='maxIter', type=int, default=300)
parser.add_argument('-dataset', metavar='dataset', type=str,
                    default='../../data/synthetic/2D/k2/data_k2_1000.pkl')
parser.add_argument('-k', metavar='k', type=int, default=2)
parser.add_argument('-bs', metavar='bs', type=int, default=100)
parser.add_argument('-verbose', dest='verbose', action='store_true')
parser.set_defaults(verbose=False)
parser.add_argument('-randomInit', dest='randomInit', action='store_true')
parser.set_defaults(randomInit=False)
parser.add_argument('-exportAssignments',
                    dest='exportAssignments', action='store_true')
parser.set_defaults(exportAssignments=False)
parser.add_argument('-exportVariationalParameters',
                    dest='exportVariationalParameters', action='store_true')
parser.set_defaults(exportVariationalParameters=False)
parser.add_argument('-exportELBOs', dest='exportELBOs', action='store_true')
parser.set_defaults(exportELBOs=False)
parser.add_argument('-optimizer', metavar='optimizer',
                    type=str, default='rmsprop')
args = parser.parse_args()

K = args.k
VERBOSE = args.verbose
INITIAL_LR = 0.1
THRESHOLD = 1e-6
BATCH_SIZE = args.bs

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))


def dirichlet_expectation(alpha):
    """
    Dirichlet expectation computation
    \Psi(\alpha_{k}) - \Psi(\sum_{i=1}^{K}(\alpha_{i}))
    """
    return tf.subtract(tf.digamma(tf.add(alpha, np.finfo(np.float32).eps)),
                       tf.digamma(tf.reduce_sum(alpha)))


def multilgamma(a, D, D_t):
    """
    ln multigamma Tensorflow implementation
    """
    res = tf.multiply(tf.multiply(D_t, tf.multiply(tf.subtract(D_t, 1),
                                                   tf.cast(0.25,
                                                           dtype=tf.float64))),
                      tf.log(tf.cast(np.pi, dtype=tf.float64)))
    res += tf.reduce_sum(tf.lgamma([tf.subtract(a, tf.div(
        tf.subtract(tf.cast(j, dtype=tf.float64),
                    tf.cast(1., dtype=tf.float64)),
        tf.cast(2., dtype=tf.float64))) for j in range(1, D + 1)]), axis=0)
    return res


def log_(x):
    return tf.log(tf.add(x, np.finfo(np.float32).eps))


def log_beta_function(x):
    """
    Log beta function
    ln(\gamma(x)) - ln(\gamma(\sum_{i=1}^{N}(x_{i}))
    """
    return tf.subtract(
        tf.reduce_sum(tf.lgamma(tf.add(x, np.finfo(np.float32).eps))),
        tf.lgamma(tf.reduce_sum(tf.add(x, np.finfo(np.float32).eps))))


def generate_random_positive_matrix(D):
    """
    Generate a random semidefinite positive matrix
    :param D: Dimension
    :return: DxD matrix
    """
    aux = random.rand(D, D)
    return np.dot(aux, aux.transpose())


def init_kmeans(xn, N, K):
    """
    Init points assignations (lambda_phi) with Kmeans clustering
    """
    lambda_phi = 0.1 / (K - 1) * np.ones((N, K))
    labels = KMeans(K).fit(xn).predict(xn)
    for i, lab in enumerate(labels):
        lambda_phi[i, lab] = 0.9
    return lambda_phi


# Get data
with open('{}'.format(args.dataset), 'rb') as inputfile:
    data = pkl.load(inputfile, encoding='latin1')
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
lambda_m_var = np.random.rand(K, D)
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
lambda_nu = tf.add(tf.nn.softplus(lambda_nu_var),
                   tf.cast(D, dtype=tf.float64))

# Semidefinite positive matrices definition with Cholesky descomposition
mats = []
for k in range(K):
    aux1 = tf.matrix_set_diag(tf.matrix_band_part(lambda_w_var[k], -1, 0),
                              tf.nn.softplus(tf.diag_part(lambda_w_var[k])))
    mats.append(tf.matmul(aux1, aux1, transpose_b=True))
lambda_w = tf.convert_to_tensor(mats)

idx_tensor = tf.placeholder(tf.int32, shape=(BATCH_SIZE))

alpha_o = tf.convert_to_tensor(alpha_o, dtype=tf.float64)
nu_o = tf.convert_to_tensor(nu_o, dtype=tf.float64)
w_o = tf.convert_to_tensor(w_o, dtype=tf.float64)
m_o = tf.convert_to_tensor(m_o, dtype=tf.float64)
beta_o = tf.convert_to_tensor(beta_o, dtype=tf.float64)

# Evidence Lower Bound definition
det1 = [tf.matrix_determinant(lambda_w[k, :, :]) for k in range(K)]
inv1 = tf.matrix_inverse(w_o)
det2 = tf.matrix_determinant(w_o)

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
logdet = tf.log(tf.convert_to_tensor(det1))
logDeltak = tf.add(tf.digamma(tf.div(lambda_nu, 2.)),
                   tf.add(tf.digamma(tf.div(tf.subtract(
                       lambda_nu, tf.cast(1., dtype=tf.float64)),
                       tf.cast(2., dtype=tf.float64))),
                       tf.add(tf.multiply(tf.cast(2., dtype=tf.float64),
                                          tf.cast(tf.log(2.),
                                                  dtype=tf.float64)),
                              logdet)))
product = tf.convert_to_tensor([tf.reduce_sum(tf.matmul(
    tf.matmul(tf.reshape(tf.subtract(lambda_m[k, :], m_o), [1, 2]),
              lambda_w[k, :, :]),
    tf.reshape(tf.transpose(tf.subtract(lambda_m[k, :],
                                        m_o)), [2, 1]))) for k in range(K)])
for i in range(BATCH_SIZE):
    n = idx_tensor[i]
    e2 = tf.add(e2, tf.reduce_sum(
        tf.multiply(tf.gather(lambda_phi, n),
                    dirichlet_expectation(lambda_pi))))
    h2 = tf.add(h2, -tf.reduce_sum(
        tf.multiply(tf.gather(lambda_phi, n), log_(
            tf.gather(lambda_phi, n)))))
    product = tf.convert_to_tensor([tf.reduce_sum(tf.matmul(
        tf.matmul(tf.reshape(tf.subtract(tf.gather(xn, n), lambda_m[k, :]),
                             [1, 2]), lambda_w[k, :, :]),
        tf.reshape(tf.transpose(tf.subtract(tf.gather(xn, n),
                                            lambda_m[k, :])),
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
                    tf.multiply(tf.gather(lambda_phi, n), aux))))
traces = tf.convert_to_tensor([tf.trace(
    tf.matmul(inv1, lambda_w[k, :, :])) for k in range(K)])
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
    tf.add(tf.multiply(lambda_nu, tf.log(tf.cast(2.,
                                                 dtype=tf.float64))), aux))
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
    tf.multiply(tf.div(nu_o, tf.cast(2., dtype=tf.float64)), tf.log(det2)),
    tf.add(tf.multiply(nu_o, tf.cast(tf.log(2.), dtype=tf.float64)),
           tf.add(tf.multiply(tf.cast(1. / 2., dtype=tf.float64),
                              tf.cast(tf.log(np.pi), dtype=tf.float64)),
                  tf.add(tf.lgamma(
                      tf.div(nu_o, tf.cast(2., dtype=tf.float64))),
                         tf.lgamma(tf.div(tf.subtract(
                             nu_o, tf.cast(1., dtype=tf.float64)),
                                          tf.cast(2.,
                                                  dtype=tf.float64)))))))
e5 = tf.reduce_sum(tf.add(-logB, tf.subtract(
    tf.multiply(tf.div(tf.subtract(nu_o, tf.cast(3., dtype=tf.float64)),
                       tf.cast(2., dtype=tf.float64)), logDeltak),
    tf.multiply(tf.div(lambda_nu, tf.cast(2., dtype=tf.float64)), traces))))
LB = e1 + e2 + e3 + e4 + e5 + h1 + h2 + h4 + h5

# Optimizer definition
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(INITIAL_LR, global_step,
                                           100, 0.96, staircase=True)
if args.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
elif args.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
elif args.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
elif args.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)

with tf.device('/gpu:0'):
    grads_and_vars = optimizer.compute_gradients(
        -LB, var_list=[lambda_pi_var, lambda_m,
                       lambda_beta_var, lambda_nu_var, lambda_w_var])
    train = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


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
                      lambda_nu, lambda_w, lambda_beta, xn, idx, K, D):
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
    for n in idx:
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

    # Inference
    init = tf.global_variables_initializer()
    sess.run(init)
    lbs = []
    aux_lbs = []
    n_iters = 0

    phi_out = sess.run(lambda_phi)
    pi_out = sess.run(lambda_pi)
    m_out = sess.run(lambda_m)
    nu_out = sess.run(lambda_nu)
    w_out = sess.run(lambda_w)
    beta_out = sess.run(lambda_beta)

    for i in range(args.maxIter * int(N / BATCH_SIZE)):

        # Sample xn
        idx = np.random.randint(N, size=BATCH_SIZE)

        # Update local variational parameter lambda_phi
        new_lambda_phi = update_lambda_phi(phi_out, pi_out, m_out, nu_out,
                                           w_out, beta_out, xn, idx, K, D)
        sess.run(lambda_phi.assign(new_lambda_phi))

        # ELBO computation and global variational parameter updates
        _, lb, pi_out, phi_out, m_out, beta_out, nu_out, w_out = sess.run(
            [train, LB, lambda_pi, lambda_phi, lambda_m,
             lambda_beta, lambda_nu, lambda_w], feed_dict={idx_tensor: idx})
        lb = lb * (N / BATCH_SIZE)
        aux_lbs.append(lb)
        if len(aux_lbs) == (N / BATCH_SIZE):
            lbs.append(np.mean(aux_lbs))
            n_iters += 1
            aux_lbs = []

        if VERBOSE:
            tf.logging.info('\n******* ITERATION {} *******'.format(n_iters))
            tf.logging.info('Time: {} seconds'.format(time()-init_time))
            tf.logging.info('lambda_pi: {}'.format(pi_out))
            tf.logging.info('lambda_phi: {}'.format(phi_out[0:9, :]))
            tf.logging.info('lambda_m: {}'.format(m_out))
            tf.logging.info('lambda_beta: {}'.format(beta_out))
            tf.logging.info('lambda_nu: {}'.format(nu_out))
            tf.logging.info('ELBO: {}'.format(lb))

            # Break condition
            improve = lb - lbs[n_iters - 1] if n_iters > 0 else lb
            if VERBOSE: tf.logging.info('Improve: {}'.format(improve))
            if n_iters > 0 and 0 <= improve < THRESHOLD: break

    zn = np.array([np.argmax(phi_out[n, :]) for n in range(N)])

    if VERBOSE:
        tf.logging.info('\n******* RESULTS *******')
        for k in range(K):
            tf.logging.info('Mu k{}: {}'.format(k, m_out[k, :]))
        final_time = time()
        exec_time = final_time - init_time
        tf.logging.info('Time: {} seconds'.format(exec_time))
        tf.logging.info('Iterations: {}'.format(n_iters))
        tf.logging.info('ELBOs: {}'.format(lbs[len(lbs)-10:len(lbs)]))

        if args.exportAssignments:
            with open('generated/sgavi_{}_assignments.csv'
                              .format(args.optimizer), 'wb') as output:
                writer = csv.writer(output, delimiter=';', quotechar='',
                                    escapechar='\\', quoting=csv.QUOTE_NONE)
                writer.writerow(['zn'])
                for i in range(len(zn)):
                    writer.writerow([zn[i]])

        if args.exportVariationalParameters:
            with open('generated/sgavi_{}_variational_parameters.pkl'
                              .format(args.optimizer), 'w') as output:
                pkl.dump({'lambda_pi': pi_out, 'lambda_m': m_out,
                          'lambda_beta': beta_out, 'lambda_nu': nu_out,
                          'lambda_w': w_out, 'K': K, 'D': D}, output)

        if args.exportELBOs:
            with open('generated/sgavi_{}_elbos.pkl'
                              .format(args.optimizer), 'w') as output:
                pkl.dump({'elbos': lbs, 'iter_time': exec_time/n_iters}, output)


if __name__ == '__main__': main()
