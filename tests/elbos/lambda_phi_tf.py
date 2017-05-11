# -*- coding: UTF-8 -*-

import pickle as pkl

import numpy as np
import tensorflow as tf
from scipy import random

K = 2

sess = tf.Session()


def dirichlet_expectation_k(alpha, k):
    """
    Dirichlet expectation computation
    \Psi(\alpha_{k}) - \Psi(\sum_{i=1}^{K}(\alpha_{i}))
    """
    return tf.subtract(tf.digamma(tf.add(alpha[k], np.finfo(np.float32).eps)),
                       tf.digamma(tf.reduce_sum(alpha)))


def softmax(x):
    """
    Softmax computation
    e^{x} / sum_{i=1}^{K}(e^x_{i})
    """
    return tf.div(tf.add(tf.exp(tf.subtract(x, tf.reduce_max(x))),
                         np.finfo(np.float32).eps),
                  tf.reduce_sum(
                      tf.add(tf.exp(tf.subtract(x, tf.reduce_max(x))),
                             np.finfo(np.float32).eps)))


def generate_random_positive_matrix(D):
    """
    Generate a random semidefinite positive matrix
    :param D: Dimension
    :return: DxD matrix
    """
    aux = random.rand(D, D)
    return np.dot(aux, aux.transpose())


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
            inv_lambda_w = tf.matrix_inverse(lambda_w[k, :, :])
            lambda_phi[n, k] = dirichlet_expectation_k(lambda_pi, k)
            lambda_phi[n, k] = tf.add(
                lambda_phi[n, k], tf.matmul(
                    lambda_m[k, :], tf.matmul(tf.multiply(
                        lambda_nu[k], inv_lambda_w), xn[n, :])))
            lambda_phi[n, k] = tf.subtract(
                lambda_phi[n, k], tf.trace(tf.matmul(
                    tf.multiply((1 / 2.) * lambda_nu[k], inv_lambda_w),
                    tf.multiply(tf.reshape(xn[n, :], [D, 1]),
                                tf.reshape(xn[n, :], [1,D])))))
            lambda_phi[n, k] = tf.subtract(
                lambda_phi[n, k], (D / 2.) * tf.div(1, lambda_beta[k]))
            lambda_phi[n, k] = tf.subtract(
                lambda_phi[n, k], (1. / 2.) * tf.matmul(
                    tf.matmul(lambda_nu[k] * tf.transpose(lambda_m[k, :]),
                              inv_lambda_w), lambda_m[k, :]))
            lambda_phi[n, k] = tf.add(lambda_phi[n, k],
                                             (D / 2.) * tf.log(2.))
            lambda_phi[n, k] = tf.add(
                lambda_phi[n, k],
                (1 / 2.) * tf.reduce_sum([tf.digamma(tf.add(tf.div(
                    lambda_nu[k], 2.), ((1 - i) / 2.))) for i in range(D)]))
            lambda_phi[n, k] = tf.subtract(
                lambda_phi[n, k],
                (1 / 2.) * tf.log(tf.matrix_determinant(lambda_w[k, :, :])))
        lambda_phi[n, :] = softmax(lambda_phi[n, :])
    return lambda_phi


def main():

    # Get data
    with open('../../data/synthetic/2D/k2/data_k2_100.pkl', 'r') as inputfile:
        data = pkl.load(inputfile)
        xn = data['xn']
    N, D = xn.shape

    np.random.seed(0)

    # Priors
    alpha_o = np.array([7.0, 3.0])

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

    init = tf.global_variables_initializer()
    sess.run(init)

    lambda_phi = update_lambda_phi(lambda_phi, lambda_pi, lambda_m,
                                   lambda_nu, lambda_w, lambda_beta,
                                   xn, N, K, D)

    print('lambda_phi: {}'.format(sess.run(lambda_phi)))


if __name__ == '__main__': main()
