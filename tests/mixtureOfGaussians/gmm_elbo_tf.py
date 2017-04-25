# -*- coding: UTF-8 -*-

import pickle as pkl

import numpy as np
import tensorflow as tf
from numpy.linalg import det, inv
from scipy import random
from scipy.special import psi

K = 2

sess = tf.Session()


def dirichlet_expectation(alpha, k):
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


def generate_random_positive_matrix(D):
    """
    Generate a random semidefinite positive matrix
    :param D: Dimension
    :return: DxD matrix
    """
    aux = random.rand(D, D)
    return np.dot(aux, aux.transpose())


def multilgamma(a, D, D_t):
    res = tf.multiply(tf.multiply(D_t, tf.multiply(tf.subtract(D_t, 1), tf.cast(0.25, dtype=tf.float64))), tf.log(tf.cast(np.pi, dtype=tf.float64)))
    res += tf.reduce_sum(tf.lgamma([tf.subtract(a, tf.div(tf.subtract(tf.cast(j, dtype=tf.float64), tf.cast(1., dtype=tf.float64)),
                                                          tf.cast(2., dtype=tf.float64))) for j in range(1, D+1)]), axis=0)
    return res


def update_lambda_pi(lambda_pi, lambda_phi, alpha_o):
    """
    Update lambda_pi
    alpha_o + sum_{i=1}^{N}(E_{q_{z}} I(z_{n}=i))
    """
    for k in range(K):
        lambda_pi[k] = alpha_o[k] + np.sum(lambda_phi[:, k])
    return lambda_pi


def update_lambda_beta(lambda_beta, beta_o, Nks):
    """
    Updtate lambda_beta
    beta_o + Nk
    """
    for k in range(K):
        lambda_beta[k] = beta_o + Nks[k]
    return lambda_beta


def update_lambda_nu(lambda_nu, nu_o, Nks):
    """
    Update lambda_nu
    nu_o + Nk
    """
    for k in range(K):
        lambda_nu[k] = nu_o + Nks[k]
    return lambda_nu


def update_lambda_m(lambda_m, lambda_phi, lambda_beta, m_o, beta_o, xn, N, D):
    """
    Update lambda_m
    (m_o.T * beta_o + sum_{n=1}^{N}(E_{q_{z}} I(z_{n}=i)x_{n})) / lambda_beta
    """
    for k in range(K):
        aux = np.array([0.] * D)
        for n in range(N):
            aux += lambda_phi[n, k] * xn[n, :]
        lambda_m[k, :] = ((m_o.T * beta_o + aux) / lambda_beta[k]).T
    return lambda_m


def update_lambda_w(lambda_w, lambda_phi, lambda_beta,
                    lambda_m, w_o, beta_o, m_o, xn, K, N, D):
    """
    Update lambda_w
    w_o + m_o * m_o.T + sum_{n=1}^{N}(E_{q_{z}} I(z_{n}=i)x_{n}x_{n}.T)
    - lambda_beta * lambda_m * lambda_m.T
    """

    for k in range(K):
        aux = np.array([[0.] * D] * D)
        for n in range(N):
            aux += lambda_phi[n, k] * np.outer(xn[n, :], xn[n, :].T)
        lambda_w[k, :, :] = w_o + beta_o * np.outer(m_o, m_o.T) + aux - \
                            lambda_beta[k] * np.outer(lambda_m[k, :],
                                                      lambda_m[k, :].T)
    return lambda_w


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
            lambda_phi[n, k] = dirichlet_expectation(lambda_pi, k)
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


def elbo(lambda_phi, lambda_pi, lambda_beta, lambda_nu,
         lambda_w, alpha_o, beta_o, nu_o, w_o,  N, D):
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

    # Get data
    with open('../../data/synthetic/2D/k2/data_k2_100.pkl', 'r') as inputfile:
        data = pkl.load(inputfile)
        xn = data['xn']
    N, D = xn.shape

    np.random.seed(0)

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

    lambda_pi = update_lambda_pi(lambda_pi, lambda_phi, alpha_o)
    Nks = np.sum(lambda_phi, axis=0)
    lambda_beta = update_lambda_beta(lambda_beta, beta_o, Nks)
    lambda_nu = update_lambda_nu(lambda_nu, nu_o, Nks)
    lambda_m = update_lambda_m(lambda_m, lambda_phi, lambda_beta, m_o,
                               beta_o, xn, N, D)
    lambda_w = update_lambda_w(lambda_w, lambda_phi, lambda_beta,
                               lambda_m, w_o, beta_o, m_o, xn, K, N, D)
    lambda_phi = update_lambda_phi(lambda_phi, lambda_pi, lambda_m,
                                   lambda_nu, lambda_w, lambda_beta,
                                   xn, N, K, D)

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

    lb = elbo(lambda_phi, lambda_pi, lambda_beta, lambda_nu,
              lambda_w, alpha_o, beta_o, nu_o, w_o, N, D)

    print('ELBO: {}'.format(sess.run(lb)))


if __name__ == '__main__': main()
