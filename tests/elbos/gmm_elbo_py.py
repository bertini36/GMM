# -*- coding: UTF-8 -*-

import math
import pickle as pkl

import numpy as np
from numpy.linalg import det, inv
from scipy import random
from scipy.special import gammaln, multigammaln, psi

K = 2


def dirichlet_expectation(alpha):
    """
    Dirichlet expectation computation
    \Psi(\alpha) - \Psi(\sum_{i=1}^{K}(\alpha_{i}))
    """
    if len(alpha.shape) == 1:
        return psi(alpha + np.finfo(np.float32).eps) - psi(np.sum(alpha))
    return psi(alpha + np.finfo(np.float32).eps) \
           - psi(np.sum(alpha, 1))[:, np.newaxis]


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


def generate_random_positive_matrix(D):
    """
    Generate a random semidefinite positive matrix
    :param D: Dimension
    :return: DxD matrix
    """
    aux = random.rand(D, D)
    return np.dot(aux, aux.transpose())


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


def log_beta_function(x):
    """
    Log beta function
    ln(\gamma(x)) - ln(\gamma(\sum_{i=1}^{N}(x_{i}))
    """
    return np.sum(gammaln(x + np.finfo(np.float32).eps)) - gammaln(
        np.sum(x + np.finfo(np.float32).eps))


def log_(x):
    return np.log(x + np.finfo(np.float32).eps)


def elbo(lambda_phi, lambda_pi, lambda_beta, lambda_nu,
         lambda_w, alpha_o, beta_o, nu_o, w_o, N, D):
    """
    ELBO computation
    """
    lb = gammaln(np.sum(alpha_o)) - np.sum(gammaln(alpha_o)) \
         - gammaln(np.sum(lambda_pi)) + np.sum(gammaln(lambda_pi))
    lb -= N * D / 2. * np.log(2. * np.pi)
    for k in range(K):
        lb += -(nu_o[0] * D * np.log(2.)) / 2. + (lambda_nu[k] * D * np.log(
            2.)) / 2.
        lb += -multigammaln(nu_o[0] / 2., D) + multigammaln(lambda_nu[k] / 2.,
                                                            D)
        lb += (D / 2.) * np.log(np.absolute(beta_o[0])) - (D / 2.) * np.log(
            np.absolute(lambda_beta[k]))
        lb += (nu_o[0] / 2.) * np.log(det(w_o)) - (lambda_nu[k] / 2.) * np.log(
            det(lambda_w[k, :, :]))
        lb -= np.dot(np.log(lambda_phi[:, k]).T, lambda_phi[:, k])
    return lb


def elbo2(xn, alpha_o, lambda_pi, lambda_phi, m_o, lambda_m, beta_o,
          lambda_beta, nu_o, lambda_nu, w_o, lambda_w, N, K):
    """
    ELBO computation
    """
    e3 = e2 = h2 = 0
    e1 = - log_beta_function(alpha_o) \
         + np.dot((alpha_o - np.ones(K)), dirichlet_expectation(lambda_pi))
    h1 = log_beta_function(lambda_pi) \
         - np.dot((lambda_pi - np.ones(K)), dirichlet_expectation(lambda_pi))
    logdet = np.log(np.array([det(lambda_w[k, :, :]) for k in xrange(K)]))
    logDeltak = psi(lambda_nu / 2.) \
                + psi((lambda_nu - 1.) / 2.) + 2. * np.log(2.) + logdet
    for n in range(N):
        e2 += np.dot(lambda_phi[n, :], dirichlet_expectation(lambda_pi))
        h2 += -np.dot(lambda_phi[n, :], log_(lambda_phi[n, :]))
        product = np.array([np.dot(np.dot(
            xn[n, :] - lambda_m[k, :], lambda_w[k, :, :]),
            (xn[n, :] - lambda_m[k, :]).T) for k in xrange(K)])
        e3 += 1. / 2 * np.dot(lambda_phi[n, :],
                              (logDeltak - 2. * np.log(2 * math.pi) -
                               lambda_nu * product - 2. / lambda_beta).T)
    product = np.array([np.dot(np.dot(lambda_m[k, :] - m_o, lambda_w[k, :, :]),
                               (lambda_m[k, :] - m_o).T) for k in xrange(K)])
    traces = np.array([np.trace(np.dot(inv(w_o),
                                       lambda_w[k, :, :])) for k in xrange(K)])
    h4 = np.sum(
        (1. + np.log(2. * math.pi) - 1. / 2 * (np.log(lambda_beta) + logdet)))
    logB = lambda_nu / 2. * logdet + lambda_nu * np.log(2.) + 1. / 2 * np.log(
        math.pi) \
           + gammaln(lambda_nu / 2.) + gammaln((lambda_nu - 1) / 2.)
    h5 = np.sum((logB - (lambda_nu - 3.) / 2. * logDeltak + lambda_nu))
    e4 = np.sum((1. / 2 * (np.log(beta_o) + logDeltak - 2 * np.log(2. * math.pi)
                           - beta_o * lambda_nu * product - 2. * beta_o / lambda_beta)))
    logB = nu_o / 2. * np.log(np.linalg.det(w_o)) + nu_o * np.log(2.) \
           + 1. / 2 * np.log(math.pi) + gammaln(nu_o / 2.) + gammaln(
        (nu_o - 1) / 2.)
    e5 = np.sum(
        (-logB + (nu_o - 3.) / 2. * logDeltak - lambda_nu / 2. * traces))
    return e1 + e2 + e3 + e4 + e5 + h1 + h2 + h4 + h5


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

    # ELBO computation
    lb = elbo(lambda_phi, lambda_pi, lambda_beta, lambda_nu,
              lambda_w, alpha_o, beta_o, nu_o, w_o, N, D)
    print('ELBO: {}'.format(lb))

    # ELBO 2 computation
    lb2 = elbo2(xn, alpha_o, lambda_pi, lambda_phi, m_o, lambda_m, beta_o,
                lambda_beta, nu_o, lambda_nu, w_o, lambda_w, N, K)
    print('ELBO2: {}'.format(lb2))


if __name__ == '__main__': main()
