# -*- coding: UTF-8 -*-

import math
import numpy as np
import pickle as pkl
from scipy.special import psi, gammaln

np.random.seed(7)


def initialize(xn, alpha, m_o, beta_o):
    N, D = xn.shape
    phi = np.random.dirichlet(alpha, N)
    lambda_pi = alpha + np.sum(phi, axis=0)
    lambda_mu_beta = beta_o + np.sum(phi, axis=0)
    lambda_mu_m = np.tile(1. / lambda_mu_beta, (2, 1)).T * (
    beta_o * m_o + np.dot(phi.T, xn))
    return lambda_pi, phi, lambda_mu_m, lambda_mu_beta


def ELBO(xn, K, alpha, m_o, beta_o, Delta_o, lambda_pi, lambda_mu_m,
         lambda_mu_beta, phi):
    ELBO = log_beta_function(lambda_pi)
    ELBO -= log_beta_function(alpha)
    ELBO += np.dot(alpha - lambda_pi, dirichlet_expectation(lambda_pi))
    ELBO += K / 2. * np.log(np.linalg.det(beta_o * Delta_o))
    ELBO += K * D / 2.
    for k in xrange(K):
        a1 = lambda_mu_m[k, :] - m_o
        a2 = np.dot(Delta_o, (lambda_mu_m[k, :] - m_o).T)
        a3 = beta_o / 2. * np.dot(a1, a2)
        a4 = D * beta_o / (2. * lambda_mu_beta[k])
        a5 = 1 / 2. * np.log(np.linalg.det(lambda_mu_beta[k] * Delta_o))
        a6 = a3 + a4 + a5
        ELBO -= a6
        b1 = phi[:, k].T
        b2 = dirichlet_expectation(lambda_pi)[k]
        b3 = np.log(phi[:, k])
        b4 = 1 / 2. * np.log(np.linalg.det(Delta_o) / (2. * math.pi))
        b5 = xn - lambda_mu_m[k, :]
        b6 = np.dot(Delta_o, (xn - lambda_mu_m[k, :]).T)
        b7 = 1 / 2. * np.diagonal(np.dot(b5, b6))
        b8 = D / (2. * lambda_mu_beta[k])
        ELBO += np.dot(b1, b2 - b3 + b4 - b7 - b8)
    return ELBO


def dirichlet_expectation(alpha):
    if len(alpha.shape) == 1:
        return psi(alpha + np.finfo(np.float32).eps) - psi(np.sum(alpha))
    return psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis]


def exp_normalize(aux):
    return (np.exp(aux - np.max(aux)) + np.finfo(np.float32).eps) / (
    np.sum(np.exp(aux - np.max(aux)) + np.finfo(np.float32).eps))


def log_beta_function(x):
    return np.sum(gammaln(x + np.finfo(np.float32).eps)) - gammaln(
        np.sum(x + np.finfo(np.float32).eps))


with open('../../data/data_k2_100.pkl', 'r') as inputfile:
    data = pkl.load(inputfile)
    xn = data['xn']

N, D = xn.shape
K = 2
alpha = [1.0, 1.0]
m_o = np.array([0.0, 0.0])
beta_o = 0.01
Delta_o = np.array([[1.0, 0.0], [0.0, 1.0]])

lambda_pi, phi, lambda_mu_m, lambda_mu_beta = initialize(xn, alpha, m_o, beta_o)

elbo = ELBO(xn, K, alpha, m_o, beta_o, Delta_o,
            lambda_pi, lambda_mu_m, lambda_mu_beta, phi)
print('ELBO={}'.format(elbo))
