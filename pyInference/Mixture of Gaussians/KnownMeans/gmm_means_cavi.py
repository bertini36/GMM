# -*- coding: UTF-8 -*-

"""
Coordinate Ascent Variational Inference
process to approximate Mixture of Gaussians
"""

import math
import argparse
import numpy as np
import pickle as pkl
from time import time
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.special import psi, gammaln

parser = argparse.ArgumentParser(description='CAVI in Mixture of Gaussians')
parser.add_argument('-maxIter', metavar='maxIter', type=int, default=10000000)
parser.add_argument('-dataset', metavar='dataset',
                    type=str, default='../../../data/data_k2_100.pkl')
parser.add_argument('-k', metavar='k', type=int, default=2)
parser.add_argument('--timing', dest='timing', action='store_true')
parser.add_argument('--no-timing', dest='timing', action='store_false')
parser.set_defaults(timing=False)
parser.add_argument('--getNIter', dest='getNIter', action='store_true')
parser.add_argument('--no-getNIter', dest='getNIter', action='store_false')
parser.set_defaults(getNIter=False)
parser.add_argument('--getELBO', dest='getELBO', action='store_true')
parser.add_argument('--no-getELBO', dest='getELBO', action='store_false')
parser.set_defaults(getELBO=False)
parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--no-debug', dest='debug', action='store_false')
parser.set_defaults(debug=True)
parser.add_argument('--plot', dest='plot', action='store_true')
parser.add_argument('--no-plot', dest='plot', action='store_false')
parser.set_defaults(plot=True)
args = parser.parse_args()

if args.timing:
    init_time = time()

MAX_ITERS = args.maxIter
K = args.k
THRESHOLD = 1e-6


def initialize(xn, alpha, m_o, beta_o):
    N, D = xn.shape
    phi = np.random.dirichlet(alpha, N)
    lambda_pi = alpha + np.sum(phi, axis=0)
    lambda_mu_beta = beta_o + np.sum(phi, axis=0)
    lambda_mu_m = np.tile(1. / lambda_mu_beta, (2, 1)).T * (
    beta_o * m_o + np.dot(phi.T, xn))
    return lambda_pi, phi, lambda_mu_m, lambda_mu_beta


def ELBO(xn, K, alpha, m_o, beta_o, Delta_o,
         lambda_pi, lambda_mu_m, lambda_mu_beta, phi):
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


if __name__ == '__main__':

    # Get data
    with open('{}'.format(args.dataset), 'r') as inputfile:
        data = pkl.load(inputfile)
        xn = data['xn']

    if args.plot:
        plt.scatter(xn[:, 0], xn[:, 1], c=(1. * data['zn']) / max(data['zn']),
                    cmap=cm.gist_rainbow, s=5)
        plt.show()

    N, D = xn.shape

    # Initializations
    N, D = xn.shape
    K = 2
    alpha = [1.0, 1.0]
    m_o = np.array([0.0, 0.0])
    beta_o = 0.01
    Delta_o = np.array([[1.0, 0.0], [0.0, 1.0]])
    lambda_pi, phi, lambda_mu_m, lambda_mu_beta = initialize(xn, alpha,
                                                             m_o, beta_o)
    elbos = []
    for i in xrange(MAX_ITERS):

        # Parameter updates
        lambda_pi = alpha + np.sum(phi, axis=0)
        c1 = dirichlet_expectation(lambda_pi)
        for n in xrange(N):
            aux = np.copy(c1)
            for k in xrange(K):
                c2 = xn[n, :] - lambda_mu_m[k, :]
                c3 = np.dot(Delta_o, (xn[n, :] - lambda_mu_m[k, :]).T)
                c4 = -1. / 2 * np.dot(c2, c3)
                c5 = D / (2. * lambda_mu_beta[k])
                aux[k] += c4 - c5
            phi[n, :] = exp_normalize(aux)
        lambda_mu_beta = beta_o + np.sum(phi, axis=0)
        d1 = np.tile(1. / lambda_mu_beta, (K, 1)).T
        d2 = m_o * beta_o + np.dot(phi.T, xn)
        lambda_mu_m = d1 * d2

        # Compute ELBO
        lb = ELBO(xn, K, alpha, m_o, beta_o, Delta_o,
                  lambda_pi, lambda_mu_m, lambda_mu_beta, phi)
        if args.debug:
            print('Iter {}: Mus={} Precision={} Pi={} ELBO={}'
                  .format(i, lambda_mu_m, lambda_mu_beta, lambda_pi, lb))

        # Break condition
        if i > 0:
            if abs(lb - old_lb) < THRESHOLD:
                if args.getNIter:
                    n_iters = i + 1
                break
        old_lb = lb

    if args.plot:
        plt.scatter(xn[:, 0], xn[:, 1], c=np.array(
            1 * [np.random.choice(K, 1, p=phi[n, :])[0] for n in xrange(N)]),
                    cmap=cm.gist_rainbow, s=5)
        plt.show()

    if args.timing:
        final_time = time()
        exec_time = final_time - init_time
        print('Time: {} seconds'.format(exec_time))

    if args.getNIter:
        print('Iterations: {}'.format(n_iters))

    if args.getELBO:
        print('ELBO: {}'.format(lb))
