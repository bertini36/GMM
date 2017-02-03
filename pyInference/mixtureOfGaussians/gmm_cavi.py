# -*- coding: UTF-8 -*-

"""
Coordinate Ascent Variational Inference
process to approximate a mixture of gaussians
[DOING]
"""

import argparse
import pickle as pkl
from time import time

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammaln, psi

parser = argparse.ArgumentParser(description='CAVI in mixture of gaussians')
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

MAX_ITERS = args.maxIter
K = args.k
THRESHOLD = 1e-6


def dirichlet_expectation(alpha):
    if len(alpha.shape) == 1:
        return psi(alpha + np.finfo(np.float32).eps) - psi(np.sum(alpha))
    return psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis]


def exp_normalize(aux):
    return (np.exp(aux-np.max(aux))+ np.finfo(np.float32).eps) / \
           (np.sum(np.exp(aux-np.max(aux))+np.finfo(np.float32).eps))


def log_beta_function(x):
    return np.sum(gammaln(x + np.finfo(np.float32).eps)) - \
           gammaln(np.sum(x + np.finfo(np.float32).eps))


def elbo():
    pass


def initialize(xn, K, alpha, m_o, beta_o, nu_o, Delta_o):
    N, D = xn.shape
    phi = np.random.dirichlet(alpha, N)
    lambda_pi = alpha + np.sum(phi, axis=0)
    Nk = np.sum(phi, axis=0)
    xk_ = np.tile(1. / Nk, (2, 1)).T * np.dot(phi.T, xn)
    lambda_mu_beta = beta_o + Nk
    lambda_delta_nu = nu_o + Nk
    lambda_mu_m = np.tile(1. / lambda_mu_beta, (2, 1)).T *\
                  (m_o * beta_o + np.dot(phi.T, xn))
    lambda_delta_W = np.zeros((K, D, D))
    for k in xrange(K):
        S = 1. / Nk[k] * np.dot((xn - xk_[k, :]).T,
                                np.dot(np.diag(phi[:, k]), (xn - xk_[k, :])))
        lambda_delta_W[k, :, :] = np.linalg.inv(
            np.linalg.inv(Delta_o) + Nk[k] * S \
            + beta_o * Nk[k] / (beta_o + Nk[k]) * np.dot(
                np.tile((xk_[k, :] - m_o), (1, 1)).T,
                np.tile(xk_[k, :] - m_o, (1, 1))))
    return lambda_pi, phi, lambda_mu_m, lambda_mu_beta, \
           lambda_delta_nu, lambda_delta_W


def main():
    # Get data
    with open('{}'.format(args.dataset), 'r') as inputfile:
        data = pkl.load(inputfile)
        xn = data['xn']
    N, D = xn.shape

    if args.timing:
        init_time = time()

    if args.plot:
        plt.scatter(xn[:, 0], xn[:, 1], c=(1. * data['zn']) / max(data['zn']),
                    cmap=cm.gist_rainbow, s=5)
        plt.show()

    alpha = args.alpha
    m_o = np.array(args.m_o)
    beta_o = args.beta_o
    Delta_o = np.array([args.Delta_o[0:D], args.Delta_o[D:2 * D]])
    nu_o = args.nu_o

    lambda_pi, phi, lambda_mu_m, lambda_mu_beta, lambda_delta_nu,\
        lambda_delta_W = initialize(xn, K, alpha, m_o, beta_o, nu_o, Delta_o)

    for it in xrange(args.maxIter):
        print it
        lambda_pi = alpha + np.sum(phi, axis=0)
        Elogpi = dirichlet_expectation(lambda_pi)
        for n in xrange(N):
            aux = np.copy(Elogpi)
            for k in xrange(K):
                aux[k] += 1. / 2 * (psi(lambda_delta_nu[k] / 2.) + psi(
                    (lambda_delta_nu[k] - 1.) / 2.)
                                    + np.log(
                    np.linalg.det(lambda_delta_W[k, :, :])) - 2. /
                                    lambda_mu_beta[k]
                                    - lambda_delta_nu[k] * (
                                        np.dot((xn[n, :] - lambda_mu_m[k, :]).T,
                                               np.dot(lambda_delta_W[k, :, :], (
                                                   xn[n, :] - lambda_mu_m[k,
                                                              :])))))
            phi[n, :] = exp_normalize(aux)

        Nk = np.sum(phi, axis=0)
        lambda_mu_beta = beta_o + Nk
        lambda_mu_m = np.tile(1. / lambda_mu_beta, (2, 1)).T * (
            m_o * beta_o + np.dot(phi.T, xn))

        lambda_delta_nu = nu_o + Nk
        xk_ = np.tile(1. / Nk, (2, 1)).T * np.dot(phi.T, xn)
        for k in xrange(K):
            S = 1. / Nk[k] * np.dot((xn - xk_[k, :]).T,
                                    np.dot(np.diag(phi[:, k]),
                                           (xn - xk_[k, :])))
            lambda_delta_W[k, :, :] = np.linalg.inv(
                np.linalg.inv(Delta_o) + Nk[k] * S
                + beta_o * Nk[k] / (beta_o + Nk[k]) * np.dot(
                    np.tile((xk_[k, :] - m_o), (1, 1)).T,
                    np.tile(xk_[k, :] - m_o, (1, 1))))

    print(lambda_mu_m)

    plt.scatter(xn[:, 0], xn[:, 1], c=np.array(
        1 * [np.random.choice(K, 1, p=phi[n, :])[0] for n in xrange(N)]))
    plt.show()


if __name__ == '__main__': main()
