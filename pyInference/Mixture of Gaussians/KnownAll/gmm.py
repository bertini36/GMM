# -*- coding: UTF-8 -*-

import numpy as np
import pickle as pkl
import argparse
import matplotlib.pyplot as plt
from scipy.special import psi
from utils import dirichlet_expectation, exp_normalize


def initialize(xn, K, alpha, m_o, beta_o, nu_o, Delta_o):
    N, D = xn.shape
    phi = np.random.dirichlet(alpha, N)
    lambda_pi = alpha + np.sum(phi, axis=0)
    Nk = np.sum(phi, axis=0)
    xk_ = np.tile(1. / Nk, (2, 1)).T * np.dot(phi.T, xn)
    lambda_mu_beta = beta_o + Nk
    lambda_delta_nu = nu_o + Nk
    lambda_mu_m = np.tile(1. / lambda_mu_beta, (2, 1)).T * (
        m_o * beta_o + np.dot(phi.T, xn))
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Inference in the gaussian mixture data with unknown means')
    parser.add_argument('-maxIter', metavar='maxIter', type=int, default=100)
    parser.add_argument('-K', metavar='K', type=int, default=2)
    parser.add_argument('-filename', metavar='filename', type=str,
                        default="data_means.pkl")
    parser.add_argument('-alpha', metavar='alpha', nargs='+', type=float,
                        default=[1.] * 2)
    parser.add_argument('-m_o', metavar='m_o', nargs='+', type=float,
                        default=[0., 0.])
    parser.add_argument('-beta_o', metavar='beta_o', type=float, default=0.01)
    parser.add_argument('-Delta_o', metavar='Delta_o', nargs='+', type=float,
                        default=[1., 0., 0., 1.])
    parser.add_argument('-nu_o', metavar='nu_o', type=float, default=2.)
    args = parser.parse_args()

    with open('data/' + args.filename, 'r') as inputfile:
        data = pkl.load(inputfile)
    xn = data['xn']

    plt.scatter(xn[:, 0], xn[:, 1], c=(1. * data['zn']) / max(data['zn']))
    plt.show()

    N, D = xn.shape
    K = args.K
    alpha = args.alpha
    m_o = np.array(args.m_o)
    beta_o = args.beta_o
    Delta_o = np.array([args.Delta_o[0:D], args.Delta_o[D:2 * D]])
    nu_o = args.nu_o

    lambda_pi, phi, lambda_mu_m, lambda_mu_beta, lambda_delta_nu, lambda_delta_W = initialize(
        xn, K, alpha, m_o, beta_o, nu_o, Delta_o)

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
