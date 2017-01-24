# -*- coding: UTF-8 -*-

"""
Coordinate Ascent Variational Inference
process to approximate an Univariate Gaussian
"""

import math
import argparse
import numpy as np
from time import time
from scipy.special import psi, gammaln

parser = argparse.ArgumentParser(description='CAVI in Univariate Gaussian')
parser.add_argument('-maxIter', metavar='maxIter', type=int, default=10000000)
parser.add_argument('-nElements', metavar='nElements', type=int, default=100)
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
args = parser.parse_args()

N = args.nElements
MAX_ITERS = args.maxIter
DATA_MEAN = 7
THRESHOLD = 1e-6


def elbo(xn, m, beta, a, b, m_mu, beta_mu, a_gamma, b_gamma):
    lb = 0
    lb += 1. / 2 * np.log(beta / beta_mu) + 1. / 2 * (
        m_mu ** 2 + 1. / beta_mu) * (beta_mu - beta) - m_mu * (
        beta_mu * m_mu - beta * m) + 1. / 2 * (
        beta_mu * m_mu ** 2 - beta * m ** 2)
    lb += a * np.log(b) - a_gamma * np.log(b_gamma) + gammaln(
        a_gamma) - gammaln(a) + (psi(a_gamma) - np.log(b_gamma)) * (
        a - a_gamma) + a_gamma / b_gamma * (b_gamma - b)
    lb += N / 2. * (psi(a_gamma) - np.log(b_gamma)) - N / 2. * np.log(
        2 * math.pi) - 1. / 2 * a_gamma / b_gamma * sum(
        xn ** 2) + a_gamma / b_gamma * sum(
        xn) * m_mu - N / 2. * a_gamma / b_gamma * (m_mu ** 2 + 1. / beta_mu)
    return lb


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


def main():
    # Data generation
    xn = np.random.normal(DATA_MEAN, 1, N)

    if args.timing:
        init_time = time()

    # Model hyperparameters
    m = 0.
    beta = 0.0001
    a = 0.001
    b = 0.001

    # Variational parameters
    a_gamma = np.random.gamma(1, 1, 1)[0]
    b_gamma = np.random.gamma(1, 1, 1)[0]
    # m_mu = np.random.normal(0., beta ** (-1.), 1)[0]
    # beta_mu = np.random.gamma(a_gamma, b_gamma, 1)[0]

    lbs = []
    for i in xrange(MAX_ITERS):

        # Parameter updates
        m_mu = (beta * m + a_gamma / b_gamma * sum(xn)) / (
            beta + N * a_gamma / b_gamma)
        beta_mu = beta + N * a_gamma / b_gamma
        a_gamma = a + N / 2.
        b_gamma = b + 1. / 2 * sum(xn ** 2) - m_mu * sum(xn) + N / 2. * (
            m_mu ** 2 + 1. / beta_mu)

        # ELBO computation
        lb = elbo(xn, m, beta, a, b, m_mu, beta_mu, a_gamma, b_gamma)
        if args.debug:
            print('Iter {}: Mu={} Precision={} ELBO={}'
                  .format(i, m_mu, a_gamma / b_gamma, lb))

        # Break condition
        if i > 0:
            if abs(lb - lbs[i - 1]) < THRESHOLD:
                if args.getNIter:
                    n_iters = i + 1
                break
        lbs.append(lb)

    if args.timing:
        final_time = time()
        exec_time = final_time - init_time
        print('Time: {} seconds'.format(exec_time))

    if args.getNIter:
        print('Iterations: {}'.format(n_iters))

    if args.getELBO:
        print('ELBOs: {}'.format(lbs))


if __name__ == '__main__': main()
