# -*- coding: UTF-8 -*-

"""
Coordinate Ascent Variational Inference process
to approximate an Univariate Gaussian (UGM)
"""

from __future__ import absolute_import

import argparse
import math
from time import time

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammaln, psi

"""
Parameters:
    * maxIter: Max number of iterations
    * nElements: Number of data points to generate
    * verbose: Printing time, intermediate variational parameters, plots, ...
    
Execution:
    python ugm_cavi.py -nElements 1000 -verbose 
"""

parser = argparse.ArgumentParser(description='CAVI in univariate gaussian')
parser.add_argument('-maxIter', metavar='maxIter', type=int, default=300)
parser.add_argument('-nElements', metavar='nElements', type=int, default=1000)
parser.add_argument('-verbose', dest='verbose', action='store_true')
parser.set_defaults(verbose=False)
args = parser.parse_args()

N = args.nElements
VERBOSE = args.verbose
DATA_MEAN = 7
THRESHOLD = 1e-6


def update_lambda_m(lambda_a, lambda_b, m_o, beta_o, xn):
    """
    Update lambda_m
    """
    return (beta_o * m_o + lambda_a / lambda_b * sum(xn)) / \
           (beta_o + N * lambda_a / lambda_b)


def update_lambda_beta(lambda_a, lambda_b, beta_o):
    """
    Update lambda_beta
    """
    return beta_o + N * lambda_a / lambda_b


def update_lambda_a(a_o, N):
    """
    Update lambda_a
    """
    return a_o + N / 2.


def update_lambda_b(lambda_m, lambda_beta, b_o, xn):
    """
    Update lambda_b
    """
    return b_o + 1. / 2 * sum(xn ** 2) - lambda_m * sum(xn) + \
           N / 2. * (lambda_m ** 2 + 1. / lambda_beta)


def elbo(xn, m_o, beta_o, a_o, b_o, lambda_m, lambda_beta, lambda_a, lambda_b):
    """
    ELBO computation
    """
    lb = 0
    lb += 1. / 2 * np.log(beta_o / lambda_beta) + 1. / 2 * (
        lambda_m ** 2 + 1. / lambda_beta) * (lambda_beta - beta_o) \
          - lambda_m * (lambda_beta * lambda_m - beta_o * m_o) + 1. / 2 * (
        lambda_beta * lambda_m ** 2 - beta_o * m_o ** 2)
    lb += a_o * np.log(b_o) - lambda_a * np.log(lambda_b) + gammaln(
        lambda_a) - gammaln(a_o) + (psi(lambda_a) - np.log(lambda_b)) * (
        a_o - lambda_a) + lambda_a / lambda_b * (lambda_b - b_o)
    lb += N / 2. * (psi(lambda_a) - np.log(lambda_b)) - N / 2. * np.log(
        2 * math.pi) - 1. / 2 * lambda_a / lambda_b * sum(
        xn ** 2) + lambda_a / lambda_b * sum(xn) * lambda_m \
          - N / 2. * lambda_a / lambda_b * (lambda_m ** 2 + 1. / lambda_beta)
    return lb


def main():

    # Data generation
    xn = np.random.normal(DATA_MEAN, 1, N)

    if VERBOSE:
        plt.plot(xn, 'ro', markersize=3)
        plt.title('Simulated dataset')
        plt.show()
        init_time = time()

    # Priors
    m_o = 0.
    beta_o = 0.0001
    a_o = 0.001
    b_o = 0.001

    # Variational parameters intialization
    lambda_a = np.random.gamma(1, 1, 1)[0]
    lambda_b = np.random.gamma(1, 1, 1)[0]

    lbs = []
    n_iters = 0
    for _ in range(args.maxIter):

        # Variational parameter updates
        lambda_m = update_lambda_m(lambda_a, lambda_b, m_o, beta_o, xn)
        lambda_beta = update_lambda_beta(lambda_a, lambda_b, beta_o)
        lambda_a = update_lambda_a(a_o, N)
        lambda_b = update_lambda_b(lambda_m, lambda_beta, b_o, xn)

        # ELBO computation
        lb = elbo(xn, m_o, beta_o, a_o, b_o,
                  lambda_m, lambda_beta, lambda_a, lambda_b)
        lbs.append(lb)

        if VERBOSE:
            print('\n******* ITERATION {} *******'.format(n_iters))
            print('lambda_m: {}'.format(lambda_m))
            print('lambda_beta: {}'.format(lambda_beta))
            print('lambda_a: {}'.format(lambda_a))
            print('lambda_b: {}'.format(lambda_b))
            print('ELBO: {}'.format(lb))

        # Break condition
        improve = lb - lbs[n_iters - 1]
        if VERBOSE: print('Improve: {}'.format(improve))
        if (n_iters == (args.maxIter - 1)) \
                or (n_iters > 0 and 0 < improve < THRESHOLD):
            break

        n_iters += 1

    if VERBOSE:
        plt.scatter(xn, mlab.normpdf(xn, lambda_m, lambda_a / lambda_b), s=5)
        plt.title('Result')
        plt.show()
        final_time = time()
        exec_time = final_time - init_time
        print('Time: {} seconds'.format(exec_time))
        print('Iterations: {}'.format(n_iters))
        print('ELBOs: {}'.format(lbs))


if __name__ == '__main__': main()
