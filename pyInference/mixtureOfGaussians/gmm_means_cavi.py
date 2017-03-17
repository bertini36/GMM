# -*- coding: UTF-8 -*-

"""
Coordinate Ascent Variational Inference process to
approximate a Mixture of Gaussians (GMM) with known precisions
"""

import argparse
import math
import pickle as pkl
from time import time

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammaln, psi

from sklearn.cluster import KMeans
from viz import create_cov_ellipse

"""
Parameters:
    * maxIter: Max number of iterations
    * dataset: Dataset path
    * k: Number of clusters
    * verbose: Printing time, intermediate variational parameters, plots, ...
    * randomInit: Init assignations randomly or with Kmeans
"""

parser = argparse.ArgumentParser(description='CAVI in mixture og gaussians')
parser.add_argument('-maxIter', metavar='maxIter', type=int, default=10000000)
parser.add_argument('-dataset', metavar='dataset',
                    type=str, default='../../data/k8/data_k8_1000.pkl')
parser.add_argument('-k', metavar='k', type=int, default=8)
parser.add_argument('--verbose', dest='verbose', action='store_true')
parser.add_argument('--no-verbose', dest='verbose', action='store_false')
parser.set_defaults(verbose=True)
parser.add_argument('--randomInit', dest='randomInit', action='store_true')
parser.add_argument('--no-randomInit', dest='randomInit', action='store_false')
parser.set_defaults(randomInit=False)
args = parser.parse_args()

MAX_ITERS = args.maxIter
K = args.k
VERBOSE = args.verbose
RANDOM_INIT = args.randomInit
THRESHOLD = 1e-6
PATH_IMAGE = 'img/gmm_means'


def dirichlet_expectation(alpha):
    """
    Dirichlet expectation computation
    \Psi(\alpha_{k}) - \Psi(\sum_{i=1}^{K}(\alpha_{i}))
    """
    return psi(alpha + np.finfo(np.float32).eps) - psi(np.sum(alpha))


def softmax(x):
    """
    Softmax computation
    e^{x} / sum_{i=1}^{K}(e^x_{i})
    """
    e_x = np.exp(x - np.max(x))
    return (e_x + np.finfo(np.float32).eps) / (
        e_x.sum(axis=0) + np.finfo(np.float32).eps)


def log_beta_function(x):
    """
    Log beta function
    ln(\gamma(x)) - ln(\gamma(\sum_{i=1}^{N}(x_{i}))
    """
    return np.sum(gammaln(x + np.finfo(np.float32).eps)) - gammaln(
        np.sum(x + np.finfo(np.float32).eps))


def update_lambda_pi(lambda_phi, alpha_o):
    """
    Update lambda_pi
    """
    return alpha_o + np.sum(lambda_phi, axis=0)


def update_lambda_phi(lambda_pi, lambda_m, lambda_beta,
                      lambda_phi, delta_o, xn, N, D):
    """
    Update lambda_phi
    """
    c1 = dirichlet_expectation(lambda_pi)
    for n in xrange(N):
        aux = np.copy(c1)
        for k in xrange(K):
            c2 = xn[n, :] - lambda_m[k, :]
            c3 = np.dot(delta_o, (xn[n, :] - lambda_m[k, :]).T)
            c4 = -1. / 2 * np.dot(c2, c3)
            c5 = D / (2. * lambda_beta[k])
            aux[k] += c4 - c5
        lambda_phi[n, :] = softmax(aux)
    return lambda_phi


def update_lambda_beta(lambda_phi, beta_o):
    """
    Update lambda_beta
    """
    return beta_o + np.sum(lambda_phi, axis=0)


def update_lambda_m(lambda_beta, lambda_phi, m_o, beta_o, xn, D):
    """
    Update lambda_m
    """
    d1 = np.tile(1. / lambda_beta, (D, 1)).T
    d2 = m_o * beta_o + np.dot(lambda_phi.T, xn)
    return d1 * d2


def elbo(xn, D, K, alpha, m_o, beta_o, delta_o,
         lambda_pi, lambda_m, lambda_beta, phi):
    """
    ELBO computation
    """
    lb = log_beta_function(lambda_pi)
    lb -= log_beta_function(alpha)
    lb += np.dot(alpha - lambda_pi, dirichlet_expectation(lambda_pi))
    lb += K / 2. * np.log(np.linalg.det(beta_o * delta_o))
    lb += K * D / 2.
    for k in xrange(K):
        a1 = lambda_m[k, :] - m_o
        a2 = np.dot(delta_o, (lambda_m[k, :] - m_o).T)
        a3 = beta_o / 2. * np.dot(a1, a2)
        a4 = D * beta_o / (2. * lambda_beta[k])
        a5 = 1 / 2. * np.log(np.linalg.det(lambda_beta[k] * delta_o))
        a6 = a3 + a4 + a5
        lb -= a6
        b1 = phi[:, k].T
        b2 = dirichlet_expectation(lambda_pi)[k]
        b3 = np.log(phi[:, k])
        b4 = 1 / 2. * np.log(np.linalg.det(delta_o) / (2. * math.pi))
        b5 = xn - lambda_m[k, :]
        b6 = np.dot(delta_o, (xn - lambda_m[k, :]).T)
        b7 = 1 / 2. * np.diagonal(np.dot(b5, b6))
        b8 = D / (2. * lambda_beta[k])
        lb += np.dot(b1, b2 - b3 + b4 - b7 - b8)
    return lb


def plot_iteration(ax_spatial, circs, sctZ, lambda_m, delta_o, xn, i):
    """
    Plot the Gaussians in every iteration
    """
    if i == 0:
        plt.scatter(xn[:, 0], xn[:, 1], cmap=cm.gist_rainbow, s=5)
        sctZ = plt.scatter(lambda_m[:, 0], lambda_m[:, 1],
                           color='black', s=5)
    else:
        for circ in circs: circ.remove()
        circs = []
        for k in range(K):
            cov = delta_o
            circ = create_cov_ellipse(cov, lambda_m[k, :],
                                      color='r', alpha=0.3)
            circs.append(circ)
            ax_spatial.add_artist(circ)
        sctZ.set_offsets(lambda_m)
    plt.draw()
    plt.pause(0.001)
    return ax_spatial, circs, sctZ


def init_kmeans(xn, N, K):
    """
    Init points assignations (lambda_phi) with Kmeans clustering
    """
    lambda_phi = 0.1 / (K - 1) * np.ones((N, K))
    labels = KMeans(K).fit(xn).predict(xn)
    for i, lab in enumerate(labels):
        lambda_phi[i, lab] = 0.9
    return lambda_phi


def main():

    # Get data
    with open('{}'.format(args.dataset), 'r') as inputfile:
        data = pkl.load(inputfile)
        xn = data['xn']
    N, D = xn.shape

    if VERBOSE: init_time = time()

    # Priors
    alpha_o = [1.0] * K
    m_o = np.array([0.0, 0.0])
    beta_o = 0.01
    delta_o = np.zeros((D, D), long)
    np.fill_diagonal(delta_o, 1)

    # Variational parameters intialization
    lambda_phi = np.random.dirichlet(alpha_o, N) \
        if RANDOM_INIT else init_kmeans(xn, N, K)
    lambda_beta = beta_o + np.sum(lambda_phi, axis=0)
    lambda_m = np.tile(1. / lambda_beta, (2, 1)).T * \
               (beta_o * m_o + np.dot(lambda_phi.T, xn))

    # Plot configs
    if VERBOSE:
        plt.ion()
        fig = plt.figure(figsize=(10, 10))
        ax_spatial = fig.add_subplot(1, 1, 1)
        circs = []
        sctZ = None

    # Inference
    n_iters = 0
    lbs = []
    for i in xrange(MAX_ITERS):

        # Variational parameter updates
        lambda_pi = update_lambda_pi(lambda_phi, alpha_o)
        lambda_phi = update_lambda_phi(lambda_pi, lambda_m, lambda_beta,
                                       lambda_phi, delta_o, xn, N, D)
        lambda_beta = update_lambda_beta(lambda_phi, beta_o)
        lambda_m = update_lambda_m(lambda_beta, lambda_phi, m_o, beta_o, xn, D)

        # ELBO computation
        lb = elbo(xn, D, K, alpha_o, m_o, beta_o, delta_o,
                  lambda_pi, lambda_m, lambda_beta, lambda_phi)
        lbs.append(lb)
        n_iters += 1

        if VERBOSE:
            print('\n******* ITERATION {} *******'.format(i))
            print('lambda_pi: {}'.format(lambda_pi))
            print('lambda_beta: {}'.format(lambda_beta))
            print('lambda_m: {}'.format(lambda_m))
            print('lambda_phi: {}'.format(lambda_phi[0:9, :]))
            print('ELBO: {}'.format(lb))
            ax_spatial, circs, sctZ = plot_iteration(ax_spatial, circs, sctZ,
                                                     lambda_m, delta_o, xn,
                                                     i)

        # Break condition
        if i > 0 and abs(lb - lbs[i - 1]) < THRESHOLD:
            plt.savefig('{}.png'.format(PATH_IMAGE))
            break

    if VERBOSE:
        print('\n******* RESULTS *******')
        for k in range(K):
            print('Mu k{}: {}'.format(k, lambda_m[k, :]))
        final_time = time()
        exec_time = final_time - init_time
        print('Time: {} seconds'.format(exec_time))
        print('Iterations: {}'.format(n_iters))
        print('ELBOs: {}'.format(lbs))


if __name__ == '__main__': main()
