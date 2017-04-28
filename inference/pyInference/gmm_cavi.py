# -*- coding: UTF-8 -*-

"""
Coordinate Ascent Variational Inference
process to approximate a Mixture of Gaussians (GMM)
"""

import argparse
import csv
import pickle as pkl
import sys
from time import time

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import det, inv
from scipy.special import gammaln, multigammaln, psi

from libs.common import (dirichlet_expectation,
                         generate_random_positive_matrix, init_kmeans, softmax)
from libs.viz import plot_iteration

"""
Parameters:
    * maxIter: Max number of iterations
    * dataset: Dataset path (pkl)
    * k: Number of clusters
    * verbose: Printing time, intermediate variational parameters, plots, ...
    * randomInit: Init assignations randomly or with Kmeans
    * exportAssignments: If true generate a csv with the cluster assignments
    * exportVariationalParameters: If true generate a pkl of a dictionary with
                                   the variational parameters inferred

Execution:
    python gmm_cavi.py -dataset data_k2_1000.pkl -k 2 -verbose 
                       -exportAssignments -exportVariationalParameters
"""

parser = argparse.ArgumentParser(description='CAVI in mixture of gaussians')
parser.add_argument('-maxIter', metavar='maxIter', type=int, default=100)
parser.add_argument('-dataset', metavar='dataset', type=str,
                    default='../../data/synthetic/2D/k2/data_k2_1000.pkl')
parser.add_argument('-k', metavar='k', type=int, default=2)
parser.set_defaults(exportVariationalParameters=False)
parser.add_argument('-verbose', dest='verbose', action='store_true')
parser.set_defaults(verbose=False)
parser.add_argument('-randomInit', dest='randomInit', action='store_true')
parser.set_defaults(randomInit=False)
parser.add_argument('-exportAssignments',
                    dest='exportAssignments', action='store_true')
parser.set_defaults(exportAssignments=False)
parser.add_argument('-exportVariationalParameters',
                    dest='exportVariationalParameters', action='store_true')
args = parser.parse_args()

K = args.k
VERBOSE = args.verbose
THRESHOLD = 1e-6


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
    lb = gammaln(np.sum(alpha_o)) - np.sum(gammaln(alpha_o)) \
           - gammaln(np.sum(lambda_pi)) + np.sum(gammaln(lambda_pi))
    lb -= N * D / 2. * np.log(2. * np.pi)
    for k in xrange(K):
        lb += -(nu_o[0] * D * np.log(2.)) / 2. + (lambda_nu[k] * D * np.log(2.)) / 2.
        lb += -multigammaln(nu_o[0] / 2., D) + multigammaln(lambda_nu[k] / 2., D)
        lb += (D / 2.) * np.log(np.absolute(beta_o[0])) - (D / 2.) * np.log(np.absolute(lambda_beta[k]))
        lb += (nu_o[0] / 2.) * np.log(det(w_o)) - (lambda_nu[k] / 2.) * np.log(det(lambda_w[k, :, :]))
        lb -= np.dot(np.log(lambda_phi[:, k]).T, lambda_phi[:, k])
    return lb


def main():
    try:
        if not('.pkl' in args.dataset): raise Exception('input_format')

        # Get data
        with open('{}'.format(args.dataset), 'r') as inputfile:
            data = pkl.load(inputfile)
            xn = data['xn']
        N, D = xn.shape

        if VERBOSE: init_time = time()

        # Priors
        alpha_o = np.array([1.0] * K)
        nu_o = np.array([float(D)])
        if nu_o[0] < D: raise Exception('degrees_of_freedom')
        w_o = generate_random_positive_matrix(D)
        m_o = np.array([0.0] * D)
        beta_o = np.array([0.7])

        # Variational parameters intialization
        lambda_phi = np.random.dirichlet(alpha_o, N) \
            if args.randomInit else init_kmeans(xn, N, K)
        lambda_pi = np.zeros(shape=K)
        lambda_beta = np.zeros(shape=K)
        lambda_nu = np.zeros(shape=K)
        lambda_m = np.zeros(shape=(K, D))
        lambda_w = np.zeros(shape=(K, D, D))

        # Plot configs
        if VERBOSE and D == 2:
            plt.ion()
            fig = plt.figure(figsize=(10, 10))
            ax_spatial = fig.add_subplot(1, 1, 1)
            circs = []
            sctZ = None

        # Inference
        lbs = []
        n_iters = 0
        for _ in range(args.maxIter):

            # Variational parameter updates
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
            lb = elbo(lambda_phi, lambda_pi, lambda_beta,lambda_nu,
                      lambda_w, alpha_o, beta_o, nu_o, w_o,  N, D)
            lbs.append(lb)

            if VERBOSE:
                print('\n******* ITERATION {} *******'.format(n_iters))
                print('lambda_pi: {}'.format(lambda_pi))
                print('lambda_beta: {}'.format(lambda_beta))
                print('lambda_nu: {}'.format(lambda_nu))
                print('lambda_m: {}'.format(lambda_m))
                print('lambda_w: {}'.format(lambda_w))
                print('lambda_phi: {}'.format(lambda_phi[0:9, :]))
                print('ELBO: {}'.format(lb))
                print('\n******* ITERATION {} *******'.format(n_iters))
                if D == 2:
                    covs = [lambda_w[k, :, :] / (lambda_nu[k] - D - 1)
                            for k in range(K)]
                    ax_spatial, circs, sctZ = plot_iteration(ax_spatial, circs,
                                                             sctZ, lambda_m,
                                                             covs, xn,
                                                             n_iters, K)

            # Break condition
            improve = lb - lbs[n_iters - 1]
            if VERBOSE: print('Improve: {}'.format(improve))
            if n_iters > 0 and improve < THRESHOLD:
                if VERBOSE and D == 2: plt.savefig('generated/plot.png')
                break

            n_iters += 1

        zn = np.array([np.argmax(lambda_phi[n, :]) for n in xrange(N)])

        if VERBOSE:
            print('\n******* RESULTS *******')
            for k in range(K):
                print('Mu k{}: {}'.format(k, lambda_m[k, :]))
            final_time = time()
            exec_time = final_time - init_time
            print('Time: {} seconds'.format(exec_time))
            print('Iterations: {}'.format(n_iters))
            print('ELBOs: {}'.format(lbs))
            if D == 3:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(xn[:, 0], xn[:, 1], xn[:, 2],
                           c=zn, cmap=cm.gist_rainbow, s=5)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                plt.show()
            plt.gcf().clear()
            plt.plot(np.arange(len(lbs)), lbs)
            plt.ylabel('ELBO')
            plt.xlabel('Iterations')
            plt.savefig('generated/elbos.png')

        if args.exportAssignments:
            with open('generated/assignments.csv', 'wb') as output:
                writer = csv.writer(output, delimiter=';', quotechar='',
                                    escapechar='\\', quoting=csv.QUOTE_NONE)
                writer.writerow(['zn'])
                for i in range(len(zn)):
                    writer.writerow([zn[i]])

        if args.exportVariationalParameters:
            with open('generated/variational_parameters.pkl', 'w') as output:
                pkl.dump({'lambda_pi': lambda_pi, 'lambda_m': lambda_m,
                          'lambda_beta': lambda_beta, 'lambda_nu': lambda_nu,
                          'lambda_w': lambda_w, 'K': K, 'D': D}, output)

    except IOError:
        print('File not found!')
    except Exception as e:
        if e.args[0] == 'input_format': print('Input must be a pkl file')
        elif e.args[0] == 'degrees_of_freedom':
            print('Degrees of freedom can not be smaller than D!')
        else:
            print('Unexpected error: {}'.format(sys.exc_info()[0]))
            raise


if __name__ == '__main__': main()
