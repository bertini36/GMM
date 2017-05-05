# -*- coding: UTF-8 -*-

"""
Gradient Ascent Variational Inference
process to approximate a Mixture of Gaussians (GMM)
[DOING]
"""

from __future__ import absolute_import

import argparse
import csv
import os
import pickle as pkl
import sys
from time import time

import autograd.numpy as agnp
import autograd.scipy.special as agscipy
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import det, inv
from scipy.special import psi
from autograd import elementwise_grad

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import dirichlet_expectation, init_kmeans, log_, \
                  log_beta_function, softmax, softplus

from common import generate_random_positive_matrix, softmax
from viz import plot_iteration



"""
Parameters:
    * maxIter: Max number of iterations
    * dataset: Dataset path
    * k: Number of clusters
    * verbose: Printing time, intermediate variational parameters, plots, ...
    * randomInit: Init assignations randomly or with Kmeans
    * exportAssignments: If true generate a csv with the cluster assignments
     
Execution:
    python gmm_gavi.py -dataset data_k2_1000.pkl -k 2 -verbose
"""

parser = argparse.ArgumentParser(description='CAVI in mixture of gaussians')
parser.add_argument('-maxIter', metavar='maxIter', type=int, default=1000)
parser.add_argument('-dataset', metavar='dataset', type=str,
                    default='../../data/synthetic/2D/k2/data_k2_1000.pkl')
parser.add_argument('-k', metavar='k', type=int, default=2)
parser.add_argument('-verbose', dest='verbose', action='store_true')
parser.set_defaults(verbose=False)
parser.add_argument('-randomInit', dest='randomInit', action='store_true')
parser.set_defaults(randomInit=False)
parser.add_argument('-exportAssignments',
                    dest='exportAssignments', action='store_true')
parser.set_defaults(exportAssignments=False)
args = parser.parse_args()

K = args.k
VERBOSE = args.verbose
THRESHOLD = 1e-6
PATH_IMAGE = 'generated/plot.png'
MACHINE_PRECISION = 2.2204460492503131e-16

# Gradient ascent step sizes of variational parameters
ps = {
    'lambda_phi': 0.001,
    'lambda_pi': 0.001,
    'lambda_m': 0.001,
    'lambda_w': 0.001,
    'lambda_beta': 0.001,
    'lambda_nu': 0.001
}


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


def dirichlet_expectation_k(alpha, k):
    """
    Dirichlet expectation computation
    \Psi(\alpha_{k}) - \Psi(\sum_{i=1}^{K}(\alpha_{i}))
    """
    return psi(alpha[k] + np.finfo(np.float32).eps) - psi(np.sum(alpha))


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


def elbo((lambda_pi, lambda_phi, lambda_m, lambda_beta, lambda_nu, lambda_w)):
    """
    ELBO computation
    """
    e3 = e2 = h2 = 0

    e1 = - log_beta_function(alpha_o) \
         + agnp.dot((alpha_o - agnp.ones(K)), dirichlet_expectation(lambda_pi))
    h1 = log_beta_function(lambda_pi) \
         - agnp.dot((lambda_pi - agnp.ones(K)),
                    dirichlet_expectation(lambda_pi))
    logdet = agnp.log(
        agnp.array([agnp.linalg.det(lambda_w[k, :, :]) for k in range(K)]))
    logDeltak = agscipy.psi(lambda_nu / 2.) \
                + agscipy.psi((lambda_nu - 1.) / 2.) + 2. * agnp.log(
        2.) + logdet

    for n in range(N):
        e2 += agnp.dot(lambda_phi[n, :], dirichlet_expectation(lambda_pi))
        h2 += -agnp.dot(lambda_phi[n, :], log_(lambda_phi[n, :]))
        product = agnp.array([agnp.dot(agnp.dot(
            xn[n, :] - lambda_m[k, :], lambda_w[k, :, :]),
            (xn[n, :] - lambda_m[k, :]).T) for k in range(K)])
        e3 += 1. / 2 * agnp.dot(lambda_phi[n, :],
                                (logDeltak - 2. * agnp.log(2 * agnp.pi) -
                                 lambda_nu * product - 2. / lambda_beta).T)

    product = agnp.array([agnp.dot(agnp.dot(lambda_m[k, :] - m_o,
                                            lambda_w[k, :, :]),
                                   (lambda_m[k, :] - m_o).T) for k in
                          range(K)])
    traces = agnp.array([agnp.trace(agnp.dot(agnp.linalg.inv(w_o),
                                             lambda_w[k, :, :])) for k in
                         range(K)])
    h4 = agnp.sum((1. + agnp.log(2. * agnp.pi) - 1. / 2 * (
    agnp.log(lambda_beta) + logdet)))
    logB = lambda_nu / 2. * logdet + lambda_nu * agnp.log(
        2.) + 1. / 2 * agnp.log(agnp.pi) \
           + agscipy.gammaln(lambda_nu / 2.) + agscipy.gammaln(
        (lambda_nu - 1) / 2.)
    h5 = agnp.sum((logB - (lambda_nu - 3.) / 2. * logDeltak + lambda_nu))
    e4 = agnp.sum(
        (1. / 2 * (agnp.log(beta_o) + logDeltak - 2 * agnp.log(2. * agnp.pi)
                   - beta_o * lambda_nu * product - 2. * beta_o / lambda_beta)))
    logB = nu_o / 2. * agnp.log(agnp.linalg.det(w_o)) + nu_o * agnp.log(2.) \
           + 1. / 2 * agnp.log(agnp.pi) + agscipy.gammaln(
        nu_o / 2.) + agscipy.gammaln((nu_o - 1) / 2.)
    e5 = agnp.sum(
        (-logB + (nu_o - 3.) / 2. * logDeltak - lambda_nu / 2. * traces))

    return e1 + e2 + e3 + e4 + e5 + h1 + h2 + h4 + h5


# Get data
with open('{}'.format(args.dataset), 'r') as inputfile:
    data = pkl.load(inputfile)
    xn = agnp.array(data['xn'])
N, D = xn.shape

if VERBOSE: init_time = time()

# Priors
alpha_o = agnp.array([1.0] * K)
nu_o = agnp.array([float(D)])
if nu_o[0] < D: raise Exception('degrees_of_freedom')
w_o = generate_random_positive_matrix(D)
m_o = agnp.array([0.0] * D)
beta_o = agnp.array([0.7])

# Variational parameters intialization
lambda_phi = np.random.dirichlet(alpha_o, N) \
    if args.randomInit else init_kmeans(xn, N, K)
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

print('lambda_pi: {}'.format(lambda_pi))
print('lambda_beta: {}'.format(lambda_beta))
print('lambda_nu: {}'.format(lambda_nu))
print('lambda_m: {}'.format(lambda_m))
print('lambda_w: {}'.format(lambda_w))
print('lambda_phi: {}'.format(lambda_phi[0:9, :]))


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
for _ in range(args.maxIter):

    # Maximize ELBO
    grads = elementwise_grad(elbo)((lambda_pi, lambda_phi, lambda_m,
                                    lambda_beta, lambda_nu, lambda_w))

    # Variational parameter updates (gradient ascent)
    lambda_pi -= ps['lambda_pi'] * grads[0]
    lambda_phi -= ps['lambda_phi'] * grads[1]
    lambda_m -= ps['lambda_m'] * grads[2]
    lambda_beta -= ps['lambda_beta'] * grads[3]
    lambda_nu -= ps['lambda_nu'] * grads[4]
    lambda_w -= ps['lambda_w'] * grads[5]

    lambda_phi = agnp.array([softmax(lambda_phi[i]) for i in range(N)])
    lambda_beta = softplus(lambda_beta)
    lambda_nu = softplus(lambda_nu)
    lambda_pi = softplus(lambda_pi)
    lambda_w = agnp.array([agnp.dot(lambda_w[k], lambda_w[k].T)
                           for k in range(K)])

    """
    lambda_pi = update_lambda_pi(lambda_pi, lambda_phi, alpha_o)
    Nks = np.sum(lambda_phi, axis=0)
    lambda_beta = update_lambda_beta(lambda_beta, beta_o, Nks)
    lambda_nu = update_lambda_nu(lambda_nu, nu_o, Nks)
    lambda_m -= ps['lambda_m'] * grads[2]
    lambda_w = lambda_w - (ps['lambda_w'] * grads[5])
    lambda_phi -= ps['lambda_phi'] * grads[1]
    lambda_phi = agnp.array([softmax(lambda_phi[i]) for i in range(N)])
    """

    # ELBO computation
    lb = elbo((lambda_pi, lambda_phi, lambda_m,
               lambda_beta, lambda_nu, lambda_w))
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
    if n_iters > 0 and abs(improve) < THRESHOLD:
        if VERBOSE and D == 2: plt.savefig(PATH_IMAGE)
        break

    n_iters += 1

zn = agnp.array([agnp.argmax(lambda_phi[n, :]) for n in range(N)])

if VERBOSE:
    print('\n******* RESULTS *******')
    for k in range(K):
        print('Mu k{}: {}'.format(k, lambda_m[k, :]))
        print('SD k{}: {}'.format(k, agnp.sqrt(
            agnp.diag(lambda_w[k, :, :] / (lambda_nu[k] - D - 1)))))
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
    plt.plot(agnp.arange(len(lbs)), lbs)
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
