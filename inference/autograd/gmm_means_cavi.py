# -*- coding: UTF-8 -*-

"""
Gradient Ascent Variational Inference
process to approximate a Mixture of Gaussians (GMM) with known precisions
"""

import argparse
import math
import pickle as pkl
from time import time

import autograd.numpy as agnp
import matplotlib.pyplot as plt
from autograd import elementwise_grad

from libs.common import dirichlet_expectation, log_beta_function
from libs.viz import plot_iteration

"""
Parameters:
    * maxIter: Max number of iterations
    * dataset: Dataset path
    * k: Number of clusters
    * verbose: Printing time, intermediate variational parameters, plots, ...
    
Execution: 
    python gmm_means_cavi.py -dataset data_k2_1000.pkl -k 2 -verbose
"""

parser = argparse.ArgumentParser(description='CAVI in mixture of gaussians')
parser.add_argument('-maxIter', metavar='maxIter', type=int, default=1000)
parser.add_argument('-dataset', metavar='dataset', type=str,
                    default='../../data/synthetic/2D/k2/data_k2_1000.pkl')
parser.add_argument('-k', metavar='k', type=int, default=2)
parser.add_argument('-verbose', dest='verbose', action='store_true')
parser.set_defaults(verbose=False)
args = parser.parse_args()

K = args.k
VERBOSE = args.verbose
THRESHOLD = 1e-3
PATH_IMAGE = 'generated/gmm_means_cavi'
MACHINE_PRECISION = 2.2204460492503131e-16

# Gradient ascent step sizes of variational parameters
ps = {
    'lambda_pi': 0.1,
    'lambda_phi': 0.1,
    'lambda_m': 0.001,
    'lambda_beta': 0.1
}


def initialize():
    """
    Variational parameters initialization
    """
    phi = agnp.random.dirichlet(alpha_o, N)
    lambda_pi = alpha_o + agnp.sum(phi, axis=0)
    lambda_mu_beta = beta_o + agnp.sum(phi, axis=0)
    lambda_mu_m = agnp.tile(1. / lambda_mu_beta, (2, 1)).T * (
        beta_o * m_o + agnp.dot(phi.T, xn))
    return lambda_pi, phi, lambda_mu_m, lambda_mu_beta


def elbo((lambda_pi, lambda_phi, lambda_m, lambda_beta)):
    """
    ELBO computation
    """
    lambda_pi_aux = agnp.log(1 + agnp.exp(lambda_pi) + MACHINE_PRECISION)
    phi_aux = (agnp.exp(lambda_phi) + MACHINE_PRECISION) / agnp.tile(
        (agnp.exp(lambda_phi) + MACHINE_PRECISION).sum(axis=1), (K, 1)).T
    lambda_mu_beta_aux = agnp.log(1 + agnp.exp(lambda_beta) + MACHINE_PRECISION)
    ELBO = log_beta_function(lambda_pi_aux) - log_beta_function(alpha_o) \
           + agnp.dot(alpha_o - lambda_pi_aux,
                      dirichlet_expectation(lambda_pi_aux)) \
           + K / 2. * agnp.log(agnp.linalg.det(beta_o * delta_o)) + K * D / 2.
    for k in range(K):
        ELBO -= beta_o / 2. * agnp.dot((
            lambda_m[k, :] - m_o), agnp.dot(delta_o, (lambda_m[k, :] - m_o).T))
        for n in range(N):
            ELBO += phi_aux[n, k] * (
                dirichlet_expectation(lambda_pi_aux)[k] - agnp.log(
                    phi_aux[n, k]) + 1 / 2. * agnp.log(1. / (2. * math.pi))
                - 1 / 2. * agnp.dot((xn[n, :] - lambda_m[k, :]),
                                    agnp.dot(delta_o,
                                             (xn[n, :] - lambda_m[k, :]).T))
                - D / (2. * lambda_mu_beta_aux[k]))
    return -ELBO


# Get data
with open('{}'.format(args.dataset), 'r') as inputfile:
    data = pkl.load(inputfile)
    xn = data['xn']
N, D = xn.shape

if VERBOSE: init_time = time()

# Priors
alpha_o = [1.] * K
m_o = agnp.array([0., 0.])
beta_o = 0.01
delta_o = agnp.array([[1., 0., 0., 1.][0:D], [1., 0., 0., 1.][D:2 * D]])

# Variational parameters intialization
lambda_pi, lambda_phi, lambda_m, lambda_beta = initialize()

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
    grads = elementwise_grad(elbo) \
        ((lambda_pi, lambda_phi, lambda_m, lambda_beta))

    # Variational parameter updates (gradient ascent)
    lambda_pi -= ps['lambda_pi'] * grads[0]
    lambda_phi -= ps['lambda_phi'] * grads[1]
    lambda_m -= ps['lambda_m'] * grads[2]
    lambda_beta -= ps['lambda_beta'] * grads[3]

    # ELBO computation
    lb = elbo((lambda_pi, lambda_phi, lambda_m, lambda_beta))
    lbs.append(lb)

    if VERBOSE:
        print('\n******* ITERATION {} *******'.format(n_iters))
        print('lambda_pi: {}'.format(lambda_pi))
        print('lambda_beta: {}'.format(lambda_beta))
        print('lambda_m: {}'.format(lambda_m))
        print('lambda_phi: {}'.format(lambda_phi[0:9, :]))
        print('ELBO: {}'.format(lb))
        ax_spatial, circs, sctZ = plot_iteration(ax_spatial, circs, sctZ,
                                                 lambda_m, delta_o, xn,
                                                 n_iters, K)

    # Break condition
    if n_iters > 0 and abs(lb - lbs[n_iters - 1]) < THRESHOLD:
        plt.savefig('{}.png'.format(PATH_IMAGE))
        break

    n_iters += 1

if VERBOSE:
    print('\n******* RESULTS *******')
    for k in range(K):
        print('Mu k{}: {}'.format(k, lambda_m[k, :]))
    final_time = time()
    exec_time = final_time - init_time
    print('Time: {} seconds'.format(exec_time))
    print('Iterations: {}'.format(n_iters))
    print('ELBOs: {}'.format(lbs))
