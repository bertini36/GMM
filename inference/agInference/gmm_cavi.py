# -*- coding: UTF-8 -*-

"""
Gradient Ascent Variational Inference
process to approximate a Mixture of Gaussians (GMM)
[DOING]
"""

import argparse
import csv
import pickle as pkl
from time import time

import autograd.numpy as agnp
import autograd.scipy.special as agscipy
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from autograd import elementwise_grad
from scipy import random
from sklearn.cluster import KMeans

from viz import create_cov_ellipse

"""
Parameters:
    * maxIter: Max number of iterations
    * dataset: Dataset path
    * k: Number of clusters
    * verbose: Printing time, intermediate variational parameters, plots, ...
    * randomInit: Init assignations randomly or with Kmeans
    * exportAssignments: If true generate a csv with the cluster assignments
"""

parser = argparse.ArgumentParser(description='CAVI in mixture of gaussians')
parser.add_argument('-maxIter', metavar='maxIter', type=int, default=200)
parser.add_argument('-dataset', metavar='dataset', type=str, default='')
parser.add_argument('-k', metavar='k', type=int, default=2)
parser.add_argument('--verbose', dest='verbose', action='store_true')
parser.add_argument('--no-verbose', dest='verbose', action='store_false')
parser.set_defaults(verbose=True)
parser.add_argument('--randomInit', dest='randomInit', action='store_true')
parser.add_argument('--no-randomInit', dest='randomInit', action='store_false')
parser.set_defaults(randomInit=False)
parser.add_argument('--exportAssignments',
                    dest='exportAssignments', action='store_true')
parser.add_argument('--no-exportAssignments',
                    dest='exportAssignments', action='store_false')
parser.set_defaults(exportAssignments=True)
args = parser.parse_args()

MAX_ITERS = args.maxIter
K = args.k
VERBOSE = args.verbose
RANDOM_INIT = args.randomInit
THRESHOLD = 1e-6
PATH_IMAGE = 'generated/plot.png'
MACHINE_PRECISION = 2.2204460492503131e-16
EXPORT_ASSIGNMENTS = args.exportAssignments

# Gradient ascent step sizes of variational parameters
ps = {
    'lambda_phi': 0.001,
    'lambda_pi': 0.001,
    'lambda_m': 0.001,
    'lambda_w': 0.001,
    'lambda_beta': 0.001,
    'lambda_nu': 0.0001
}


def generate_random_positive_matrix(D):
    """
    Generate a random semidefinite positive matrix
    :param D: Dimension
    :return: DxD matrix
    """
    aux = random.rand(D, D)
    return np.dot(aux, aux.transpose())


def elbo((lambda_phi, lambda_pi, lambda_w, lambda_beta, lambda_nu)):
    """
    ELBO computation
    """
    lb = agscipy.gammaln(agnp.sum(alpha_o)) - np.sum(agscipy.gammaln(alpha_o)) \
           - agscipy.gammaln(agnp.sum(lambda_pi)) + agnp.sum(agscipy.gammaln(lambda_pi))
    lb -= N * D / 2. * np.log(2. * np.pi)
    for k in xrange(K):
        lb += -(nu_o[0] * D * np.log(2.)) / 2. + (lambda_nu[k] * D * np.log(2.)) / 2.
        lb += -agscipy.multigammaln(nu_o[0] / 2., D) + agscipy.multigammaln(lambda_nu[k] / 2., D)
        lb += (D / 2.) * agnp.log(agnp.absolute(beta_o[0])) - (D / 2.) * agnp.log(agnp.absolute(lambda_beta[k]))
        lb += (nu_o[0] / 2.) * agnp.log(agnp.linalg.det(w_o)) - (lambda_nu[k] / 2.) * agnp.log(agnp.linalg.det(lambda_w[k, :, :]))
        lb -= agnp.dot(agnp.log(lambda_phi[:, k]).T, lambda_phi[:, k])
    return lb


def plot_iteration(ax_spatial, circs, sctZ, lambda_m,
                   lambda_w, lambda_nu, xn, D, n_iters):
    """
    Plot the Gaussians in every iteration
    """
    if n_iters == 0:
        plt.scatter(xn[:, 0], xn[:, 1], cmap=cm.gist_rainbow, s=5)
        sctZ = plt.scatter(lambda_m[:, 0], lambda_m[:, 1],
                           color='black', s=5)
    else:
        for circ in circs: circ.remove()
        circs = []
        for k in range(K):
            cov = lambda_w[k, :, :] / (lambda_nu[k] - D - 1)
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
    lambda_phi = 0.1 / (K - 1) * agnp.ones((N, K))
    labels = KMeans(K).fit(xn).predict(xn)
    for i, lab in enumerate(labels):
        lambda_phi[i, lab] = 0.9
    return lambda_phi


# Get data
with open('{}'.format(args.dataset), 'r') as inputfile:
    data = pkl.load(inputfile)
    xn = agnp.array(data['xn'])
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
    if RANDOM_INIT else init_kmeans(xn, N, K)
lambda_pi = np.zeros(shape=K)
lambda_beta = np.zeros(shape=K)
lambda_nu = np.zeros(shape=K)
lambda_m = np.zeros(shape=(K, D))
lambda_w = np.zeros(shape=(K, D, D))

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
for _ in range(2):

    # Maximize ELBO
    Nks = agnp.sum(lambda_phi, axis=0)
    grads = elementwise_grad(elbo)((lambda_phi, lambda_pi,
                                    lambda_w, lambda_beta, lambda_nu))

    # Variational parameter updates (gradient ascent)
    lambda_phi -= ps['lambda_phi'] * grads[0]
    lambda_pi -= ps['lambda_pi'] * grads[1]
    lambda_m -= ps['lambda_m'] * grads[2]
    lambda_w -= ps['lambda_w'] * grads[3]
    lambda_beta -= ps['lambda_beta'] * grads[4]
    lambda_nu -= ps['lambda_nu'] * grads[5]

    # ELBO computation
    lb = elbo((lambda_phi, lambda_pi, lambda_w, lambda_beta, lambda_nu))
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
        if VERBOSE and D == 2: plt.savefig(PATH_IMAGE)
        break

    n_iters += 1

zn = np.array([np.argmax(lambda_phi[n, :]) for n in xrange(N)])

if VERBOSE:
    print('\n******* RESULTS *******')
    for k in range(K):
        print('Mu k{}: {}'.format(k, lambda_m[k, :]))
        print('SD k{}: {}'.format(k, np.sqrt(
            np.diag(lambda_w[k, :, :] / (lambda_nu[k] - D - 1)))))
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

if EXPORT_ASSIGNMENTS:
    with open('generated/assignments.csv', 'wb') as output:
        writer = csv.writer(output, delimiter=';', quotechar='',
                            escapechar='\\', quoting=csv.QUOTE_NONE)
        writer.writerow(['zn'])
        for i in range(len(zn)):
            writer.writerow([zn[i]])
