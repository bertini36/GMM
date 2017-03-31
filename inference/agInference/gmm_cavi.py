# -*- coding: UTF-8 -*-

"""
Gradient Ascent Variational Inference
process to approximate a Mixture of Gaussians (GMM)
[DOING]
"""

import argparse
import pickle as pkl
from time import time

import autograd.numpy as agnp
import autograd.scipy.special as agscipy
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from autograd import elementwise_grad
from sklearn.cluster import KMeans

from viz import create_cov_ellipse

"""
Parameters:
    * maxIter: Max number of iterations
    * dataset: Dataset path
    * k: Number of clusters
    * verbose: Printing time, intermediate variational parameters, plots, ...
"""

parser = argparse.ArgumentParser(description='CAVI in mixture of gaussians')
parser.add_argument('-maxIter', metavar='maxIter', type=int, default=100000)
parser.add_argument('-dataset', metavar='dataset', type=str,
                    default='../../data/synthetic/k2/data_k2_100.pkl')
parser.add_argument('-k', metavar='k', type=int, default=2)
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
PATH_IMAGE = 'generated/plot.png'
MACHINE_PRECISION = 2.2204460492503131e-16

# Gradient ascent step sizes of variational parameters
ps = {
    'lambda_phi': 0.001,
    'lambda_pi': 0.001,
    'lambda_m': 0.001,
    'lambda_w': 0.001,
    'lambda_beta': 0.001,
    'lambda_nu': 0.0001
}


def dirichlet_expectation(alpha, k):
    """
    Dirichlet expectation computation
    \Psi(\alpha_{k}) - \Psi(\sum_{i=1}^{K}(\alpha_{i}))
    """
    aux = agscipy.psi(agnp.add(alpha[k], agnp.finfo(agnp.float32).eps)) \
           - agscipy.psi(agnp.sum(alpha))
    return aux


def softmax(x):
    """
    Softmax computation
    e^{x} / sum_{i=1}^{K}(e^x_{i})
    """
    e_x = agnp.exp(x - agnp.max(x))
    return (e_x + agnp.finfo(agnp.float32).eps) / (
        e_x.sum(axis=0) + agnp.finfo(agnp.float32).eps)


def softplus(x):
    lt_34 = (x >= 34)
    gt_n37 = (x <= -36.8)
    neither_nor = agnp.logical_not(agnp.logical_or(lt_34, gt_n37))
    rval = agnp.where(gt_n37, 0., x)
    return agnp.where(neither_nor, agnp.log(1 + agnp.exp(x[neither_nor])), rval)


def NIW_sufficient_statistics(k, D, lambda_nu, lambda_w, lambda_m, lambda_beta):
    """
    Expectations Normal Inverse Wishart sufficient statistics computation
        E[\Sigma^{-1}\mu] = \nu * W^{-1} * m
        E[-1/2\Sigma^{-1}] = -1/2 * \nu * W^{-1}
        E[-1/2\mu.T\Sigma^{-1}\mu] = -D/2 * \beta^{-1} - \nu * m.T * W^{-1} * m
        E[-1/2log|\Sigma|] = D/2 * log(2) + 1/2 *
            sum_{i=1}^{D}(Psi(\nu/2 + (1-i)/2)) - 1/2 * log(|W|)
    """
    return \
        [
            agnp.dot(lambda_nu[k] * agnp.linalg.inv(lambda_w[k, :, :]),
                     lambda_m[k, :]),
            (-1 / 2.) * lambda_nu[k] * agnp.linalg.inv(lambda_w[k, :, :]),
            (-D / 2.) * (1 / lambda_beta[k]) - (1 / 2.) * lambda_nu[
                k] * agnp.dot(
                agnp.dot(lambda_m[k, :].T, agnp.linalg.inv(lambda_w[k, :, :])),
                lambda_m[k, :]),
            agnp.subtract(agnp.add((D / 2.) * agnp.log(2.), (1 / 2.) * agnp.sum(
                agscipy.psi(agnp.array(
                    [agnp.add((lambda_nu[k] / 2.), ((1 - i) / 2.)) for i in
                     range(D)])))), (1 / 2.) * agnp.log(
                agnp.linalg.det(lambda_w[k, :, :])))
        ]


def elbo((lambda_phi, lambda_pi, lambda_w, lambda_beta, lambda_nu)):
    """
    ELBO computation
    """
    lb = agscipy.gammaln(agnp.sum(alpha_o)) - agnp.sum(agscipy.gammaln(alpha_o)) \
         - agscipy.gammaln(agnp.sum(lambda_pi)) + agnp.sum(agscipy.gammaln(lambda_pi))
    lb -= N * D / 2. * agnp.log(2. * agnp.pi)
    for k in xrange(K):
        lb += -(nu_o[0] * D * agnp.log(2.)) / 2. + (lambda_nu[k] * D * agnp.log(2.)) / 2.
        lb += -agscipy.multigammaln(nu_o[0] / 2., D) + agscipy.multigammaln(lambda_nu[k] / 2., D)
        lb += (D / 2.) * agnp.log(beta_o[0]) - (D / 2.) * agnp.log(lambda_beta[k])
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
alpha_o = agnp.array([1.0] * K)
nu_o = agnp.array([3.0])
w_o = agnp.array([[20., 30.], [25., 40.]])
m_o = agnp.array([0.0, 0.0])
beta_o = agnp.array([0.7])

# Variational parameters intialization
lambda_phi = agnp.random.dirichlet(alpha_o, N) \
    if RANDOM_INIT else init_kmeans(xn, N, K)
lambda_pi = agnp.copy(alpha_o)
lambda_beta = agnp.array([beta_o] * K)
lambda_nu = agnp.array([nu_o] * K)
lambda_m = agnp.array([m_o] * K)
lambda_w = agnp.array([w_o] * K)

xn_xnt = agnp.array([agnp.outer(xn[n, :], xn[n, :].T) for n in range(N)])

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
        ax_spatial, circs, sctZ = plot_iteration(ax_spatial, circs,
                                                 sctZ, lambda_m, lambda_w,
                                                 lambda_nu, xn, D, n_iters)

    # Break condition
    if n_iters > 0 and lb - lbs[n_iters - 1] < THRESHOLD:
        plt.savefig(PATH_IMAGE)
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
