# -*- coding: UTF-8 -*-

"""
Coordinate Ascent Variational Inference
process to approximate a Mixture of Gaussians (GMM)
"""

import argparse
import pickle as pkl
from time import time

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import det, inv
from scipy import random
from scipy.special import gammaln, multigammaln, psi
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

parser = argparse.ArgumentParser(description='CAVI in mixture of gaussians')
parser.add_argument('-maxIter', metavar='maxIter', type=int, default=50)
parser.add_argument('-dataset', metavar='dataset', type=str,
                    default='/home/alberto/Documentos/data/Synthetic/k8/data_k8_1000.pkl')
parser.add_argument('-k', metavar='k', type=int, default=8)
parser.add_argument('--verbose', dest='verbose', action='store_true')
parser.add_argument('--no-verbose', dest='verbose', action='store_false')
parser.set_defaults(verbose=True)
parser.add_argument('--randomInit', dest='randomInit', action='store_true')
parser.add_argument('--no-randomInit', dest='randomInit', action='store_false')
parser.set_defaults(randomInit=True)
args = parser.parse_args()

MAX_ITERS = args.maxIter
K = args.k
VERBOSE = args.verbose
RANDOM_INIT = args.randomInit
THRESHOLD = 1e-12
PATH_IMAGE = 'img/gmm_cavi_k8_1000'


def dirichlet_expectation(alpha, k):
    """
    Dirichlet expectation computation
    \Psi(\alpha_{k}) - \Psi(\sum_{i=1}^{K}(\alpha_{i}))
    """
    return psi(alpha[k] + np.finfo(np.float32).eps) - psi(np.sum(alpha))


def softmax(x):
    """
    Softmax computation
    e^{x} / sum_{i=1}^{K}(e^x_{i})
    """
    e_x = np.exp(x - np.max(x))
    return (e_x + np.finfo(np.float32).eps) / (
        e_x.sum(axis=0) + np.finfo(np.float32).eps)


def generate_random_positive_matrix(D):
    """
    Generate a random semidefinite positive matrix
    """
    aux = random.rand(D, D)
    return np.dot(aux, aux.transpose())


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
                    lambda_m, w_o, beta_o, m_o, xn_xnt, K, N, D):
    """
    Update lambda_w
    w_o + m_o * m_o.T + sum_{n=1}^{N}(E_{q_{z}} I(z_{n}=i)x_{n}x_{n}.T)
    - lambda_beta * lambda_m * lambda_m.T
    """

    for k in range(K):
        aux = np.array([[0.] * D] * D)
        for n in range(N):
            aux += lambda_phi[n, k] * xn_xnt[n]
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
                psi([((lambda_nu[k] / 2.) + ((1 - i) / 2.)) for i in range(D)]))
            lambda_phi[n, k] -= (1 / 2.) * np.log(det(lambda_w[k, :, :]))
        lambda_phi[n, :] = softmax(lambda_phi[n, :])
    return lambda_phi


# TODO: Crear clase NIW y DIR
def NIW_sufficient_statistics(k, D, lambda_nu, lambda_w, lambda_m, lambda_beta):
    """
    Expectations Normal Inverse Wishart sufficient statistics computation
        E[\Sigma^{-1}\mu] = \nu * W^{-1} * m
        E[-1/2\Sigma^{-1}] = -1/2 * \nu * W^{-1}
        E[-1/2\mu.T\Sigma^{-1}\mu] = -D/2 * \beta^{-1} - \nu * m.T * W^{-1} * m
        E[-1/2log|\Sigma|] = D/2 * log(2) + 1/2 *
            sum_{i=1}^{D}(Psi(\nu/2 + (1-i)/2)) - 1/2 * log(|W|)
    """
    inv_lambda_w = inv(lambda_w[k, :, :])
    return np.array([
        np.dot(lambda_nu[k] * inv_lambda_w, lambda_m[k, :]),
        (-1 / 2.) * lambda_nu[k] * inv_lambda_w,
        (-D / 2.) * (1 / lambda_beta[k]) - (1 / 2.) * lambda_nu[k] * np.dot(
            np.dot(lambda_m[k, :].T, inv_lambda_w), lambda_m[k, :]),
        (D / 2.) * np.log(2.) + (1 / 2.) * np.sum(
            [psi(((lambda_nu[k] / 2.) + ((1 - i) / 2.))) for i in range(D)]) - (
            1 / 2.) * np.log(det(lambda_w[k, :, :]))
    ])


def DIR_sufficient_statistics(k, lambda_pi):
    """
    Expectations Dirichlet sufficient statistics computation
    E[log(\pi_{k})] = \Psi(\alpha_{k}) - \Psi(\sum_{i=1}^{K}(\alpha_{i}))
    """
    return dirichlet_expectation(lambda_pi, k)


def NIW_nat_params(k, D, lambda_m, lambda_w, lambda_beta, lambda_nu):
    """
    Normal Inverse Wishart natural parameters
    m.T * beta
    W + beta * m * m.T
    beta
    nu - D - 2
    """
    return np.array([
        lambda_beta[k] * lambda_m[k, :],
        (lambda_w[k, :, :] + lambda_beta[k] *
         np.outer(lambda_m[k, :], lambda_m[k, :])),
        lambda_beta[k],
        lambda_nu[k] + D + 2
    ])


def DIR_nat_params(k, lambda_pi):
    """
    Dirichlet natural parameters
    \alpha_{k} - 1
    """
    return lambda_pi[k] - 1


def elbo(lambda_phi, lambda_pi, lambda_w, lambda_beta,
         lambda_nu, alpha_o, nu_o, beta_o, w_o,  N, D):
    """
    ELBO computation
    """

    elbo = K * gammaln(np.sum(alpha_o)) - np.sum(gammaln(alpha_o)) \
           - K * gammaln(np.sum(lambda_pi)) + np.sum(gammaln(lambda_pi))
    elbo -= N * D / 2. * np.log(2. * np.pi)
    for k in xrange(K):
        elbo += -(nu_o[0] * D * np.log(2.)) /\
                2. + (lambda_nu[k] * D * np.log(2.)) / 2.
        elbo += -multigammaln(nu_o[0] / 2., D)\
                + multigammaln(lambda_nu[k] / 2., D)
        elbo += (D / 2.) * np.log(beta_o[0]) - (D / 2.) * np.log(lambda_beta[k])
        elbo += (nu_o[0] / 2.) * np.log(det(w_o)) \
                - (lambda_nu[k] / 2.) * np.log(det(lambda_w[k, :, :]))
        elbo -= np.dot(np.log(lambda_phi[:, k]).T, lambda_phi[:, k])
    return elbo


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
    alpha_o = np.array([1.0] * K)
    nu_o = np.array([3.0])
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

    xn_xnt = np.array([np.outer(xn[n, :], xn[n, :].T) for n in range(N)])

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
    for _ in range(MAX_ITERS):
        print('\n******* ITERATION {} *******'.format(n_iters))

        # Variational parameter updates
        lambda_pi = update_lambda_pi(lambda_pi, lambda_phi, alpha_o)
        Nks = np.sum(lambda_phi, axis=0)
        lambda_beta = update_lambda_beta(lambda_beta, beta_o, Nks)
        lambda_nu = update_lambda_nu(lambda_nu, nu_o, Nks)
        lambda_m = update_lambda_m(lambda_m, lambda_phi, lambda_beta, m_o,
                                   beta_o, xn, N, D)
        lambda_w = update_lambda_w(lambda_w, lambda_phi, lambda_beta,
                                   lambda_m, w_o, beta_o, m_o, xn_xnt, K, N, D)
        lambda_phi = update_lambda_phi(lambda_phi, lambda_pi, lambda_m,
                                       lambda_nu, lambda_w, lambda_beta,
                                       xn, N, K, D)

        # ELBO computation
        lb = elbo(lambda_phi, lambda_pi, lambda_w, lambda_beta,
                  lambda_nu, alpha_o, nu_o, beta_o, w_o,  N, D)
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
            if D == 2:
                ax_spatial, circs, sctZ = plot_iteration(ax_spatial, circs,
                                                         sctZ, lambda_m,
                                                         lambda_w, lambda_nu,
                                                         xn, D, n_iters)

        # Break condition
        improve = abs(lb - lbs[n_iters - 1])
        if VERBOSE: print('Improve: {}'.format(improve))
        if n_iters > 0 and improve < THRESHOLD:
            if VERBOSE and D == 2: plt.savefig('{}.png'.format(PATH_IMAGE))
            break

        n_iters += 1

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
            ax.scatter(xn[:, 0], xn[:, 1], xn[:, 2], c=np.array(
                [np.argmax(lambda_phi[n, :]) for n in xrange(N)]),
                       cmap=cm.gist_rainbow, s=5)
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            plt.show()
        plt.gcf().clear()
        plt.plot(np.arange(len(lbs)), lbs)
        plt.savefig('elbos.png')


if __name__ == '__main__': main()
