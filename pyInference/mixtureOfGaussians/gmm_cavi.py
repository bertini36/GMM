# -*- coding: UTF-8 -*-

"""
Coordinate Ascent Variational Inference
process to approximate a mixture of gaussians
[DOING]
"""

import argparse
import pickle as pkl
from time import time, sleep

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import det, inv
from scipy.special import gammaln, multigammaln, psi
from viz import create_cov_ellipse
from scipy.stats import invwishart

parser = argparse.ArgumentParser(description='CAVI in mixture of gaussians')
parser.add_argument('-maxIter', metavar='maxIter', type=int, default=100)
parser.add_argument('-dataset', metavar='dataset',
                    type=str, default='../../data/data_k8_1000.pkl')
parser.add_argument('-k', metavar='k', type=int, default=8)
parser.add_argument('--timing', dest='timing', action='store_true')
parser.add_argument('--no-timing', dest='timing', action='store_false')
parser.set_defaults(timing=False)
parser.add_argument('--getNIter', dest='getNIter', action='store_true')
parser.add_argument('--no-getNIter', dest='getNIter', action='store_false')
parser.set_defaults(getNIter=True)
parser.add_argument('--getELBOs', dest='getELBOs', action='store_true')
parser.add_argument('--no-getELBOs', dest='getELBOs', action='store_false')
parser.set_defaults(getELBOs=True)
parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--no-debug', dest='debug', action='store_false')
parser.set_defaults(debug=True)
parser.add_argument('--plot', dest='plot', action='store_true')
parser.add_argument('--no-plot', dest='plot', action='store_false')
parser.set_defaults(plot=True)
args = parser.parse_args()

MAX_ITERS = args.maxIter
K = args.k
THRESHOLD = 1e-10


def dirichlet_expectation(alpha, k):
    return psi(alpha[k] + np.finfo(np.float32).eps) - psi(np.sum(alpha))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return (e_x + np.finfo(np.float32).eps) / (e_x.sum(axis=0) + np.finfo(np.float32).eps)


def update_lambda_pi(lambda_pi, lambda_phi, alpha_o):
    for k in range(K):
        lambda_pi[k] = alpha_o[k] + np.sum(lambda_phi[:, k])
    return lambda_pi


def update_lambda_beta(lambda_beta, beta_o, Nks):
    for k in range(K):
        lambda_beta[k] = beta_o + Nks[k]
    return lambda_beta


def update_lambda_nu(lambda_nu, nu_o, Nks):
    for k in range(K):
        lambda_nu[k] = nu_o + Nks[k]
    return lambda_nu


def update_lambda_m(lambda_m, lambda_phi, lambda_beta, m_o, beta_o, xn, N):
    for k in range(K):
        aux = np.array([0., 0.])
        for n in range(N):
            aux += lambda_phi[n, k] * xn[n, :]
        lambda_m[k, :] = ((m_o.T * beta_o + aux) / lambda_beta[k]).T
    return lambda_m


def update_lambda_W(lambda_W, lambda_phi, lambda_beta, lambda_m, W_o, beta_o, m_o, xn_xnt, K, N):
    for k in range(K):
        aux = np.array([[0., 0.], [0., 0.]])
        for n in range(N):
            aux += lambda_phi[n, k] * xn_xnt[n]
        lambda_W[k, :, :] = W_o + np.outer(beta_o * m_o, m_o.T) + aux - np.outer(lambda_beta[k] * lambda_m[k, :], lambda_m[k, :].T)
    return lambda_W


def update_lambda_phi(lambda_phi, lambda_pi, lambda_m, lambda_nu, lambda_W, lambda_beta, xn, N, K, D):
    for n in range(N):
        for k in range(K):
            lambda_phi[n, k] = dirichlet_expectation(lambda_pi, k)
            lambda_phi[n, k] += np.dot(lambda_m[k, :], np.dot(lambda_nu[k] * inv(lambda_W[k, :, :]), xn[n, :]))
            lambda_phi[n, k] -= np.trace(np.dot((1 / 2.) * lambda_nu[k] * inv(lambda_W[k, :, :]), np.outer(xn[n, :], xn[n, :])))
            lambda_phi[n, k] -= (D / 2.) * (1 / lambda_beta[k])
            lambda_phi[n, k] -= (1. / 2.) * np.dot(np.dot(lambda_nu[k] * lambda_m[k, :].T, inv(lambda_W[k, :, :])), lambda_m[k, :])
            lambda_phi[n, k] += (D / 2.) * np.log(2.)
            lambda_phi[n, k] += (1 / 2.) * np.sum(psi([((lambda_nu[k] / 2.) + ((1 - i) / 2.)) for i in range(D)]))
            lambda_phi[n, k] -= (1 / 2.) * np.log(det(lambda_W[k, :, :]))
        lambda_phi[n, :] = softmax(lambda_phi[n, :])
    return lambda_phi


def sufficient_statistics_NIW(k, D, lambda_nu, lambda_W, lambda_m, lambda_beta):
    return np.array([
        np.dot(lambda_nu[k] * inv(lambda_W[k, :, :]), lambda_m[k, :]),
        (-1 / 2.) * lambda_nu[k] * inv(lambda_W[k, :, :]),
        (-D / 2.) * (1 / lambda_beta[k]) - (1 / 2.) * lambda_nu[k] * np.dot(np.dot(lambda_m[k, :].T, inv(lambda_W[k, :, :])), lambda_m[k, :]),
        (D / 2.) * np.log(2.) + (1 / 2.) * np.sum(psi([((lambda_nu[k] / 2.) + ((1 - i) / 2.)) for i in range(D)])) - (1 / 2.) * np.log(det(lambda_W[k, :, :]))
    ])


def elbo(N, D, alpha_o, nu_o, beta_o, m_o, W_o, lambda_phi, lambda_pi, lambda_m, lambda_W, lambda_beta, lambda_nu, xn, xn_xnt, Nks):
    elbop = -(((D * (N + 1)) / 2.) * K * np.log(2. * np.pi))
    for k in range(K):
        aux1 = np.array([0., 0.])
        aux2 = np.array([[0., 0.], [0., 0.]])
        for n in range(N):
            aux1 += lambda_phi[n, k] * xn[n, :]
            aux2 += lambda_phi[n, k] * xn_xnt[n]
        elbop = elbop - gammaln(alpha_o[k]) + gammaln(np.sum(alpha_o))
        elbop += (alpha_o[k] - 1 + np.sum(lambda_phi[:, k])) * dirichlet_expectation(alpha_o, k)
        ss_niw = sufficient_statistics_NIW(k, D, lambda_nu, lambda_W, lambda_m, lambda_beta)
        elbop += np.dot((m_o.T * beta_o + aux1).T, ss_niw[0])
        elbop += np.trace(np.dot((W_o + np.outer(beta_o * m_o, m_o.T) + aux2).T, ss_niw[1]))
        elbop += (beta_o + Nks[k]) * ss_niw[2]
        elbop += (nu_o + D + 2. + Nks[k]) * ss_niw[3]
    elbop -= (K * nu_o * D * np.log(2.)) / 2.
    elbop -= K * multigammaln(nu_o / 2., D)
    elbop += (D / 2.) * K * np.log(beta_o)
    elbop += (nu_o / 2.) * K * np.log(det(W_o))

    elboq = -((D / 2.) * K * np.log(2. * np.pi))
    for k in range(K):
        elboq = elboq - gammaln(lambda_pi[k]) + gammaln(np.sum(lambda_pi))
        elboq += (lambda_pi[k] - 1 + np.sum(lambda_phi[:, k])) * dirichlet_expectation(lambda_pi, k)
        ss_niw = sufficient_statistics_NIW(k, D, lambda_nu, lambda_W, lambda_m, lambda_beta)
        elboq += np.dot((lambda_m[k, :].T * lambda_beta[k]).T, ss_niw[0])
        elboq += np.trace(np.dot((lambda_W[k, :, :] + np.outer(lambda_beta[k] * lambda_m[k, :], lambda_m[k, :].T)).T, ss_niw[1]))
        elboq += lambda_beta[k] * ss_niw[2]
        elboq += (lambda_nu[k] + D + 2) * ss_niw[3]
        elboq -= ((lambda_nu[k] * D) / 2.) * np.log(2.)
        elboq -= multigammaln(lambda_nu[k]/2., D)
        elboq += (D / 2.) * np.log(lambda_beta[k])
        elboq += (lambda_nu[k] / 2.) * np.log(det(lambda_W[k, :, :]))
        elboq += np.dot(np.log(lambda_phi[:, k]).T, lambda_phi[:, k])

    print('ELBO: {}'.format(elbop - elboq))
    return elbop - elboq


def plot_iter(ax_spatial, lambda_phi, lambda_m, lambda_W, lambda_nu, xn, N, D, i):
    plt.scatter(xn[:, 0], xn[:, 1],
                c=[np.argmax(lambda_phi[n]) for n in range(N)],
                cmap=cm.gist_rainbow, s=5)
    sctZ = plt.scatter(lambda_m[:, 0], lambda_m[:, 1], color='black', s=5)
    circs = []
    """
    for k in range(K):
        cov = np.power(lambda_W[k, :, :] / (lambda_nu[k] - D - 1), 2)
        print('COV: {}'.format(cov))
        circ = create_cov_ellipse(cov, lambda_m[k, :], color='r', alpha=0.3)
        circs.append(circ)
        ax_spatial.add_artist(circ)
    """
    sctZ.set_offsets(lambda_m)
    plt.draw()
    plt.savefig('img/{}.png'.format(i))


def main():

    # Get data
    with open('{}'.format(args.dataset), 'r') as inputfile:
        data = pkl.load(inputfile)
        xn = data['xn']
        zn = data['zn']
    N, D = xn.shape

    if args.timing:
        init_time = time()

    if args.plot:
        plt.scatter(xn[:, 0], xn[:, 1], c=zn, cmap=cm.gist_rainbow, s=5)
        plt.show()

    # Priors
    alpha_o = np.array([1.0] * K)
    nu_o = np.array([3.0])
    W_o = np.array([[20., 30.], [25., 40.]])
    m_o = np.array([0.0, 0.0])
    beta_o = np.array([0.7])

    # Variational parameters
    lambda_phi = np.random.dirichlet(alpha_o, N)
    lambda_pi = np.zeros(shape=K)
    lambda_beta = np.zeros(shape=K)
    lambda_nu = np.zeros(shape=K)
    lambda_m = np.zeros(shape=(K, D))
    lambda_W = np.zeros(shape=(K, D, D))

    xn_xnt = [np.outer(xn[n, :], xn[n, :].T) for n in range(N)]

    # Plot configs
    # plt.ion()
    fig = plt.figure(figsize=(10, 10))
    ax_spatial = fig.add_subplot(1, 1, 1)
    circs = []

    # Inference
    lbs = []
    n_iters = 0
    for i in range(MAX_ITERS):
        print('\n******* ITERATION {} *******'.format(i))

        # Variational parameter updates
        lambda_pi = update_lambda_pi(lambda_pi, lambda_phi, alpha_o)
        Nks = np.sum(lambda_phi, axis=0)
        lambda_beta = update_lambda_beta(lambda_beta, beta_o, Nks)
        lambda_nu = update_lambda_nu(lambda_nu, nu_o, Nks)
        lambda_m = update_lambda_m(lambda_m, lambda_phi, lambda_beta, m_o, beta_o, xn, N)
        lambda_W = update_lambda_W(lambda_W, lambda_phi, lambda_beta, lambda_m, W_o, beta_o, m_o, xn_xnt, K, N)
        lambda_phi = update_lambda_phi(lambda_phi, lambda_pi, lambda_m, lambda_nu, lambda_W, lambda_beta, xn, N, K, D)

        print('lambda_pi: {}'.format(lambda_pi))
        print('lambda_beta: {}'.format(lambda_beta))
        print('lambda_nu: {}'.format(lambda_nu))
        print('lambda_m: {}'.format(lambda_m))
        print('lambda_W: {}'.format(lambda_W))
        print('lambda_phi: {}'.format(lambda_phi[0:9, :]))

        # ELBO computation
        lb = elbo(N, D, alpha_o, nu_o, beta_o, m_o, W_o, lambda_phi, lambda_pi, lambda_m, lambda_W, lambda_beta, lambda_nu, xn, xn_xnt, Nks)

        # Break condition
        if i > 0 and  abs(lb - lbs[i - 1]) < THRESHOLD:
            break
        lbs.append(lb)
        n_iters += 1

        # plot_iter(ax_spatial, lambda_phi, lambda_m, lambda_W, lambda_nu, xn, N, D, i)

    print('\n******* RESULTS *******')
    for k in range(K):
        print('Mu k{}: {}'.format(k, lambda_m[k, :]))
        print('SD k{}: {}'.format(k, np.sqrt(lambda_W[k, :, :] / (lambda_nu[k] - D - 1))))

    if args.timing:
        final_time = time()
        exec_time = final_time - init_time
        print('Time: {} seconds'.format(exec_time))
    if args.getNIter:
        print('Iterations: {}'.format(n_iters))
    if args.getELBOs:
        print('ELBOs: {}'.format(lbs))
        plt.scatter(range(n_iters), lbs, cmap=cm.gist_rainbow, s=5)
        plt.show()
    if args.plot:
        plt.scatter(xn[:, 0], xn[:, 1], c=[np.argmax(lambda_phi[n]) for n in range(N)], cmap=cm.gist_rainbow, s=5)
        plt.show()


if __name__ == '__main__': main()
