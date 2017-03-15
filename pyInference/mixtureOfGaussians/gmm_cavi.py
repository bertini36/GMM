# -*- coding: UTF-8 -*-

"""
Coordinate Ascent Variational Inference
process to approximate a mixture of gaussians
[DOING]
"""

import argparse
import pickle as pkl
from time import time

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import det, inv
from scipy.special import gammaln, multigammaln, psi
from scipy.stats import invwishart

parser = argparse.ArgumentParser(description='CAVI in mixture of gaussians')
parser.add_argument('-maxIter', metavar='maxIter', type=int, default=10)
parser.add_argument('-dataset', metavar='dataset',
                    type=str, default='../../data/data_k2_1000.pkl')
parser.add_argument('-k', metavar='k', type=int, default=2)
parser.add_argument('--timing', dest='timing', action='store_true')
parser.add_argument('--no-timing', dest='timing', action='store_false')
parser.set_defaults(timing=False)
parser.add_argument('--getNIter', dest='getNIter', action='store_true')
parser.add_argument('--no-getNIter', dest='getNIter', action='store_false')
parser.set_defaults(getNIter=False)
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
THRESHOLD = 1e-6


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return (e_x + np.finfo(np.float32).eps) / (e_x.sum(axis=0) + np.finfo(np.float32).eps)


def elbo(N, D, alpha_o, nu_o, beta_o, m_o, W_o, lambda_phi, lambda_pi, lambda_m, lambda_W, lambda_beta, lambda_nu, Nks, aux1, aux2):
    elbop = -(((D * (N + 1)) / 2.) * K * np.log(2. * np.pi))
    for k in range(K):
        elbop -= gammaln(alpha_o[k]) + gammaln(np.sum(alpha_o))
        elbop += (alpha_o[k] - 1 + np.sum(lambda_phi[:, k])) * (psi(alpha_o[k]) - psi(np.sum(alpha_o)))
        elbop += np.dot((m_o.T * beta_o + aux1).T, np.dot(lambda_nu[k] * inv(lambda_W[k, :, :]), lambda_m[k, :]))
        elbop += np.trace(np.dot((W_o + np.outer(beta_o * m_o, m_o.T) + aux2).T, (-1 / 2.) * lambda_nu[k] * inv(lambda_W[k, :, :])))
        elbop += (beta_o + Nks[k]) * ((-1 / 2.) * (1 / lambda_beta[k]) - lambda_nu[k] * np.dot(np.dot(lambda_m[k, :].T, inv(lambda_W[k, :, :])), lambda_m[k, :]))
        elbop += (nu_o + D + 2. + Nks[k]) * ((D / 2.) * np.log(2.) + (1 / 2.) * np.sum(psi([((lambda_nu[k] / 2.) + ((1 - i) / 2.)) for i in range(D)])) - (1 / 2.) * np.log(det(lambda_W[k, :, :])))
    elbop -= (K * nu_o * D * np.log(2.)) / 2.
    elbop -= K * multigammaln(nu_o / 2., D)
    elbop += (D / 2.) * K * np.log(beta_o)
    elbop += (nu_o / 2.) * K * np.log(det(W_o))

    elboq = -((D / 2.) * K * np.log(2. * np.pi))
    for k in range(K):
        elboq -= gammaln(lambda_pi[k]) + gammaln(np.sum(lambda_pi))
        elboq += (lambda_pi[k] - 1 + np.sum(lambda_phi[:, k])) * (psi(lambda_pi[k]) - psi(np.sum(lambda_pi)))
        elboq += np.dot((lambda_m[k, :].T * lambda_beta[k]).T, np.dot(lambda_nu[k] * inv(lambda_W[k, :, :]), lambda_m[k, :]))
        elboq += np.trace(np.dot((lambda_W[k, :, :] + np.outer(lambda_beta[k] * lambda_m[k, :], lambda_m[k, :].T)).T, (-1 / 2.) * lambda_nu[k] * inv(lambda_W[k, :, :])))
        elboq += lambda_beta[k] * ((-1 / 2.) * (1 / lambda_beta[k]) - lambda_nu[k] * np.dot(np.dot(lambda_m[k, :].T, inv(lambda_W[k, :, :])), lambda_m[k, :]))
        elboq += (lambda_nu[k] + D + 2) * ((D / 2.) * np.log(2.) + (1 / 2.) * np.sum(psi([((lambda_nu[k] / 2.) + ((lambda_nu[k] - i) / 2.)) for i in range(D)])) - (1 / 2.) * np.log(det(lambda_W[k, :, :])))
        elboq -= ((lambda_nu[k] * D) / 2.) * np.log(2.)
        elboq -= multigammaln(lambda_nu[k]/2., D)
        elboq += (D / 2.) * np.log(lambda_beta[k])
        elboq += (lambda_nu[k] / 2.) * np.log(det(lambda_W[k, :, :]))
        elboq += np.dot(np.log(lambda_phi[:, k]).T, lambda_phi[:, k])

    return elbop - elboq


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
        plt.scatter(xn[:, 0], xn[:, 1], c=data['zn'], cmap=cm.gist_rainbow, s=5)
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

    # Inference
    lbs = []
    for i in range(MAX_ITERS):
        print('\n******* ITERATION {} *******'.format(i))

        # Variational parameter updates
        for k in range(K):
            lambda_pi[k] = alpha_o[k] + np.sum(lambda_phi[:, k])

        Nks = np.sum(lambda_phi, axis=0)
        for k in range(K):
            lambda_beta[k] = beta_o + Nks[k]

        for k in range(K):
            lambda_nu[k] = nu_o + Nks[k]

        for k in range(K):
            aux1 = np.array([0., 0.])
            for n in range(N):
                aux1 += lambda_phi[n, k] * xn[n, :]
            lambda_m[k, :] = ((m_o.T * beta_o + aux1) / lambda_beta[k]).T

        for k in range(K):
            aux2 = np.array([[0., 0.], [0., 0.]])
            for n in range(N):
                aux2 += lambda_phi[n, k] * xn_xnt[n]
            lambda_W[k, :, :] = W_o + np.outer(beta_o * m_o, m_o.T) + aux2 - np.outer(lambda_beta[k] * lambda_m[k, :], lambda_m[k, :].T)

        for n in range(N):
            for k in range(K):
                lambda_phi[n, k] = psi(lambda_pi[k]) - psi(np.sum(lambda_pi))
                lambda_phi[n, k] += np.dot(np.dot(lambda_nu[k] * inv(lambda_W[k, :, :]), lambda_m[k, :]).T, xn[n, :])
                lambda_phi[n, k] -= np.dot(np.dot((1 / 2.) * lambda_nu[k] * inv(lambda_W[k, :, :]), xn[n, :]).T, xn[n, :])
                lambda_phi[n, k] -= (1 / 2.) * (1 / lambda_beta[k])
                lambda_phi[n, k] -= np.dot(np.dot(lambda_nu[k] * lambda_m[k, :].T, inv(lambda_W[k, :, :])), lambda_m[k, :])
                lambda_phi[n, k] += (D / 2.) * np.log(2.)
                lambda_phi[n, k] += (1 / 2.) * np.sum(psi([((lambda_nu[k] / 2.) + ((lambda_nu[k] - i) / 2.)) for i in range(D)]))
                lambda_phi[n, k] -= (1 / 2.) * np.log(det(lambda_W[k, :, :]))
            lambda_phi[n, :] = softmax(lambda_phi[n, :])

        print('lambda_phi: {}'.format(lambda_phi[0:9, :]))
        print('lambda_beta: {}'.format(lambda_beta))
        print('lambda_nu: {}'.format(lambda_nu))
        print('lambda_m: {}'.format(lambda_m))
        print('lambda_W: {}'.format(lambda_W))
        print('lambda_pi: {}'.format(lambda_pi))

        # ELBO computation
        lb = elbo(N, D, alpha_o, nu_o, beta_o, m_o, W_o, lambda_phi, lambda_pi, lambda_m, lambda_W, lambda_beta, lambda_nu, Nks, aux1, aux2)

        # Break condition
        if i > 1:
            if abs(lb - lbs[i - 1]) < THRESHOLD:
                if args.getNIter:
                    n_iters = i + 1
                break
        lbs.append(lb)

    print('\n******* RESULTS *******')
    for k in range(K):
        print('Mu k{}: {}'.format(k, lambda_m[k, :]))
        # print('Sigma k{}: {}'.format(k, invwishart.rvs(lambda_nu[k], lambda_W[k, :, :]) * lambda_beta[k]))
        print('SD k{}: {}'.format(k, np.sqrt(lambda_W[k, :, :] / (lambda_nu[k] - D - 1))))

    if args.timing:
        final_time = time()
        exec_time = final_time - init_time
        print('Time: {} seconds'.format(exec_time))
    if args.getNIter:
        print('Iterations: {}'.format(n_iters))
    if args.getELBOs:
        print('ELBOs: {}'.format(lbs))
    if args.plot:
        plt.scatter(xn[:, 0], xn[:, 1],
                    c=[np.argmax(lambda_phi[n]) for n in range(N)],
                    cmap=cm.gist_rainbow, s=5)
        plt.show()


if __name__ == '__main__': main()
