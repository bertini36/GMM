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
from scipy.special import psi

parser = argparse.ArgumentParser(description='CAVI in mixture of gaussians')
parser.add_argument('-maxIter', metavar='maxIter', type=int, default=100)
parser.add_argument('-dataset', metavar='dataset',
                    type=str, default='../../data/data_k2_100.pkl')
parser.add_argument('-k', metavar='k', type=int, default=2)
parser.add_argument('--timing', dest='timing', action='store_true')
parser.add_argument('--no-timing', dest='timing', action='store_false')
parser.set_defaults(timing=False)
parser.add_argument('--getNIter', dest='getNIter', action='store_true')
parser.add_argument('--no-getNIter', dest='getNIter', action='store_false')
parser.set_defaults(getNIter=False)
parser.add_argument('--getELBOs', dest='getELBOs', action='store_true')
parser.add_argument('--no-getELBOs', dest='getELBOs', action='store_false')
parser.set_defaults(getELBOs=False)
parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--no-debug', dest='debug', action='store_false')
parser.set_defaults(debug=True)
parser.add_argument('--plot', dest='plot', action='store_true')
parser.add_argument('--no-plot', dest='plot', action='store_false')
parser.set_defaults(plot=False)
args = parser.parse_args()

MAX_ITERS = args.maxIter
K = args.k
THRESHOLD = 1e-6


def elbo():
    return 100


def getNs(lambda_phi):
    ns = np.array([0] * K)
    for i in xrange(len(lambda_phi)):
        ns[np.random.choice(K, 1, p=lambda_phi[i])] += 1
    return np.array(ns)


def main():
    # Get data
    with open('{}'.format(args.dataset), 'r') as inputfile:
        data = pkl.load(inputfile)
        xn = data['xn']
    N, D = xn.shape

    if args.timing:
        init_time = time()

    if args.plot:
        plt.scatter(xn[:, 0], xn[:, 1], c=(1. * data['zn']) / max(data['zn']),
                    cmap=cm.gist_rainbow, s=5)
        plt.show()

    # Model hyperparameters (priors)
    alpha_o = np.array([1.0] * K)
    nu_o = np.array([3.0])
    W_o = np.array([[20., 30.], [25., 40.]])
    m_o = np.array([0.0, 0.0])
    beta_o = np.array([0.8])

    # Initializations
    # Shape lambda_phi: (N, K)
    lambda_phi = np.random.dirichlet(alpha_o, N)
    print('lambda_phi: {}'.format(lambda_phi))
    print('Shape lambda_phi: {}'.format(lambda_phi.shape))

    # Shape lambda_pi: (K)
    lambda_pi = alpha_o + np.sum(lambda_phi, axis=0)
    print('lambda_pi: {}'.format(lambda_pi))
    print('Shape lambda_pi: {}'.format(lambda_pi.shape))

    # Shape lambda_beta: (K)
    ns = getNs(lambda_phi)
    lambda_beta = np.zeros(shape=K)
    for k in range(K):
        lambda_beta[k] = beta_o + ns[k]
    print('lambda_beta: {}'.format(lambda_beta))
    print('Shape lambda_beta: {}'.format(lambda_beta.shape))

    # Shape lambda_nu: (K)
    lambda_nu = np.zeros(shape=K)
    for k in range(K):
        lambda_nu[k] = nu_o + ns[k]
    print('lambda_nu: {}'.format(lambda_nu))
    print('Shape lambda_nu: {}'.format(lambda_nu.shape))

    # Shape lambda_m: (K, D)
    lambda_m = np.zeros(shape=(K, D))
    for k in range(K):
        lambda_m[k] = (np.outer(m_o.T, beta_o)
                       + np.sum(np.dot(lambda_phi.T, xn), axis=0))[k] \
                      / lambda_beta[k]
    print('lambda_m: {}'.format(lambda_m))
    print('Shape lambda_m: {}'.format(lambda_m.shape))

    # Shape lambda_W: (K, D, D)
    lambda_W = np.zeros(shape=(K, D, D))
    for k in range(K):
        lambda_W[k] = W_o + np.outer(m_o, m_o.T) + \
                      np.sum(np.dot(np.dot(lambda_phi.T, xn), xn.T), axis=0)[k]\
                      - lambda_beta[k] * np.outer(lambda_m[k], lambda_m[k].T)
    print('lambda_W: {}'.format(lambda_W))
    print('Shape lambda_W: {}'.format(lambda_W.shape))

    """
    lbs = []
    for i in xrange(MAX_ITERS):

        # Parameter updates
        for n in xrange(N):
            for k in xrange(K):
                lambda_phi[n, k] = psi(lambda_pi[k]) - np.sum(psi(lambda_pi))
                lambda_phi[n, k] += np.dot(np.dot(lambda_nu[k] * np.linalg.inv(lambda_W[:, :, k]), lambda_m[:, k]), xn[n, :])
                lambda_phi[n, k] -= np.dot(np.dot(1 / 2. * lambda_nu[k] * np.linalg.inv(lambda_W[:, :, k]), xn[n, :]), xn[n, :].T)
                lambda_phi[n, k] -= 1 / 2 * np.power(lambda_beta[k], -1)
                lambda_phi[n, k] -= np.dot(np.dot(lambda_nu[k] * lambda_m[:, k].T, np.linalg.inv(lambda_W[:, :, k])), lambda_m[:, k])
                lambda_phi[n, k] += D / 2 * np.log(2)
                lambda_phi[n, k] += 1 / 2 * np.sum(psi([((lambda_nu[k] / 2) + ((1 - i) / 2)) for i in xrange(D)]))
                lambda_phi[n, k] -= 1 / 2 * np.log(np.linalg.det(lambda_W[:, :, k])) * np.log(np.sum(lambda_phi, axis=0)[k])

        lambda_pi = alpha_o + np.sum(lambda_phi, axis=0)

        ns = getNs(lambda_phi)
        for k in range(K):
            lambda_beta[k] = beta_o + ns[k]
            lambda_nu[k] = nu_o + ns[k]
            lambda_m[k] = (np.outer(m_o.T, beta_o)
                           + np.sum(np.dot(lambda_phi.T, xn), axis=0))[k] \
                          / lambda_beta[k]
            lambda_W[k] = W_o + np.outer(m_o, m_o.T) + \
                          np.sum(np.dot(np.dot(lambda_phi.T, xn), xn.T),
                                 axis=0)[k] - np.outer(lambda_beta[k], np.outer(lambda_m[k], lambda_m[k]))

        # ELBO computation
        lb = elbo()

        # Break condition
        if i > 0:
            if abs(lb - lbs[i - 1]) < THRESHOLD:
                if args.getNIter:
                    n_iters = i + 1
                break
        lbs.append(lb)

    if args.plot:
        pass

    if args.timing:
        final_time = time()
        exec_time = final_time - init_time
        print('Time: {} seconds'.format(exec_time))

    if args.getNIter:
        print('Iterations: {}'.format(n_iters))

    if args.getELBOs:
        print('ELBOs: {}'.format(lbs))
    """

if __name__ == '__main__': main()
