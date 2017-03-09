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
from scipy.special import gammaln, psi
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
parser.set_defaults(getELBOs=False)
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


def getNks(x):
    Nks = np.zeros(shape=K)
    for i in xrange(len(x)):
        Nks[np.random.choice(K, 1, p=x[i])] += 1
    return Nks


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def elbo(N, D, alpha_o, lambda_phi):
    e1 = np.zeros(shape=K)
    for k in range(K):
        e1[k] = alpha_o[k] + np.sum(lambda_phi[:, k])
    e2 = psi(alpha_o) - psi(np.sum(alpha_o))
    e3 = np.dot(e1.T, e2)
    e3 -= np.sum(gammaln(alpha_o) + gammaln(np.sum(alpha_o)))
    e3 -= ((D * (N + 1)) / 2) * K * np.log(2 * np.pi)


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

    # Model hyperparameters (priors)
    alpha_o = np.array([1.0] * K)
    nu_o = np.array([3.0])
    W_o = np.array([[20., 30.], [25., 40.]])
    m_o = np.array([0.0, 0.0])
    beta_o = np.array([0.8])

    lambda_phi = np.random.dirichlet(alpha_o, N)
    # lambda_phi = np.array([[1, 0] if zn[n] == 0 else [0, 1] for n in range(N)])
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

        # Parameter updates
        for k in range(K):
            lambda_pi[k] = alpha_o[k] + np.sum(lambda_phi[:, k])

        Nks = getNks(lambda_phi)
        for k in range(K):
            lambda_beta[k] = beta_o + Nks[k]

        for k in range(K):
            lambda_nu[k] = nu_o + Nks[k]

        for k in range(K):
            aux = np.array([0., 0.])
            for n in range(N):
                aux += lambda_phi[n, k] * xn[n, :]
            lambda_m[k, :] = ((m_o.T * beta_o + aux) / lambda_beta[k]).T

        for k in range(K):
            aux = np.array([[0., 0.], [0., 0.]])
            for n in range(N):
                aux += lambda_phi[n, k] * xn_xnt[n]
            lambda_W[k, :, :] = W_o + np.outer(m_o, m_o.T) + aux - lambda_beta[k] * np.outer(lambda_m[k, :], lambda_m[k, :].T)

        # PROBLEMA: Cuando el determinante de lambda_W[k] da negativo. No debería
        #           ocurrir ya que lambda_W[k] es una matriz definida positiva
        for k in range(K):
            print('Determinante lambda_W[{}]: {}'.format(k, np.linalg.det(lambda_W[k, :, :])))

        for n in range(N):
            for k in range(K):
                lambda_phi[n, k] = psi(lambda_pi[k]) - psi(np.sum(lambda_pi))
                lambda_phi[n, k] += np.dot(np.dot(lambda_nu[k] * inv(lambda_W[k, :, :]), lambda_m[k, :]).T, xn[n, :])
                lambda_phi[n, k] -= np.dot(np.dot((1 / 2.) * lambda_nu[k] * inv(lambda_W[k, :, :]), xn[n, :]), xn[n, :].T)
                lambda_phi[n, k] -= (1 / 2.) * (1 / lambda_beta[k])
                lambda_phi[n, k] -= np.dot(np.dot(lambda_nu[k] * lambda_m[k, :].T, inv(lambda_W[k, :, :])), lambda_m[k, :])
                lambda_phi[n, k] += (D / 2.) * np.log(2.)
                lambda_phi[n, k] += (1 / 2.) * np.sum(psi([((lambda_nu[k] / 2.) + ((lambda_nu[k] - i) / 2.)) for i in range(D)]))
                lambda_phi[n, k] -= (1 / 2.) * np.log(det(lambda_W[k, :, :]))
            # print('Antes: {}'.format(lambda_phi[n, :]))
            lambda_phi[n, :] = softmax(lambda_phi[n, :])
            # print('Despues: {}'.format(lambda_phi[n, :]))

        print('lambda_phi: {}'.format(lambda_phi[0:9, :]))
        print('lambda_beta: {}'.format(lambda_beta))
        print('lambda_nu: {}'.format(lambda_nu))
        print('lambda_m: {}'.format(lambda_m))
        print('lambda_W: {}'.format(lambda_W))
        print('lambda_pi: {}'.format(lambda_pi))

        """
        # ELBO computation
        lb = elbo()
        # Break condition
        if i > 0:
            if abs(lb - lbs[i - 1]) < THRESHOLD:
                if args.getNIter:
                    n_iters = i + 1
                break
        lbs.append(lb)
        """

    print('\n******* RESULTS *******')
    for k in range(K):
        print('Mu k{}: {}'.format(k, lambda_m[k, :]))
        print('Sigma k{}: {}'.format(k, invwishart.rvs(lambda_nu[k], lambda_W[k, :, :]) * lambda_beta[k]))

    # print('lambda_phi: {}'.format(lambda_phi))

    if args.plot:
        plt.scatter(xn[:, 0], xn[:, 1], c=[np.argmax(lambda_phi[n]) for n in range(N)], cmap=cm.gist_rainbow, s=5)
        plt.show()

    """
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
