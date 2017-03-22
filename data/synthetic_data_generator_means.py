# -*- coding: UTF-8 -*-

"""
Mixture of gaussians data generator with common variance for all classes
"""

import argparse
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

"""
Parameters:
    * N: Number of data points to generate
    * K: Number of clusters
    * D: Data dimensions
    * filename: File where the data will be saved
    * alpha_o: Dirichlet prior
    * m_o, beta_o: Normal priors
    * w_o, nu_o: Wishart priors
"""

parser = argparse.ArgumentParser(
    description='Generate synthetic gaussian mixture data')
parser.add_argument('-N', metavar='N', type=int, default=1000)
parser.add_argument('-K', metavar='K', type=int, default=2)
parser.add_argument('-D', metavar='D', type=int, default=3)
parser.add_argument('-filename', metavar='filename',
                    type=str, default='data_means.pkl')

# Priors
parser.add_argument('-alpha_o', metavar='alpha_o', type=float, default=1.)
parser.add_argument('-m_o', metavar='m_o',
                    nargs='+', type=float, default=[0., 0.])
parser.add_argument('-beta_o', metavar='beta_o', type=float, default=0.01)
parser.add_argument('-delta_o', metavar='delta_o',
                    nargs='+', type=float, default=[1., 0., 0., 1.])
args = parser.parse_args()

N = args.N
K = args.K
D = args.D


def main():
    alpha_o = [args.alpha_o] * K
    pi = np.random.dirichlet(alpha_o)

    if D == 2:
        m_o = np.array(args.m_o)
    elif D == 3:
        m_o = [0., 0., 0.]
    elif D == 5:
        m_o = [0., 0., 0., 0., 0.]
    elif D == 7:
        m_o = [0., 0., 0., 0., 0., 0., 0.]
    beta_o = args.beta_o

    if D == 2:
        delta_o = np.array([args.delta_o[0:D], args.delta_o[D:2 * D]])
    elif D == 3:
        delta_o = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    elif D == 5:
        delta_o = np.array([[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.],
                            [0., 0., 1., 0., 0.], [0., 0., 0., 1., 0.],
                            [0., 0., 0., 0., 1.]])
    elif D == 7:
        delta_o = np.array([[1., 0., 0., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0., 0., 0.],
                            [0., 0., 1., 0., 0., 0., 0.],
                            [0., 0., 0., 1., 0., 0., 0.],
                            [0., 0., 0., 0., 1., 0., 0.],
                            [0., 0., 0., 0., 0., 1., 0.],
                            [0., 0., 0., 0., 0., 0., 1.]])

    xn = np.zeros((N, D))
    muk = np.zeros((K, D))
    zn = np.zeros(N).astype(int)

    for k in range(K):
        muk[k, :] = np.random.multivariate_normal(
            m_o, np.linalg.inv(beta_o * delta_o))

    for n in range(N):
        zn[n] = np.random.choice(K, 1, p=pi)[0].astype(int)
        xn[n, :] = np.random.multivariate_normal(muk[zn[n], :],
                                                 np.linalg.inv(delta_o))

    with open(args.filename, 'w') as output:
        pkl.dump({'zn': zn, 'xn': xn}, output)

    if D == 2:
        plt.scatter(xn[:, 0], xn[:, 1], c=(1. * zn) / K)
        plt.show()
    elif D == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xn[:, 0], xn[:, 1], xn[:, 2], c=(1. * zn) / K)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()


if __name__ == '__main__': main()
