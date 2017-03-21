# -*- coding: UTF-8 -*-

"""
Mixture of gaussians data generator
"""

import math
import argparse
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wishart

if __name__ == '__main__':
    d = 2
    parser = argparse.ArgumentParser(description='Generate synthetic gaussian mixture data')
    parser.add_argument('-N', metavar='N', type=int, default=100)
    parser.add_argument('-K', metavar='K', type=int, default=5)
    parser.add_argument('-alpha', metavar='alpha', type=float, default=1.)
    parser.add_argument('-m_mu', metavar='m_mu', nargs='+', type=float, default=[0., 0.])
    parser.add_argument('-beta_mu', metavar='beta_mu', type=float, default=0.01)
    parser.add_argument('-w_delta', metavar='w_delta', nargs='+', type=float, default=[1., 0., 0., 1.])
    parser.add_argument('-nu_delta', metavar='nu_delta', type=float, default=2.)
    parser.add_argument('-filename', metavar='filename', type=str, default='data_k2_50.pkl')
    args = parser.parse_args()
    
    N = args.N 
    K = args.K

    alpha = [args.alpha]*K
    pi = np.random.dirichlet(alpha)

    m_mu = np.array(args.m_mu)      
    beta_mu = args.beta_mu

    nu_Delta = args.nu_delta
    W_Delta = np.array([args.w_delta[0:d],args.w_delta[d:2*d]]) 
   
    xn = np.zeros((N, d))
    muk = np.zeros((K, d))
    Deltak = np.zeros((K, d, d))
    zn = np.zeros(N).astype(int)

    for k in range(K):
        Deltak[k, :, :] = wishart.rvs(nu_Delta, W_Delta)
        muk[k, :] = np.random.multivariate_normal(
            m_mu, np.linalg.inv(beta_mu*Deltak[k, :, :]))

    for n in range(N):
        zn[n] = np.random.choice(K, 1, p=pi)[0].astype(int)
        xn[n, :] = np.random.multivariate_normal(
            muk[zn[n], :], np.linalg.inv(Deltak[zn[n], :, :]))

    with open(args.filename, 'w') as output:
        pkl.dump({'zn': zn, 'xn': xn}, output)

    plt.scatter(xn[:, 0], xn[:, 1], c=(1.*zn)/K)
    plt.show()
