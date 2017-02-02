# -*- coding: UTF-8 -*-

"""
mixtureOfGaussians data generator with common variance for all classes
"""

import math
import argparse
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    D = 2
    parser = argparse.ArgumentParser(description='Generate synthetic gaussian mixture data')
    parser.add_argument('-N', metavar='N', type=int, default=100)
    parser.add_argument('-K', metavar='K', type=int, default=5)
    parser.add_argument('-alpha', metavar='alpha', type=float, default=1.)
    parser.add_argument('-m_o', metavar='m_mu', nargs='+', type=float, default=[0., 0.])
    parser.add_argument('-beta_o', metavar='beta_mu', type=float, default=0.01)
    parser.add_argument('-Delta_o', metavar='w_delta', nargs='+', type=float, default=[1., 0., 0., 1.])
    parser.add_argument('-filename', metavar='filename', type=str, default='data_means.pkl')
    args = parser.parse_args()
    
    N = args.N 
    K = args.K

    alpha = [args.alpha]*K
    pi = np.random.dirichlet(alpha)

    m_o = np.array(args.m_o)      
    beta_o = args.beta_o
    Delta_o = np.array([args.Delta_o[0:D],args.Delta_o[D:2*D]]) 
   
    xn = np.zeros((N, D))
    muk = np.zeros((K, D))
    Deltak = np.zeros((K, D, D))
    zn = np.zeros(N).astype(int)

    for k in xrange(K):
        muk[k, :] = np.random.multivariate_normal(m_o, np.linalg.inv(beta_o*Delta_o))

    for n in xrange(N):
        zn[n] = np.random.choice(K, 1, p=pi)[0].astype(int)
        xn[n, :] = np.random.multivariate_normal(muk[zn[n], :], np.linalg.inv(Delta_o))

    with open(args.filename, 'w') as output:
        pkl.dump({'zn':zn, 'xn':xn}, output)

    plt.scatter(xn[:,0],xn[:,1], c=(1.*zn)/K)
    plt.show()

    