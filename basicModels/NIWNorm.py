# -*- coding: UTF-8 -*-

"""
NormalInverseWishart-Normal Model
"""

import numpy as np
from scipy.stats import invwishart
import matplotlib.cm as cm
import matplotlib.pyplot as plt

N = 1000
D = 2

# Data generation

# NIW Inverse Wishart hyperparameters
v = 3.
W = np.array([[20., 30.], [25., 40.]])
sigma = invwishart.rvs(v, W)

# NIW Normal hyperparameters
m = np.array([1., 1.])
k = 0.8
mu = np.random.multivariate_normal(m, sigma/k)

print('mu={}'.format(mu))
print('sigma={}'.format(sigma))

x = np.random.multivariate_normal(mu, sigma, N)
plt.scatter(x[:, 0], x[:, 1], cmap=cm.gist_rainbow, s=5)
plt.show()

# Prior definition
v_prior = 3.
W_prior = np.array([[1., 0.], [0., 1.]])
m_prior = np.array([0.5, 0.5])
k_prior = 0.6

# Posterior computation
k_pos = k_prior + N
v_pos = v_prior + N
m_pos = (((m_prior.T * k_prior) + np.sum(x, axis=0)) / k_pos).T
W_pos = W_prior + np.outer(k_prior * m_prior, m_prior) + \
        np.sum(np.array([np.outer(x[n, :], x[n, :]) for n in xrange(N)]),
               axis=0) - np.outer(k_pos * m_pos, m_pos)

# Posterior sampling
sigma_pos = invwishart.rvs(v_pos, W_pos)
mu_pos = np.random.multivariate_normal(m_pos, sigma_pos/k_pos)

print('Inferred mu={}'.format(mu_pos))
print('Inferred sigma={}'.format(sigma_pos))
