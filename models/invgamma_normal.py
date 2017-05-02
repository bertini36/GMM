# -*- coding: UTF-8 -*-

"""
InvGamma-Normal Model
Posterior exact inference
"""

import numpy as np
from scipy.stats import invgamma

N = 1000

# Data generation (known mean)
mu = 5
sigma = invgamma.rvs(1, scale=0.7)
x = np.random.normal(mu, sigma, N)
print('sigma={}'.format(sigma))

# Prior definition
alpha = 1
beta = 0.6

# Posterior computation
alpha_pos = alpha + (N / 2)
beta_pos = beta + (np.sum(np.power(x - mu, 2)) / 2)
sigma_pos = np.sqrt(invgamma.rvs(alpha_pos, scale=beta_pos))
print('Inferred sigma={}'.format(sigma_pos))
