# -*- coding: UTF-8 -*-

"""
Dirichlet-Categorical model
Posterior exact inference
"""

import numpy as np

N = 1000
K = 4

# Data generation
alpha = np.array([20., 30., 10., 10.])
pi = np.random.dirichlet(alpha)
z = np.array([np.random.choice(K, 1, p=pi)[0] for n in xrange(N)])
print('pi={}'.format(pi))

# Prior definition
alpha_prior = np.array([1., 1., 1., 1.])

# Posterior computation

# Etas computation (with priors)
etas = alpha_prior - 1.

# Posterior hyperparameters computation
alpha_pos = etas
for n in xrange(N):
    alpha_pos[z[n]] += 1
pi_pos = np.random.dirichlet(alpha_pos)
print('Inferred pi={}'.format(pi_pos))
