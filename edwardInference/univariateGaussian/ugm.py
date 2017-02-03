# -*- coding: UTF-8 -*-

"""
Gradient Ascent Variational Inference
process to approximate an univariate gaussian
"""

import edward as ed
import numpy as np
import tensorflow as tf
from edward.models import Gamma, Normal

N = 100

ed.set_seed(42)

# Model hyperparameters
m = [0.]
beta = [0.0001]
a = [0.001]
b = [0.001]

# Probabilistic model
mu = Normal(mu=m, sigma=beta)
sigma = Gamma(alpha=a, beta=b)
x = Normal(mu=ed.dot(tf.ones([N, 1]), mu), sigma=ed.dot(tf.ones([N, 1]), sigma))

# Variational model definition
lambda_mu_m = tf.Variable(tf.random_normal([1]))
lambda_mu_beta = tf.nn.softplus(tf.Variable(tf.zeros([1])))
lambda_a = tf.Variable(tf.random_gamma([1], 1))
lambda_b = tf.Variable(tf.random_gamma([1], 1))
qmu = Normal(mu=lambda_mu_m, sigma=lambda_mu_beta)
qsigma = Gamma(alpha=lambda_a, beta=lambda_b)

# Data generation
xn = np.random.normal(7, 1, N)

print('mu=7')
print('sigma=1')

# Inference
inference = ed.KLqp({mu: qmu, sigma: qsigma}, data={x: xn})
inference.run(n_samples=30, n_iter=40000)

sess = ed.get_session()

print('Inferred mu={}'.format(sess.run(qmu.value())))
print('Inferred sigma={}'.format(sess.run(qsigma.value())))
