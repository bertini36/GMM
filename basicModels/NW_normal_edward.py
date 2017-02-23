# -*- coding: UTF-8 -*-

"""
NormalWishart-Normal Model
Posterior inference with Edward BBVI
[DOING]
"""

import edward as ed
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from edward.models import MultivariateNormalFull, WishartCholesky
from scipy.stats import invwishart

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
mu = np.random.multivariate_normal(m, sigma / k)
xn_data = np.random.multivariate_normal(mu, sigma, N)
plt.scatter(xn_data[:, 0], xn_data[:, 1], cmap=cm.gist_rainbow, s=5)
plt.show()
print('mu={}'.format(mu))
print('sigma={}'.format(sigma))

# Prior definition
v_prior = tf.Variable(3., dtype=tf.float64, trainable=False)
W_prior = tf.Variable(np.array([[1., 0.], [0., 1.]]),
                      dtype=tf.float64, trainable=False)
m_prior = tf.Variable(np.array([0.5, 0.5]), dtype=tf.float64, trainable=False)
k_prior = tf.Variable(0.6, dtype=tf.float64, trainable=False)

print('***** PRIORS *****')
print('v_prior: {}'.format(v_prior))
print('W_prior: {}'.format(W_prior))
print('m_prior: {}'.format(m_prior))
print('k_prior: {}'.format(k_prior))

# Posterior inference
# Probabilistic model
sigma = WishartCholesky(df=v_prior, scale=W_prior)
mu = MultivariateNormalFull(m_prior, k_prior * sigma)
xn = MultivariateNormalFull(tf.reshape(tf.tile(mu, [N]), [N, D]),
                            tf.reshape(tf.tile(sigma, [N, 1]), [N, 2, 2]))

print('***** PROBABILISTIC MODEL *****')
print('mu: {}'.format(mu))
print('sigma: {}'.format(sigma))
print('xn: {}'.format(xn))

# Variational model
qmu = MultivariateNormalFull(
    tf.Variable(tf.random_normal([D], dtype=tf.float64), name='v1'),
    tf.nn.softplus(
        tf.Variable(tf.random_normal([D, D], dtype=tf.float64), name='v2')))
qsigma = WishartCholesky(
    df=tf.nn.softplus(
        tf.Variable(tf.random_normal([], dtype=tf.float64), name='v3')),
    scale=tf.nn.softplus(
        tf.Variable(tf.random_normal([D, D], dtype=tf.float64), name='v4')))

print('***** VARIATIONAL MODEL *****')
print('qmu: {}'.format(qmu))
print('qsigma: {}'.format(qsigma))

# Inference
print('xn_data: {}'.format(xn_data.dtype))
inference = ed.KLqp({mu: qmu, sigma: qsigma}, data={xn: xn_data})
inference.run(n_iter=2000, n_samples=20)

sess = ed.get_session()
