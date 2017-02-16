# -*- coding: UTF-8 -*-

"""
NormalInverseWishart-Normal Model
Posterior inference with Edward MFVI
"""

import edward as ed
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from edward.models import Normal, MultivariateNormalDiag, WishartFull
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

print('mu={}'.format(mu))
print('sigma={}'.format(sigma))

xn_data = np.random.multivariate_normal(mu, sigma, N)
plt.scatter(xn_data[:, 0], xn_data[:, 1], cmap=cm.gist_rainbow, s=5)
plt.show()

# Prior definition
v_prior = tf.Variable(3., dtype=tf.float64, trainable=False)
W_prior = tf.Variable(np.array([[1., 0.], [0., 1.]]), dtype=tf.float64,
                      trainable=False)
m_prior = tf.Variable(np.array([0.5, 0.5]), dtype=tf.float64, trainable=False)
k_prior = tf.Variable(0.6, dtype=tf.float64, trainable=False)

# Posterior inference MFVI
# Probabilistic model
sigma = WishartFull(df=v_prior, scale=W_prior)
print('sigma: {}'.format(sigma))
mu = MultivariateNormalDiag(m_prior, tf.diag_part(k_prior * sigma))
print('mu: {}'.format(mu))
xn = MultivariateNormalDiag(tf.ones([N, D], dtype=tf.float64) * mu,
                            tf.ones([N, 1], dtype=tf.float64) * tf.reshape(
                                tf.diag_part(
                                    tf.ones([D, D], dtype=tf.float64) * sigma),
                                [1, D]))

# Variational model
qmu = MultivariateNormalDiag(
    tf.Variable(tf.random_normal([D], dtype=tf.float64)),
    tf.nn.softplus(tf.Variable(tf.random_normal([D], dtype=tf.float64))))
print('qmu: {}'.format(qmu))
qsigma = WishartFull(
    df=tf.nn.softplus(tf.Variable(tf.random_normal([], dtype=tf.float64))),
    scale=tf.Variable(tf.random_normal([D, D], dtype=tf.float64)))
print('qsigma: {}'.format(qsigma))

# Inference
inference = ed.KLqp({mu: qmu, sigma: qsigma}, data={xn: xn_data})
inference.run(n_iter=5000)

sess = ed.get_session()
