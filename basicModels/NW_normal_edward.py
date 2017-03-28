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
v_prior = tf.Variable(3., dtype=tf.float32, trainable=False)
W_prior = tf.Variable(np.array([[1., 0.], [0., 1.]]),
                      dtype=tf.float32, trainable=False)
m_prior = tf.Variable(np.array([0.5, 0.5]), dtype=tf.float32, trainable=False)
k_prior = tf.Variable(0.6, dtype=tf.float32, trainable=False)

# Posterior inference
# Probabilistic model
sigma = WishartCholesky(df=v_prior, scale=W_prior)
mu = MultivariateNormalFull(m_prior, k_prior * sigma)
xn = MultivariateNormalFull(tf.reshape(tf.tile(mu, [N]), [N, D]),
                            tf.reshape(tf.tile(sigma, [N, 1]), [N, 2, 2]))
# Variational model
qmu = MultivariateNormalFull(
    tf.Variable(tf.random_normal([D], dtype=tf.float32), name='v1'),
    tf.nn.softplus(
        tf.Variable(tf.random_normal([D, D], dtype=tf.float32), name='v2')))
qsigma = WishartCholesky(
    df=tf.nn.softplus(
        tf.Variable(tf.random_normal([], dtype=tf.float32), name='v3')),
    scale=tf.nn.softplus(
        tf.Variable(tf.random_normal([D, D], dtype=tf.float32), name='v4')))

# Inference
inference = ed.KLqp({mu: qmu, sigma: qsigma}, data={xn: xn_data})
inference.run(n_iter=500, n_samples=20)

sess = ed.get_session()

print('Inferred mu: {}'.format(sess.run(qmu.mean())))
print('Inferred sigma: {}'.format(sess.run(qsigma.mean())))
