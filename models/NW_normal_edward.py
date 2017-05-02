# -*- coding: UTF-8 -*-

"""
NormalWishart-Normal Model
Posterior inference with Edward BBVI
"""

import edward as ed
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from edward.models import (MultivariateNormalCholesky, MultivariateNormalFull,
                           WishartCholesky)
from scipy import random
from scipy.stats import invwishart

N = 1000
D = 2
N_ITERS = 500
N_SAMPLES = 50


def generate_random_positive_matrix(D):
    """
    Generate a random semidefinite positive matrix
    :param D: Dimension
    :return: DxD matrix
    """
    aux = random.rand(D, D)
    return np.dot(aux, aux.transpose())


# Data generation
v = 3.
W = np.array(generate_random_positive_matrix(D))
sigma = invwishart.rvs(v, W)
m = np.array([1., 1.])
k = 0.8
mu = np.random.multivariate_normal(m, sigma / k)
xn_data = np.random.multivariate_normal(mu, sigma, N)
plt.scatter(xn_data[:, 0], xn_data[:, 1], cmap=cm.gist_rainbow, s=5)
plt.show()
print('mu={}'.format(mu))
print('sigma={}'.format(sigma))

# Prior definition
v_prior = tf.constant(3., dtype=tf.float64)
W_prior = tf.constant(generate_random_positive_matrix(D), dtype=tf.float64)
m_prior = tf.constant(np.array([0.5, 0.5]), dtype=tf.float64)
k_prior = tf.constant(0.6, dtype=tf.float64)

# Posterior inference
# Probabilistic model
sigma = WishartCholesky(df=v_prior, scale=W_prior)
mu = MultivariateNormalCholesky(m_prior, k_prior * sigma)
xn = MultivariateNormalFull(
    tf.reshape(tf.tile(mu, [N]), [N, D]),
    tf.reshape(tf.tile(sigma, [N, 1]), [N, 2, 2]))

# Variational model
random_matrix_1 = tf.Variable(tf.random_normal([D, D], dtype=tf.float64))
qmu = MultivariateNormalCholesky(
    tf.Variable(tf.random_normal([D], dtype=tf.float64)),
    tf.matmul(random_matrix_1, tf.transpose(random_matrix_1))
    + D * tf.eye(D, dtype=tf.float64))

random_matrix_2 = tf.Variable(tf.random_normal([D, D], dtype=tf.float64))
qsigma = WishartCholesky(
    df=tf.nn.softplus(tf.Variable(tf.random_normal([], dtype=tf.float64))),
    scale=tf.matmul(random_matrix_2, tf.transpose(random_matrix_2)) +
          D * tf.eye(D, dtype=tf.float64))

# Inference
inference = ed.KLqp({mu: qmu, sigma: qsigma}, data={xn: xn_data})
inference.run(n_iter=N_ITERS, n_samples=N_SAMPLES)

sess = ed.get_session()

print('Inferred mu: {}'.format(sess.run(qmu.mean())))
print('Inferred precision: {}'.format(sess.run(qsigma.mean())))
print('Inferred sigma: {}'.format(sess.run(1 / qsigma.mean())))
