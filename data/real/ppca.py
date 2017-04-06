# -*- coding: UTF-8 -*-

"""
Probabilistic Principal Component Analysis
with automatic relevance determination
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from edward.models import Normal


def build_toy_dataset(N, D, K, sigma=1):
    x_train = np.zeros([D, N])
    w = np.zeros([D, K])
    for k in range(K):
        w[k, k] = 1.0 / (k+1)
        w[k+1, k] = -1.0 / (k+1)
    z = np.random.normal(0.0, 1.0, size=(K, N))
    mean = np.dot(w, z)
    shift = np.zeros([D])
    shift[1] = 10
    for d in range(D):
        for n in range(N):
            x_train[d, n] = np.random.normal(mean[d, n], sigma) + shift[d]
    return x_train.astype(np.float32, copy=False)


N = 1000
D = 10
KR = 3
K = D

x_train = build_toy_dataset(N, D, KR, sigma=0.1)

print('Length x_train: {}'.format(len(x_train[0])))
print('Dimensions x_train: {}'.format(len(x_train)))
print('x_train: ')
print(x_train)

ds = tf.contrib.distributions

sigma = ed.models.Gamma(1.0, 1.0)

alpha = ed.models.Gamma(tf.ones([K]), tf.ones([K]))
w = Normal(mu=tf.zeros([D, K]), sigma=tf.reshape(tf.tile(alpha, [D]), [D, K]))
z = Normal(mu=tf.zeros([K, N]), sigma=tf.ones([K, N]))
mu = Normal(mu=tf.zeros([D]), sigma=tf.ones([D]))
x = Normal(mu=tf.matmul(w, z)
              + tf.transpose(tf.reshape(tf.tile(mu, [N]), [N, D])),
           sigma=sigma*tf.ones([D, N]))

# INFERENCE
qalpha = ed.models.TransformedDistribution(
    distribution=ed.models.NormalWithSoftplusSigma(
        mu=tf.Variable(tf.random_normal([K])),
        sigma=tf.Variable(tf.random_normal([K]))),
    bijector=ds.bijector.Exp(),
    name='qalpha')

qw = Normal(mu=tf.Variable(tf.random_normal([D, K])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([D, K]))))
qz = Normal(mu=tf.Variable(tf.random_normal([K, N])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([K, N]))))

# initial conditions for qmu, improves convergence
data_mean = np.mean(x_train, axis=1).astype(np.float32, copy=False)

qmu = Normal(mu=tf.Variable(data_mean+tf.random_normal([D])),
             sigma=tf.nn.softplus(tf.Variable(tf.random_normal([D]))))

qsigma = ed.models.TransformedDistribution(
    distribution=ed.models.NormalWithSoftplusSigma(
        mu=tf.Variable(0.0), sigma=tf.Variable(1.0)),
    bijector=ds.bijector.Exp(), name='qsigma')

inference = ed.KLqp({alpha: qalpha, w: qw, z: qz, mu: qmu, sigma: qsigma},
                    data={x: x_train})
inference.run(n_iter=10000, n_samples=10)

print('Inferred principal axes (columns):')
print('Mean: {}'.format(qw.mean().eval()))
print('Variance: {}'.format(qw.variance().eval()))

print('Inferred center:')
print('Mean: {}'.format(qmu.mean().eval()))
print('Variance: {}'.format(qmu.variance().eval()))

print('Length new points: {}'.format(len(qz.eval()[0])))
print('Dimensions new points: {}'.format(len(qz.eval())))
print('New points: ')
print(qz.eval())

alphas = tf.exp(qalpha.distribution.mean()).eval()
alphas.sort()
plt.plot(range(alphas.size), alphas)
plt.show()
