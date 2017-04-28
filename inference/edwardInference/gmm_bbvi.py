#!/usr/bin/env python

"""
Black Box Variational Inference
process to approximate a mixture of gaussians
[DOING]
"""

from __future__ import absolute_import, division, print_function

import edward as ed
import numpy as np
import tensorflow as tf
from edward.models import Categorical, Dirichlet, InverseGamma, Normal


def build_toy_dataset(N):
    pi = np.array([0.2, 0.8])
    mus = [[5.0, 5.0], [0.0, 0.0]]
    stds = [[1.0, 1.0], [1.0, 1.0]]
    x = np.zeros((N, 2))
    for n in range(N):
        k = np.argmax(np.random.multinomial(1, pi))
        x[n, :] = np.random.multivariate_normal(mus[k], np.diag(stds[k]))
    return x


N = 1000     # Number of data points
K = 2       # Number of components
D = 2       # Dimensionality of data

# DATA
x_data = build_toy_dataset(N)

# MODEL
pi = Dirichlet(concentration=tf.constant([1.0] * K))
mu = Normal(loc=tf.zeros([K, D]), scale=tf.ones([K, D]))
sigma = InverseGamma(concentration=tf.ones([K, D]), rate=tf.ones([K, D]))
c = Categorical(logits=tf.tile(tf.reshape(ed.logit(pi), [1, K]), [N, 1]))
x = Normal(loc=tf.gather(mu, c), scale=tf.gather(sigma, c))

# INFERENCE
qpi = Dirichlet(concentration=tf.nn.softplus(tf.Variable(tf.random_normal([K]))))
qmu = Normal(loc=tf.Variable(tf.random_normal([K, D])),
             scale=tf.nn.softplus(tf.Variable(tf.random_normal([K, D]))))
qsigma = InverseGamma(concentration=tf.nn.softplus(tf.Variable(tf.random_normal([K, D]))),
                      rate=tf.nn.softplus(tf.Variable(tf.random_normal([K, D]))))
qc = Categorical(logits=tf.Variable(tf.zeros([N, K])))

inference = ed.KLqp(
    latent_vars={pi: qpi, mu: qmu, sigma: qsigma, c: qc},
    data={x: x_data})

inference.initialize(n_iter=5000, n_samples=100)

sess = ed.get_session()
tf.global_variables_initializer().run()

for _ in range(inference.n_iter):
    info_dict = inference.update()
    inference.print_progress(info_dict)
    t = info_dict['t']
    if t == 1 or t % inference.n_print == 0:
        qpi_mean, qmu_mean = sess.run([qpi.mean(), qmu.mean()])
        print('\nInferred membership probabilities: {}'.format(qpi_mean))
        print('Inferred cluster means: {}'.format(qmu_mean))
