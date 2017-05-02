# -*- coding: UTF-8 -*-

"""
Black Box Variational Inference
process to approximate an univariate gaussian
[DOING]
"""

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from edward.models import Gamma, Normal

N = 1000

# Data generation
xn = np.random.normal(7, 1, [N])
plt.plot(xn, 'go')
plt.title('Simulated dataset')
plt.show()
print('mu=7')
print('sigma=1')

# Priors definition
m = tf.constant([0.])
beta = tf.constant([0.0001])
a = tf.constant([0.001])
b = tf.constant([0.001])

# Posterior inference
# Probabilistic model
mu = Normal(loc=m, scale=beta)
sigma = Gamma(a, b)
x = Normal(loc=tf.tile(mu, [N]), scale=tf.tile(sigma, [N]))

# Variational model
qmu = Normal(loc=tf.Variable(tf.random_normal([1])),
             scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))
qsigma = Gamma(tf.nn.softplus(tf.Variable(tf.random_normal([1]))),
               tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

# Inference
inference = ed.KLqp({mu: qmu, sigma: qsigma}, data={x: xn})
inference.run(n_iter=1500, n_samples=30)

sess = ed.get_session()

print('Inferred mu={}'.format(sess.run(qmu.mean())))
print('Inferred sigma={}'.format(sess.run(qsigma.mean())))
print('Inferred sigma={}'.format(sess.run(1/qsigma.mean())))
