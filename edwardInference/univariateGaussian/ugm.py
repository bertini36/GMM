# -*- coding: UTF-8 -*-

"""
Gradient Ascent Variational Inference
process to approximate an univariate gaussian
"""

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from edward.models import Gamma, Normal

N = 1000
ed.set_seed(42)

# Data generation
xn = np.random.normal(7, 1, [N])
plt.plot(xn, 'go')
plt.title('Simulated dataset')
plt.show()
print('mu=7')
print('sigma=1')

# Priors definition
m = tf.Variable([0.], trainable=False)
beta = tf.Variable([0.0001], trainable=False)
a = tf.Variable([0.001], trainable=False)
b = tf.Variable([0.001], trainable=False)

# Posterior inference
# Probabilistic model
mu = Normal(mu=m, sigma=beta)
sigma = Gamma(alpha=a, beta=b)
x = Normal(mu=tf.tile(mu, [N]), sigma=tf.tile(sigma, [N]))

# Variational model
qmu = Normal(mu=tf.Variable(tf.random_normal([1])),
             sigma=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))
qsigma = Gamma(alpha=tf.nn.softplus(tf.Variable(tf.random_normal([1]))),
               beta=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

# Inference
inference = ed.KLqp({mu: qmu, sigma: qsigma}, data={x: xn})
inference.run(n_iter=1500, n_samples=500)

sess = ed.get_session()

print('Inferred mu={}'.format(sess.run(qmu.mean())))
print('Inferred sigma={}'.format(sess.run(qsigma.mean())))
