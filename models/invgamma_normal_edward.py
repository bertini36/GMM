# -*- coding: UTF-8 -*-

"""
InverseGamma-Normal model (known mean)
Posterior inference with Edward BBVI
"""

import edward as ed
import numpy as np
import tensorflow as tf
from edward.models import InverseGamma, Normal

N = 1000

# Data generation (known mean)
mu = 7.0
sigma = 0.55
xn_data = np.random.normal(mu, sigma, N)
print('sigma={}'.format(sigma))

# Prior definition
alpha = tf.Variable(0.9, dtype=tf.float32, trainable=False)
beta = tf.Variable(0.5, dtype=tf.float32, trainable=False)

# Posterior inference
# Probabilistic model
ig = InverseGamma(alpha=alpha, beta=beta)
xn = Normal(mu=mu, sigma=tf.ones([N]) * tf.sqrt(ig))

# Variational model
qig = InverseGamma(alpha=tf.nn.softplus(tf.Variable(tf.random_normal([]))),
                   beta=tf.nn.softplus(tf.Variable(tf.random_normal([]))))

# Inference
inference = ed.KLqp({ig: qig}, data={xn: xn_data})
inference.run(n_iter=2000, n_samples=150, logdir='/tmp/train/')

sess = ed.get_session()
print('Inferred sigma={}'.format(sess.run(tf.sqrt(qig.mean()))))
