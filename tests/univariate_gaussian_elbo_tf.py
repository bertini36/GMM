# -*- coding: UTF-8 -*-
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import tf_log_beta_function, tf_dirichlet_expectation

N = 100
It = 50

# Needed for initilizations
aux_m = 0.
aux_beta = 0.0001
aux_a_gamma = np.random.gamma(1, 1, 1)[0]
aux_b_gamma = np.random.gamma(1, 1, 1)[0]

# Probabilistic parameters
m = tf.Variable(aux_m, dtype=tf.float64)
beta = tf.Variable(aux_beta, dtype=tf.float64)
a = tf.Variable(0.001, dtype=tf.float64)
b = tf.Variable(0.001, dtype=tf.float64)

# Variational parameters
v_a_gamma = tf.Variable(aux_a_gamma, dtype=tf.float64)
v_b_gamma = tf.Variable(aux_b_gamma, dtype=tf.float64)
v_m_mu = tf.Variable(np.random.normal(aux_m, (aux_beta)**(-1.), 1)[0], dtype=tf.float64)
v_beta_mu = tf.Variable(np.random.gamma(aux_a_gamma, aux_b_gamma, 1)[0], dtype=tf.float64)


a_gamma = tf.constant(1., dtype=tf.float64)
b_gamma = tf.constant(1., dtype=tf.float64)
m_mu = tf.constant(10., dtype=tf.float64)
beta_mu = tf.constant(1., dtype=tf.float64)

# Maintain numerical stability
#a_gamma = tf.nn.softplus(v_a_gamma)
#b_gamma = tf.nn.softplus(v_b_gamma)
#m_mu = v_m_mu
#beta_mu = tf.nn.softplus(v_beta_mu)

# Data
np.random.seed(123)
xn = tf.convert_to_tensor(np.random.normal(5, 1, N), dtype=tf.float64)

LB = tf.add(tf.mul(tf.cast(1./2, tf.float64), tf.log(tf.div(beta, beta_mu))), 
				tf.mul(tf.mul(tf.cast(1./2, tf.float64), tf.add(tf.pow(m_mu, 2), tf.div(tf.cast(1., tf.float64), beta_mu))), tf.sub(beta_mu, beta)))

LB = tf.sub(LB, tf.mul(m_mu, tf.sub(tf.mul(beta_mu, m_mu), tf.mul(beta, m))))

LB = tf.add(LB, tf.mul(tf.cast(1./2, tf.float64), tf.sub(tf.mul(beta_mu, tf.pow(m_mu, 2)), tf.mul(beta, tf.pow(m, 2)))))

LB = tf.add(LB, tf.sub(tf.mul(a, tf.log(b)), tf.mul(a_gamma, tf.log(b_gamma))))

LB = tf.add(LB, tf.add(tf.sub(tf.lgamma(a_gamma), tf.lgamma(a)), tf.mul(tf.sub(tf.digamma(a_gamma), tf.log(b_gamma)), tf.sub(a, a_gamma))))

LB = tf.add(LB, tf.mul(tf.div(a_gamma, b_gamma), tf.sub(b_gamma, b)))


LB = tf.add(LB, tf.mul(tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64)), tf.sub(tf.digamma(a_gamma), tf.log(b_gamma))))

LB = tf.sub(LB, tf.sub(tf.mul(tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64)), tf.log(tf.mul(tf.cast(2., tf.float64), math.pi))), 
				tf.mul(tf.cast(1./2, tf.float64), tf.mul(tf.div(a_gamma, b_gamma), tf.reduce_sum(tf.pow(xn, 2))))))

LB = tf.add(LB, tf.sub(tf.mul(tf.div(a_gamma, b_gamma), tf.mul(tf.reduce_sum(xn), m_mu)), 
				tf.mul(tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64)),
				tf.mul(tf.div(a_gamma, b_gamma), tf.add(tf.pow(m_mu, 2), tf.div(tf.cast(1., tf.float64), beta_mu))))))

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	mu, beta, a, b, elbo = sess.run([m_mu, beta_mu, a_gamma, b_gamma,LB])
	print('Mean={} Precision={} ELBO={}'.format(mu, a/b, elbo))