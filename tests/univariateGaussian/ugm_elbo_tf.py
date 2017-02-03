# -*- coding: UTF-8 -*-

import math
import numpy as np
import tensorflow as tf

N = 100
np.random.seed(7)
xn = tf.convert_to_tensor(np.random.normal(5, 1, N), dtype=tf.float64)

# Probabilistic parameters
m = tf.Variable(0., dtype=tf.float64)
beta = tf.Variable(0.0001, dtype=tf.float64)
a = tf.Variable(0.001, dtype=tf.float64)
b = tf.Variable(0.001, dtype=tf.float64)

a_gamma = tf.constant(1., dtype=tf.float64)
b_gamma = tf.constant(1., dtype=tf.float64)
m_mu = tf.constant(10., dtype=tf.float64)
beta_mu = tf.constant(1., dtype=tf.float64)

LB = tf.multiply(tf.cast(1. / 2, tf.float64), tf.log(tf.div(beta, beta_mu)))
LB = tf.add(LB, tf.multiply(tf.multiply(tf.cast(1. / 2, tf.float64),
                                        tf.add(tf.pow(m_mu, 2),
                                               tf.div(tf.cast(1., tf.float64),
                                                      beta_mu))),
                            tf.subtract(beta_mu, beta)))
LB = tf.subtract(LB, tf.multiply(m_mu, tf.subtract(tf.multiply(beta_mu, m_mu),
                                                   tf.multiply(beta, m))))
LB = tf.add(LB, tf.multiply(tf.cast(1. / 2, tf.float64),
                            tf.subtract(tf.multiply(beta_mu, tf.pow(m_mu, 2)),
                                        tf.multiply(beta, tf.pow(m, 2)))))

LB = tf.add(LB, tf.multiply(a, tf.log(b)))
LB = tf.subtract(LB, tf.multiply(a_gamma, tf.log(b_gamma)))
LB = tf.add(LB, tf.lgamma(a_gamma))
LB = tf.subtract(LB, tf.lgamma(a))
LB = tf.add(LB, tf.multiply(tf.subtract(tf.digamma(a_gamma), tf.log(b_gamma)),
                            tf.subtract(a, a_gamma)))
LB = tf.add(LB, tf.multiply(tf.div(a_gamma, b_gamma), tf.subtract(b_gamma, b)))

LB = tf.add(LB,
            tf.multiply(tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64)),
                        tf.subtract(tf.digamma(a_gamma), tf.log(b_gamma))))
LB = tf.subtract(LB, tf.multiply(
    tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64)),
    tf.log(tf.multiply(tf.cast(2., tf.float64), math.pi))))
LB = tf.subtract(LB, tf.multiply(tf.cast(1. / 2, tf.float64),
                                 tf.multiply(tf.div(a_gamma, b_gamma),
                                             tf.reduce_sum(tf.pow(xn, 2)))))
LB = tf.add(LB,
            tf.multiply(tf.div(a_gamma, b_gamma),
                        tf.multiply(tf.reduce_sum(xn), m_mu)))
LB = tf.subtract(LB, tf.multiply(
    tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64)),
    tf.multiply(tf.div(a_gamma, b_gamma), tf.add(tf.pow(m_mu, 2),
                                                 tf.div(
                                                     tf.cast(1.,
                                                             tf.float64),
                                                     beta_mu)))))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    mu, beta, a, b, elbo = sess.run([m_mu, beta_mu, a_gamma, b_gamma, LB])
    print('Mean: {} Precision: {} ELBO: {}'.format(mu, a / b, elbo))
