# -*- coding: UTF-8 -*-

import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils import tf_log_beta_function, tf_dirichlet_expectation

DEBUG = True

def printf(s):
    if DEBUG:
        print(s) 

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

# Restrictions
a_gamma = tf.nn.softplus(v_a_gamma)
b_gamma = tf.nn.softplus(v_b_gamma)
m_mu = tf.nn.softplus(v_m_mu)
beta_mu = tf.nn.softplus(v_beta_mu)

# Data
xn = tf.convert_to_tensor(np.random.normal(5, 1, N), dtype=tf.float64)

# Primera linea en la version de Joan
LB1 = tf.add(tf.mul(tf.cast(1./2, tf.float64), tf.log(tf.div(beta, beta_mu))), 
             tf.mul(tf.mul(tf.cast(1./2, tf.float64), tf.pow(m_mu, tf.add(tf.cast(2., tf.float64), tf.div(tf.cast(1., tf.float64), beta_mu)))), tf.sub(beta_mu, beta)))
LB2 = tf.sub(LB1, tf.mul(m_mu, tf.sub(tf.mul(beta_mu, m_mu), tf.mul(beta, m))))
LB3 = tf.add(LB2, tf.mul(tf.cast(1./2, tf.float64), tf.sub(tf.mul(beta_mu, tf.pow(m_mu, 2)), tf.mul(beta, tf.pow(m, 2)))))

# Segunda linea en la version de Joan
LB4 = tf.add(LB3, tf.sub(tf.mul(a, tf.log(b)), tf.mul(a_gamma, tf.log(b_gamma))))
LB5 = tf.add(LB4, tf.add(tf.sub(tf.lgamma(a_gamma), tf.lgamma(a)), tf.mul(tf.sub(tf.digamma(a_gamma), tf.log(b_gamma)), tf.sub(a, a_gamma))))
LB6 = tf.add(LB5, tf.mul(tf.div(a_gamma, b_gamma), tf.sub(b_gamma, b)))

# Tercera linea en la version de Joan
LB7 = tf.add(LB6, tf.mul(tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64)), tf.sub(tf.digamma(a_gamma), tf.log(b_gamma))))
LB8 = tf.sub(LB7, tf.sub(tf.mul(tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64)), tf.log(tf.mul(tf.cast(2., tf.float64), math.pi))), 
                         tf.mul(tf.cast(1./2, tf.float64), tf.mul(tf.div(a_gamma, b_gamma), tf.reduce_sum(tf.pow(xn, 2))))))
LB9 = tf.add(LB8, tf.sub(tf.mul(tf.div(a_gamma, b_gamma), tf.mul(tf.reduce_sum(xn), m_mu)), 
                         tf.mul(tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64)),
                                tf.mul(tf.div(a_gamma, b_gamma), tf.add(tf.pow(m_mu, 2), tf.div(tf.cast(1., tf.float64), beta_mu))))))

# Optimizer definition
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(LB9, var_list=[v_a_gamma, v_b_gamma, v_m_mu, v_beta_mu])

# Main program
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(5):
        _, lb1, lb2, lb3, lb4, lb5, lb6, lb7, lb8, lb9 = sess.run([train, LB1, LB2, LB3, LB4, LB5, LB6, LB7, LB8, LB9])
        printf('LB1: {}'.format(lb1))
        printf('LB2: {}'.format(lb2))
        printf('LB3: {}'.format(lb3))
        printf('LB4: {}'.format(lb4))
        printf('LB5: {}'.format(lb5))
        printf('LB6: {}'.format(lb6))
        printf('LB7: {}'.format(lb7))
        printf('LB8: {}'.format(lb8))
        printf('LB9: {}'.format(lb9))