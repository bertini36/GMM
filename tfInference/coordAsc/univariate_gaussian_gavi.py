# -*- coding: UTF-8 -*-

import math
import numpy as np
import tensorflow as tf

DEBUG = True
SUMMARIES = True
PRECISON = 0.0000001

def printf(s):
    if DEBUG:
        print(s) 

# Data
N = 100
# np.random.seed(7)
xn = tf.convert_to_tensor(np.random.normal(5, 1, N), dtype=tf.float64)

m = tf.Variable(0., dtype=tf.float64)
beta = tf.Variable(0.0001, dtype=tf.float64)
a = tf.Variable(0.001, dtype=tf.float64)
b = tf.Variable(0.001, dtype=tf.float64)

# Needed for variational initilizations
a_gamma_ini = np.random.gamma(1, 1, 1)[0]
b_gamma_ini = np.random.gamma(1, 1, 1)[0]

# Variational parameters
a_gamma_var = tf.Variable(a_gamma_ini, dtype=tf.float64)
b_gamma_var = tf.Variable(b_gamma_ini, dtype=tf.float64)
m_mu = tf.Variable(np.random.normal(0., (0.0001)**(-1.), 1)[0], dtype=tf.float64)
beta_mu_var = tf.Variable(np.random.gamma(a_gamma_ini, b_gamma_ini, 1)[0], dtype=tf.float64)

# Maintain numerical stability
a_gamma = tf.add(tf.nn.softplus(a_gamma_var), PRECISON)
b_gamma = tf.add(tf.nn.softplus(b_gamma_var), PRECISON)
beta_mu = tf.add(tf.nn.softplus(beta_mu_var), PRECISON)

LB = tf.mul(tf.cast(1./2, tf.float64), tf.log(tf.div(beta, beta_mu)))
LB = tf.add(LB, tf.mul(tf.mul(tf.cast(1./2, tf.float64), tf.add(tf.pow(m_mu, 2), tf.div(tf.cast(1., tf.float64), beta_mu))), tf.sub(beta_mu, beta)))
LB = tf.sub(LB, tf.mul(m_mu, tf.sub(tf.mul(beta_mu, m_mu), tf.mul(beta, m))))
LB = tf.add(LB, tf.mul(tf.cast(1./2, tf.float64), tf.sub(tf.mul(beta_mu, tf.pow(m_mu, 2)), tf.mul(beta, tf.pow(m, 2)))))

LB = tf.add(LB, tf.mul(a, tf.log(b)))
LB = tf.sub(LB, tf.mul(a_gamma, tf.log(b_gamma)))
LB = tf.add(LB, tf.lgamma(a_gamma))
LB = tf.sub(LB, tf.lgamma(a))
LB = tf.add(LB, tf.mul(tf.sub(tf.digamma(a_gamma), tf.log(b_gamma)), tf.sub(a, a_gamma)))
LB = tf.add(LB, tf.mul(tf.div(a_gamma, b_gamma), tf.sub(b_gamma, b)))

LB = tf.add(LB, tf.mul(tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64)), tf.sub(tf.digamma(a_gamma), tf.log(b_gamma))))
LB = tf.sub(LB, tf.mul(tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64)), tf.log(tf.mul(tf.cast(2., tf.float64), math.pi))))
LB = tf.sub(LB, tf.mul(tf.cast(1./2, tf.float64), tf.mul(tf.div(a_gamma, b_gamma), tf.reduce_sum(tf.pow(xn, 2)))))
LB = tf.add(LB, tf.mul(tf.div(a_gamma, b_gamma), tf.mul(tf.reduce_sum(xn), m_mu)))
LB = tf.sub(LB, tf.mul(tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64)), tf.mul(tf.div(a_gamma, b_gamma), tf.add(tf.pow(m_mu, 2), tf.div(tf.cast(1., tf.float64), beta_mu)))))

# Optimizer definition
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
grads_and_vars = optimizer.compute_gradients(-LB, var_list=[a_gamma_var, b_gamma_var, m_mu, beta_mu_var])
train = optimizer.apply_gradients(grads_and_vars)

# Summaries definition
if SUMMARIES:
    tf.summary.histogram('m_mu', m_mu)
    tf.summary.histogram('beta_mu', beta_mu)
    tf.summary.histogram('a_gamma', a_gamma)
    tf.summary.histogram('b_gamma', b_gamma)
    merged = tf.summary.merge_all()
    file_writer = tf.summary.FileWriter('/tmp/tensorboard/', tf.get_default_graph())
    run_calls = 0

# Main program
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(10):
        if SUMMARIES:
            _, mer, lb, grads = sess.run([train, merged, LB, grads_and_vars])
            run_calls += 1
            file_writer.add_summary(mer, run_calls)
        else:
            _, lb = sess.run([train, LB])
        printf('***** Epoch {} *****'.format(epoch))
        printf('ELBO={}'.format(lb))
        printf('a_gamma: value={} gradient={}'.format(grads[0][1], grads[0][0]))
        printf('b_gamma: value={} gradient={}'.format(grads[1][1], grads[1][0]))
        printf('m_mu: value={} gradient={}'.format(grads[2][1], grads[2][0]))
        printf('beta_mu: value={} gradient={}'.format(grads[3][1], grads[3][0]))