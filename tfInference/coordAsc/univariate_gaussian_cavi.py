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
np.random.seed(7)
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

LB1 = tf.mul(tf.cast(1./2, tf.float64), tf.log(tf.div(beta, beta_mu)))
LB2 = tf.add(LB1, tf.mul(tf.mul(tf.cast(1./2, tf.float64), tf.add(tf.pow(m_mu, 2), tf.div(tf.cast(1., tf.float64), beta_mu))), tf.sub(beta_mu, beta)))
LB3 = tf.sub(LB2, tf.mul(m_mu, tf.sub(tf.mul(beta_mu, m_mu), tf.mul(beta, m))))
LB4 = tf.add(LB3, tf.mul(tf.cast(1./2, tf.float64), tf.sub(tf.mul(beta_mu, tf.pow(m_mu, 2)), tf.mul(beta, tf.pow(m, 2)))))

LB5 = tf.add(LB4, tf.mul(a, tf.log(b)))
LB6 = tf.sub(LB5, tf.mul(a_gamma, tf.log(b_gamma)))
LB7 = tf.add(LB6, tf.lgamma(a_gamma))
LB8 = tf.sub(LB7, tf.lgamma(a))
LB9 = tf.add(LB8, tf.mul(tf.sub(tf.digamma(a_gamma), tf.log(b_gamma)), tf.sub(a, a_gamma)))
LB10 = tf.add(LB9, tf.mul(tf.div(a_gamma, b_gamma), tf.sub(b_gamma, b)))

LB11 = tf.add(LB10, tf.mul(tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64)), tf.sub(tf.digamma(a_gamma), tf.log(b_gamma))))
LB12 = tf.sub(LB11, tf.mul(tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64)), tf.log(tf.mul(tf.cast(2., tf.float64), math.pi))))
LB13 = tf.sub(LB12, tf.mul(tf.cast(1./2, tf.float64), tf.mul(tf.div(a_gamma, b_gamma), tf.reduce_sum(tf.pow(xn, 2)))))
LB14 = tf.add(LB13, tf.mul(tf.div(a_gamma, b_gamma), tf.mul(tf.reduce_sum(xn), m_mu)))
LB15 = tf.sub(LB14, tf.mul(tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64)), tf.mul(tf.div(a_gamma, b_gamma), tf.add(tf.pow(m_mu, 2), tf.div(tf.cast(1., tf.float64), beta_mu)))))

# Optimizer definition
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(LB15, var_list=[a_gamma_var, b_gamma_var, m_mu, beta_mu_var])

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
        printf('***** Epoch {} *****'.format(epoch))
        if SUMMARIES:
            _, m, lb1, lb2, lb3, lb4, lb5, lb6, lb7, lb8, lb9, lb10, lb11, lb12, lb13, lb14, lb15 = sess.run([train, merged, LB1, LB2, LB3, LB4, LB5, LB6, LB7, LB8, LB9, LB10, LB11, LB12, LB13, LB14, LB15])
            run_calls += 1
            file_writer.add_summary(m, run_calls)
        else:
            _, lb1, lb2, lb3, lb4, lb5, lb6, lb7, lb8, lb9, lb10, lb11, lb12, lb13, lb14, lb15 = sess.run([train, LB1, LB2, LB3, LB4, LB5, LB6, LB7, LB8, LB9, LB10, LB11, LB12, LB13, LB14, LB15])
        printf('ELBO1={}'.format(lb1))
        printf('ELBO2={}'.format(lb2))
        printf('ELBO3={}'.format(lb3))
        printf('ELBO4={}'.format(lb4))
        printf('ELBO5={}'.format(lb5))
        printf('ELBO6={}'.format(lb6))
        printf('ELBO7={}'.format(lb7))
        printf('ELBO8={}'.format(lb8))
        printf('ELBO9={}'.format(lb9))
        printf('ELBO10={}'.format(lb10))
        printf('ELBO11={}'.format(lb11))
        printf('ELBO12={}'.format(lb12))
        printf('ELBO13={}'.format(lb13))
        printf('ELBO14={}'.format(lb14))
        printf('ELBO15={}'.format(lb15))