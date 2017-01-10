# -*- coding: UTF-8 -*-

import math
import numpy as np
import tensorflow as tf

DEBUG = True
SUMMARIES = True
MAX_EPOCHS = 5000

def printf(s):
    if DEBUG:
        print(s) 

# Data
N = 100
np.random.seed(7)
xn = tf.convert_to_tensor(np.random.normal(17, 1, N), dtype=tf.float64)

m = tf.Variable(0., dtype=tf.float64)
beta = tf.Variable(0.0001, dtype=tf.float64)
a = tf.Variable(0.001, dtype=tf.float64)
b = tf.Variable(0.001, dtype=tf.float64)

# Needed for variational initilizations
a_gamma_ini = np.random.gamma(1, 1, 1)[0]
b_gamma_ini = np.random.gamma(1, 1, 1)[0]

# Variational parameters
a_gamma = tf.Variable(a_gamma_ini, dtype=tf.float64)
b_gamma = tf.Variable(b_gamma_ini, dtype=tf.float64)
m_mu = tf.Variable(np.random.normal(0., (0.0001)**(-1.), 1)[0], dtype=tf.float64)
beta_mu = tf.Variable(np.random.gamma(a_gamma_ini, b_gamma_ini, 1)[0], dtype=tf.float64)

# Lower Bound definition
ELBO1 = tf.mul(tf.cast(1./2, tf.float64), tf.log(tf.div(beta, beta_mu)))
ELBO2 = tf.add(ELBO1, tf.mul(tf.mul(tf.cast(1./2, tf.float64), tf.add(tf.pow(m_mu, 2), tf.div(tf.cast(1., tf.float64), beta_mu))), tf.sub(beta_mu, beta)))
ELBO3 = tf.sub(ELBO2, tf.mul(m_mu, tf.sub(tf.mul(beta_mu, m_mu), tf.mul(beta, m))))
ELBO4 = tf.add(ELBO3, tf.mul(tf.cast(1./2, tf.float64), tf.sub(tf.mul(beta_mu, tf.pow(m_mu, 2)), tf.mul(beta, tf.pow(m, 2)))))

ELBO5 = tf.add(ELBO4, tf.mul(a, tf.log(b)))
ELBO6 = tf.sub(ELBO5, tf.mul(a_gamma, tf.log(b_gamma)))
ELBO7 = tf.add(ELBO6, tf.lgamma(a_gamma))
ELBO8 = tf.sub(ELBO7, tf.lgamma(a))
ELBO9 = tf.add(ELBO8, tf.mul(tf.sub(tf.digamma(a_gamma), tf.log(b_gamma)), tf.sub(a, a_gamma)))
ELBO10 = tf.add(ELBO9, tf.mul(tf.div(a_gamma, b_gamma), tf.sub(b_gamma, b)))

ELBO11 = tf.add(ELBO10, tf.mul(tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64)), tf.sub(tf.digamma(a_gamma), tf.log(b_gamma))))
ELBO12 = tf.sub(ELBO11, tf.mul(tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64)), tf.log(tf.mul(tf.cast(2., tf.float64), math.pi))))
ELBO13 = tf.sub(ELBO12, tf.mul(tf.cast(1./2, tf.float64), tf.mul(tf.div(a_gamma, b_gamma), tf.reduce_sum(tf.pow(xn, 2)))))
ELBO14 = tf.add(ELBO13, tf.mul(tf.div(a_gamma, b_gamma), tf.mul(tf.reduce_sum(xn), m_mu)))
ELBO15 = tf.sub(ELBO14, tf.mul(tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64)), 
                           tf.mul(tf.div(a_gamma, b_gamma), tf.add(tf.pow(m_mu, 2), tf.div(tf.cast(1., tf.float64), beta_mu)))))

# Parameter updates
assign_m_mu = m_mu.assign(tf.div(tf.add(tf.mul(beta, m), tf.mul(tf.div(a_gamma, b_gamma), tf.reduce_sum(xn))), 
                                        tf.add(beta, tf.mul(tf.cast(N, tf.float64), tf.div(a_gamma, b_gamma)))))
assign_beta_mu = beta_mu.assign(tf.add(beta, tf.mul(tf.cast(N, tf.float64), tf.div(a_gamma, b_gamma))))
assign_a_gamma = a_gamma.assign(tf.add(a, tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64))))

# b_gamma = b + 1./2*sum(xn**2) - m_mu*sum(xn) + N/2.*(m_mu**2+1./beta_mu)
aux2 = tf.mul(tf.cast(1./2, tf.float64), tf.reduce_sum(tf.pow(xn, 2)))
aux3 = tf.mul(m_mu, tf.reduce_sum(xn))

# N/2.*(m_mu**2+1./beta_mu)
aux5 = tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64))
aux6 = tf.pow(m_mu, 2)
aux7 = tf.div(tf.cast(1., tf.float64), beta_mu)
aux4 = tf.mul(aux5, tf.add(aux6, aux7))

# aux1 = aux2-aux3+aux4 
aux1 = tf.add(tf.sub(aux2, aux3), aux4)

assign_b_gamma = b_gamma.assign(tf.add(b, aux1))

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
    epoch = 0
    m_mu_out, beta_mu_out, a_gamma_out, b_gamma_out = sess.run([m_mu, beta_mu, a_gamma, b_gamma])
    printf('Init: m_mu={} beta_mu={} a_gamma={} b_gamma={} Precision={}'.format(epoch, m_mu_out, beta_mu_out, a_gamma_out, b_gamma_out, a_gamma_out/b_gamma_out))
    printf('**************************************************************')
    # while epoch < MAX_EPOCHS:
    for i in xrange(10):
        sess.run(assign_m_mu)
        sess.run(assign_beta_mu)
        sess.run(assign_a_gamma)
        sess.run(assign_b_gamma)
        m_mu_out, beta_mu_out, a_gamma_out, b_gamma_out = sess.run([m_mu, beta_mu, a_gamma, b_gamma])
        printf('Epoch {}: m_mu={} beta_mu={} a_gamma={} b_gamma={} Precision={}'.format(i, m_mu_out, beta_mu_out, a_gamma_out, b_gamma_out, a_gamma_out/b_gamma_out))
        a1, a2, a3, a4, a5, a6, a7 = sess.run([aux1, aux2, aux3, aux4, aux5, aux6, aux7])
        printf('aux1={} = aux2={}, aux3={}, aux4={}'.format(a1, a2, a3, a4))
        lb1, lb2, lb3, lb4 = sess.run([ELBO1, ELBO2, ELBO3, ELBO4])
        # printf('Epoch {}: ELBO1={} ELBO2={} ELBO3={} ELBO4={} aux1={}'.format(i, lb1, lb2, lb3, lb4, a1))

        lb = sess.run(ELBO15)
        printf('Epoch {}: ELBO={}'.format(i, lb))

        printf('**************************************************************')

        """
        if epoch > 0:
            inc = (old_lb-lb)/old_lb*100
            if inc < 1e-8:
                break
        old_lb = lb
        epoch += 1
        """