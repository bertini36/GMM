# -*- coding: UTF-8 -*-

import math
import numpy as np
import tensorflow as tf


DEBUG = True
MAX_EPOCHS = 400
N = 100
DATA_MEAN = 5

def printf(s):
    if DEBUG:
        print(s) 

# Data
# np.random.seed(42)
xn = tf.convert_to_tensor(np.random.normal(DATA_MEAN, 1, N), dtype=tf.float64)

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
LB = tf.sub(LB, tf.mul(tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64)), 
                           tf.mul(tf.div(a_gamma, b_gamma), tf.add(tf.pow(m_mu, 2), tf.div(tf.cast(1., tf.float64), beta_mu)))))

# Parameter updates
assign_m_mu = m_mu.assign(tf.div(tf.add(tf.mul(beta, m), tf.mul(tf.div(a_gamma, b_gamma), tf.reduce_sum(xn))), 
                                        tf.add(beta, tf.mul(tf.cast(N, tf.float64), tf.div(a_gamma, b_gamma)))))
assign_beta_mu = beta_mu.assign(tf.add(beta, tf.mul(tf.cast(N, tf.float64), tf.div(a_gamma, b_gamma))))
assign_a_gamma = a_gamma.assign(tf.add(a, tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64))))
assign_b_gamma = b_gamma.assign(tf.add(b, tf.add(tf.sub(tf.mul(tf.cast(1./2, tf.float64), tf.reduce_sum(tf.pow(xn, 2))), tf.mul(m_mu, tf.reduce_sum(xn))), 
                                                 tf.mul(tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64)), 
                                                        tf.add(tf.pow(m_mu, 2), tf.div(tf.cast(1., tf.float64), beta_mu))))))

# Summaries definition
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
    while epoch < MAX_EPOCHS:
        sess.run([assign_m_mu, assign_beta_mu, assign_a_gamma])
        sess.run(assign_b_gamma)
        m_mu_out, beta_mu_out, a_gamma_out, b_gamma_out = sess.run([m_mu, beta_mu, a_gamma, b_gamma])
        mer, lb = sess.run([merged, LB])
        printf('Epoch {}: Mu={} Precision={} ELBO={}'.format(epoch, m_mu_out, a_gamma_out/b_gamma_out, lb))
        run_calls += 1
        file_writer.add_summary(mer, run_calls)
        if epoch > 0:
            inc = (old_lb-lb)/old_lb*100
            if inc < 1e-8:
                break
        old_lb = lb
        epoch += 1