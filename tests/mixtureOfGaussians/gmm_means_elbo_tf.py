# -*- coding: UTF-8 -*-

import math
import numpy as np
import pickle as pkl
import tensorflow as tf

np.random.seed(7)
sess = tf.Session()


def tf_log_beta_function(x):
    return tf.subtract(
        tf.reduce_sum(tf.lgamma(tf.add(x, np.finfo(np.float32).eps))),
        tf.lgamma(tf.reduce_sum(tf.add(x, np.finfo(np.float32).eps))))


def tf_dirichlet_expectation(alpha):
    if len(alpha.get_shape()) == 1:
        return tf.subtract(tf.digamma(tf.add(alpha, np.finfo(np.float32).eps)),
                           tf.digamma(tf.reduce_sum(alpha)))
    return tf.subtract(tf.digamma(alpha),
                       tf.digamma(tf.reduce_sum(alpha, 1))[:, tf.newaxis])


with open('../../data/data_k2_100.pkl', 'r') as inputfile:
    data = pkl.load(inputfile)
    xn = data['xn']

# Configurations
N, D = xn.shape
K = 2
alpha_aux = [1.0, 1.0]
alpha = tf.convert_to_tensor([alpha_aux], dtype=tf.float64)
m_o_aux = np.array([0.0, 0.0])
m_o = tf.convert_to_tensor([list(m_o_aux)], dtype=tf.float64)
beta_o_aux = 0.01
beta_o = tf.convert_to_tensor(beta_o_aux, dtype=tf.float64)
Delta_o_aux = np.array([[1.0, 0.0], [0.0, 1.0]])
Delta_o = tf.convert_to_tensor(Delta_o_aux, dtype=tf.float64)

# Initializations
phi_aux = np.random.dirichlet(alpha_aux, N)
lambda_pi_aux = alpha_aux + np.sum(phi_aux, axis=0)
lambda_mu_beta_aux = beta_o_aux + np.sum(phi_aux, axis=0)
lambda_mu_m_aux = np.tile(1. / lambda_mu_beta_aux, (2, 1)).T * \
                  (beta_o_aux * m_o_aux + np.dot(phi_aux.T, data['xn']))

# Variables
phi = tf.Variable(phi_aux, dtype=tf.float64)
lambda_pi = tf.Variable(lambda_pi_aux, dtype=tf.float64)
lambda_mu_beta = tf.Variable(lambda_mu_beta_aux, dtype=tf.float64)
lambda_mu_m = tf.Variable(lambda_mu_m_aux, dtype=tf.float64)

xn = tf.convert_to_tensor(xn, dtype=tf.float64)

# Reshapes
lambda_mu_beta_res = tf.reshape(lambda_mu_beta, [K, 1])
lambda_pi_res = tf.reshape(lambda_pi, [K, 1])

init = tf.global_variables_initializer()
sess.run(init)

# ELBO 
ELBO = tf_log_beta_function(lambda_pi)
ELBO = tf.subtract(ELBO, tf_log_beta_function(alpha))
ELBO = tf.add(ELBO, tf.matmul(tf.subtract(alpha, lambda_pi),
                              tf.reshape(tf_dirichlet_expectation(lambda_pi),
                                         [K, 1])))
ELBO = tf.add(ELBO, tf.multiply(tf.cast(K / 2., tf.float64), tf.log(
    tf.matrix_determinant(tf.multiply(beta_o, Delta_o)))))
ELBO = tf.add(ELBO, tf.cast(K * (D / 2.), tf.float64))
for k in range(K):
    a1 = tf.subtract(lambda_mu_m[k, :], m_o)
    a2 = tf.matmul(Delta_o, tf.transpose(tf.subtract(lambda_mu_m[k, :], m_o)))
    a3 = tf.multiply(tf.div(beta_o, 2.), tf.matmul(a1, a2))
    a4 = tf.div(tf.multiply(tf.cast(D, tf.float64), beta_o),
                tf.multiply(tf.cast(2., tf.float64), lambda_mu_beta_res[k]))
    a5 = tf.multiply(tf.cast(1 / 2., tf.float64), tf.log(
        tf.multiply(tf.pow(lambda_mu_beta_res[k], 2),
                    tf.matrix_determinant(Delta_o))))
    a6 = tf.add(a3, tf.add(a4, a5))
    ELBO = tf.subtract(ELBO, a6)
    b1 = tf.transpose(phi[:, k])
    b2 = tf_dirichlet_expectation(lambda_pi)[k]
    b3 = tf.log(phi[:, k])
    b4 = tf.multiply(tf.cast(1 / 2., tf.float64), tf.log(
        tf.div(tf.matrix_determinant(Delta_o),
               tf.multiply(tf.cast(2., tf.float64), math.pi))))
    b5 = tf.subtract(xn, lambda_mu_m[k, :])
    b6 = tf.matmul(Delta_o, tf.transpose(tf.subtract(xn, lambda_mu_m[k, :])))
    b7 = tf.multiply(tf.cast(1 / 2., tf.float64),
                     tf.stack([tf.matmul(b5, b6)[i, i] for i in range(N)]))
    b8 = tf.div(tf.cast(D, tf.float64),
                tf.multiply(tf.cast(2., tf.float64), lambda_mu_beta[k]))
    b9 = tf.subtract(tf.subtract(tf.add(tf.subtract(b2, b3), b4), b7), b8)
    b1 = tf.reshape(b1, [1, N])
    b9 = tf.reshape(b9, [N, 1])
    ELBO = tf.add(ELBO, tf.reshape(tf.matmul(b1, b9), [1]))

elbo = sess.run(ELBO)
print('ELBO={}'.format(elbo))
