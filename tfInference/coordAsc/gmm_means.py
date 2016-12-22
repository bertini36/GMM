# -*- coding: UTF-8 -*-

import math
import argparse
import numpy as np
import tensorflow as tf
import pickle as pkl
import matplotlib.pyplot as plt

from utils import tf_log_beta_function, tf_dirichlet_expectation

DEBUG = True

def printf(s):
	if DEBUG:
		print(s) 

parser = argparse.ArgumentParser(description='Inference in the gaussian mixture data with unknown means')
parser.add_argument('-maxIter', metavar='maxIter', type=int, default=10)
parser.add_argument('-K', metavar='K', type=int, default=4)
parser.add_argument('-filename', metavar='filename', type=str, default='data_means.pkl')
parser.add_argument('-alpha', metavar='alpha', nargs='+', type=float, default=[1.]*4)
parser.add_argument('-m_o', metavar='m_o', nargs='+', type=float, default=[0., 0.])
parser.add_argument('-beta_o', metavar='beta_o', nargs='+', type=float, default=0.01)
parser.add_argument('-Delta_o', metavar='Delta_o', nargs='+', type=float, default=[1., 0., 0., 1.])
args = parser.parse_args()

# Get data
with open('../../data/' + args.filename, 'r') as inputfile:
	data = pkl.load(inputfile)
xn = tf.convert_to_tensor(data['xn'], dtype=tf.float32)

plt.scatter(data['xn'][:,0], data['xn'][:,1], c=(1.*data['zn'])/max(data['zn']))
plt.show()

# Get params
N, D = data['xn'].shape
printf('N: {}, D: {}'.format(N, D))
K = args.K
alpha_aux = args.alpha
m_o_aux = np.array(args.m_o)
beta_o_aux = args.beta_o
Delta_o_aux = np.array([args.Delta_o[0:D],args.Delta_o[D:2*D]])

alpha = tf.convert_to_tensor([alpha_aux], dtype=tf.float32)
printf('alpha shape: {}'.format(alpha.get_shape()))
m_o = tf.convert_to_tensor([list(m_o_aux)], dtype=tf.float32)
printf('m_o shape: {}'.format(m_o.get_shape()))
beta_o = tf.convert_to_tensor(beta_o_aux, dtype=tf.float32)
printf('beta_o shape: {}'.format(beta_o.get_shape()))
Delta_o = tf.convert_to_tensor(Delta_o_aux, dtype=tf.float32)
printf('Delta_o shape: {}'.format(Delta_o.get_shape()))

# Variable initialization
phi_aux = np.random.dirichlet(alpha_aux, N)
printf('phi_aux: {}'.format(phi_aux))
lambda_pi_aux = alpha_aux + np.sum(phi_aux, axis=0)
printf('lambda_pi_aux: {}'.format(lambda_pi_aux))
lambda_mu_beta_aux = beta_o_aux + np.sum(phi_aux, axis=0)
printf('lambda_mu_beta_aux: {}'.format(lambda_mu_beta_aux))
lambda_mu_m_aux = np.tile(1./lambda_mu_beta_aux, (2, 1)).T * \
				  (beta_o_aux * m_o_aux + np.dot(phi_aux.T, data['xn']))
printf('lambda_mu_m_aux: {}'.format(lambda_mu_m_aux))

phi = tf.Variable(phi_aux, dtype=tf.float32)
printf('phi shape: {}'.format(phi.get_shape()))
lambda_pi = tf.Variable(lambda_pi_aux, dtype=tf.float32)
lambda_pi = tf.reshape(lambda_pi, [4, 1])
printf('lambda_pi shape: {}'.format(lambda_pi.get_shape()))
lambda_mu_beta = tf.Variable(lambda_mu_beta_aux, dtype=tf.float32)
lambda_mu_beta = tf.reshape(lambda_mu_beta, [4, 1])
printf('lambda_mu_beta shape: {}'.format(lambda_mu_beta.get_shape()))
lambda_mu_m = tf.Variable(lambda_mu_m_aux, dtype=tf.float32)
printf('lambda_mu_m shape: {}'.format(lambda_mu_m.get_shape()))

# ELBO computation graph
ELBO = tf.sub(tf_log_beta_function(lambda_pi), tf_log_beta_function(alpha))
ELBO = tf.add(ELBO, 
			  tf.matmul(tf.sub(alpha, lambda_pi), tf_dirichlet_expectation(lambda_pi)))
ELBO = tf.add(ELBO,  
			  tf.mul(K/2., tf.log(tf.matrix_determinant(tf.mul(beta_o, Delta_o)))))
ELBO = tf.add(ELBO, 
			  K*D/2)
for k in xrange(K):


	ELBO = tf.sub(ELBO,
				  tf.mul(tf.div(beta_o, 2.), tf.matmul(tf.sub(lambda_mu_m[k,:], m_o), 
				  tf.matmul(Delta_o, tf.transpose(tf.sub(lambda_mu_m[k,:], m_o))))))
	ELBO = tf.sub(ELBO,
				  tf.div(tf.mul(float(D), beta_o), tf.mul(2., lambda_mu_beta[k])))
	ELBO = tf.sub(ELBO,
				  tf.mul(1/2., tf.log(tf.matrix_determinant(tf.mul(lambda_mu_beta[k], \
				  												   Delta_o)))))

	for n in xrange(N):

		aux1 = tf.add(tf.sub(tf_dirichlet_expectation(lambda_pi)[k], tf.log(phi[n,k])), 
					  tf.mul(1/2., tf.log(tf.div(tf.matrix_determinant(Delta_o), tf.mul(2., math.pi)))))
		a = tf.transpose(tf.sub(xn[n,:], lambda_mu_m[k,:]))
		a = tf.reshape(a, [2, 1])
		b = tf.sub(xn[n,:], lambda_mu_m[k,:])
		b = tf.reshape(b, [1, 2])
		aux2 = tf.mul(1/2.,
					  tf.matmul(b, 
					 		 tf.matmul(Delta_o, a)))
		ELBO = tf.add(ELBO,
					  tf.mul(phi[n,k],
							   tf.sub(tf.sub(aux1, aux2), tf.div(float(D), tf.mul(2., lambda_mu_beta[k])))))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(ELBO)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	printf('ENTRO!!!')
	for epoch in range(10):
		_, elbo = sess.run([train, ELBO])
		printf('ELBO: {}'.format(elbo))