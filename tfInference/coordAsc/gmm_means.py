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
parser.add_argument('-K', metavar='K', type=int, default=2)
parser.add_argument('-filename', metavar='filename', type=str, default='data_k2_100.pkl')
parser.add_argument('-alpha', metavar='alpha', nargs='+', type=float, default=[1.]*2)
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
K = args.K
alpha_aux = args.alpha
m_o_aux = np.array(args.m_o)
beta_o_aux = args.beta_o
Delta_o_aux = np.array([args.Delta_o[0:D],args.Delta_o[D:2*D]])

printf('Gaussian Mixture Model')
printf('N: {}, D: {}'.format(N, D))

# Castings
alpha = tf.convert_to_tensor([alpha_aux], dtype=tf.float32)
m_o = tf.convert_to_tensor([list(m_o_aux)], dtype=tf.float32)
beta_o = tf.convert_to_tensor(beta_o_aux, dtype=tf.float32)
Delta_o = tf.convert_to_tensor(Delta_o_aux, dtype=tf.float32)

# Variable initialization
phi_aux = np.random.dirichlet(alpha_aux, N)
lambda_pi_aux = alpha_aux + np.sum(phi_aux, axis=0)
lambda_mu_beta_aux = beta_o_aux + np.sum(phi_aux, axis=0)
lambda_mu_m_aux = np.tile(1./lambda_mu_beta_aux, (2, 1)).T * \
				  (beta_o_aux * m_o_aux + np.dot(phi_aux.T, data['xn']))
phi = tf.Variable(phi_aux, dtype=tf.float32)
lambda_pi = tf.Variable(lambda_pi_aux, dtype=tf.float32)
lambda_pi_res = tf.reshape(lambda_pi, [K, 1])
lambda_mu_beta = tf.Variable(lambda_mu_beta_aux, dtype=tf.float32)
lambda_mu_beta_res = tf.reshape(lambda_mu_beta, [K, 1])
lambda_mu_m = tf.Variable(lambda_mu_m_aux, dtype=tf.float32)

# ELBO computation graph
ELBO = tf.sub(tf_log_beta_function(lambda_pi_res), tf_log_beta_function(alpha))
ELBO = tf.add(ELBO, 
			  tf.matmul(tf.sub(alpha, lambda_pi_res), 
			  			tf_dirichlet_expectation(lambda_pi_res)))
ELBO = tf.add(ELBO,  
			  tf.mul(K/2., tf.log(tf.matrix_determinant(tf.mul(beta_o, Delta_o)))))
ELBO = tf.add(ELBO, K*(D/2.))
for k in xrange(K):
	ELBO = tf.sub(ELBO,
				  tf.mul(tf.div(beta_o, 2.), tf.matmul(tf.sub(lambda_mu_m[k,:], m_o), 
				  tf.matmul(Delta_o, tf.transpose(tf.sub(lambda_mu_m[k,:], m_o))))))
	ELBO = tf.sub(ELBO,
				  tf.div(tf.mul(tf.cast(D, dtype=tf.float32), beta_o), tf.mul(2., lambda_mu_beta_res[k])))
	r = tf.mul(lambda_mu_beta_res[k], Delta_o)
	printf('Shape r: {}'.format(r.get_shape()))
	ELBO = tf.sub(ELBO,
				  tf.mul(1/2., tf.log(tf.matrix_determinant(r))))
	for n in xrange(N):
		printf('Iter: {}'.format((k*n)+n))
		aux1 = tf.add(tf.sub(tf_dirichlet_expectation(lambda_pi_res)[k], tf.log(phi[n,k])), 
					  tf.mul(1/2.,
					  		 tf.log(tf.div(tf.matrix_determinant(Delta_o), 2.*math.pi))))
		aux2 = tf.mul(1/2.,
					  tf.matmul(tf.reshape(tf.sub(xn[n,:], lambda_mu_m[k,:]), [1, 2]), 
					 		 	tf.matmul(Delta_o, 
								  		  tf.reshape(tf.transpose(tf.sub(xn[n,:], lambda_mu_m[k,:])), [2, 1]))))
		ELBO = tf.add(ELBO,
					  tf.mul(phi[n,k],
							 tf.sub(tf.sub(aux1, aux2),
							 		tf.div(tf.cast(D, dtype=tf.float32), tf.mul(2., lambda_mu_beta_res[k])))))
ELBO = tf.sub(0., ELBO)

printf('He acabado de definir el grafo')

# Optimizer definition
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(ELBO, var_list=[lambda_pi, phi, lambda_mu_m, lambda_mu_beta])

# Main program
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	printf('ENTRO!!!')
	for epoch in range(10):
		sess.run([train])