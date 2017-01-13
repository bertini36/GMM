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
Delta_o_aux = np.array([args.Delta_o[0:D], args.Delta_o[D:2*D]])

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

# Optimization variables
phi = tf.Variable(phi_aux, dtype=tf.float32)
lambda_pi = tf.Variable(lambda_pi_aux, dtype=tf.float32)
lambda_mu_beta = tf.Variable(lambda_mu_beta_aux, dtype=tf.float32)
lambda_mu_m = tf.Variable(lambda_mu_m_aux, dtype=tf.float32)

# Reshapes
lambda_mu_beta_res = tf.reshape(lambda_mu_beta, [K, 1])
lambda_pi_res = tf.reshape(lambda_pi, [K, 1])

# ELBO computation graph
q1 = tf_log_beta_function(lambda_pi_res)																			# [2, 1] 
q2 = tf_log_beta_function(alpha)																					# [1, 2]
q3 = tf.matmul(tf.sub(alpha, lambda_pi_res), tf_dirichlet_expectation(lambda_pi_res))								# [2, 1]
q4 = tf.mul(K/2., tf.log(tf.matrix_determinant(tf.mul(beta_o, Delta_o))))											# Scalar

ELBO1 = tf.sub(q1, tf.add(q2, tf.add(q3, tf.add(q4, K*(D/2.)))))														# [2, 2]

for k in xrange(K):
	s1 = tf.mul(tf.div(beta_o, 2.), tf.matmul(tf.sub(lambda_mu_m[k,:], m_o), 										# [1, 1]
				  		 					  tf.matmul(Delta_o, tf.transpose(tf.sub(lambda_mu_m[k,:], m_o)))))
	s2 = tf.div(tf.mul(float(D), beta_o), tf.mul(2., lambda_mu_beta_res[k]))										# [1, ]
	s3 = tf.mul(1/2., tf.log(tf.mul(tf.pow(lambda_mu_beta_res[k], 2), tf.matrix_determinant(Delta_o))))				# [1, ]

	ELBO2 = tf.sub(ELBO1, tf.add(s1, tf.add(s2, s3)))

	r1 = tf.reshape(tf.transpose(phi[:,k]), [1,N])																	# [N, 1]
	r2 = tf.reshape(tf.sub(0., tf.log(phi[:,k])), [N,1])															# [N, 1]
	r3 = tf.mul(1/2., tf.log(tf.div(tf.matrix_determinant(Delta_o), 2.*math.pi)))									# Scalar
	r4 = tf.mul(1/2., tf.reshape(tf.pack([tf.matmul(tf.sub(xn, lambda_mu_m[k,:]),								
										  tf.matmul(Delta_o, tf.transpose(tf.sub(xn, lambda_mu_m[k,:]))))[i, i] for i in range(N)]), [N,1]))
	r5 = tf.div(float(D), tf.mul(2., lambda_mu_beta[k]))															# Scalar

	ELBO3 = tf.add(ELBO2, tf.matmul(r1, tf.add(r2, tf.sub(r3, tf.sub(r4, r5)))))

# Optimizer definition
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(ELBO3, var_list=[lambda_pi, phi, lambda_mu_m, lambda_mu_beta])

# Summaries definition
if DEBUG:
	tf.summary.histogram('phi', phi)
	tf.summary.histogram('lambda_pi', lambda_pi)
	tf.summary.histogram('lambda_mu_m', lambda_mu_m)
	tf.summary.histogram('lambda_mu_beta', lambda_mu_beta)
	merged = tf.summary.merge_all()
	file_writer = tf.summary.FileWriter('/tmp/tensorboard/', \
										tf.get_default_graph())
	run_calls = 0

# Main program
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	for epoch in range(5):
		_, m, elbo1, elbo2, elbo3, r_1, r_2, r_3, r_4, r_5 = sess.run([train, merged, ELBO1, ELBO2, ELBO3, r1, r2, r3, r4, r5])
		if DEBUG:
			run_calls += 1
			file_writer.add_summary(m, run_calls)
		printf('r1: {}'.format(r_1))
		printf('r2: {}'.format(r_2))
		printf('r3: {}'.format(r_3))
		printf('r4: {}'.format(r_4))
		printf('r5: {}'.format(r_5))
		printf('ELBO1: {}'.format(elbo1))
		printf('ELBO2: {}'.format(elbo2))
		printf('ELBO3: {}'.format(elbo3))