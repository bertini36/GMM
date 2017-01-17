# -*- coding: UTF-8 -*-

import math
import numpy as np
import tensorflow as tf
import pickle as pkl
import matplotlib.pyplot as plt

DEBUG = True
MAX_EPOCHS = 400

np.random.seed(7)
sess = tf.Session()

# Aux functions
def printf(s):
	if DEBUG:
		print(s) 

def tf_log_beta_function(x):
	return tf.sub(tf.reduce_sum(tf.lgamma(tf.add(x, np.finfo(np.float32).eps))), \
				  tf.lgamma(tf.reduce_sum(tf.add(x, np.finfo(np.float32).eps))))

def tf_dirichlet_expectation(alpha):
	if len(alpha.get_shape()) == 1:
		return tf.sub(tf.digamma(tf.add(alpha, np.finfo(np.float32).eps)), \
					  tf.digamma(tf.reduce_sum(alpha)))
	return tf.sub(tf.digamma(alpha), \
				  tf.digamma(tf.reduce_sum(alpha, 1))[:, tf.newaxis])

def tf_exp_normalize(aux):
	return tf.div(tf.add(tf.exp(tf.sub(aux, tf.maximum(aux))), np.finfo(np.float32).eps), \
				  tf.reduce_sum(tf.add(tf.exp(tf.sub(aux, tf.maximum(aux))), np.finfo(np.float32).eps)))

# Get data
with open('../../../data/data_k2_100.pkl', 'r') as inputfile:
	data = pkl.load(inputfile)
	xn = data['xn']

# plt.scatter(xn[:,0],xn[:,1], c=(1.*data['zn'])/max(data['zn']))
# plt.show()

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
Delta_o = tf.convert_to_tensor(Delta_o_aux , dtype=tf.float64)

# Initializations
phi_aux = np.random.dirichlet(alpha_aux, N)
lambda_pi_aux = alpha_aux + np.sum(phi_aux, axis=0)
lambda_mu_beta_aux = beta_o_aux + np.sum(phi_aux, axis=0)
lambda_mu_m_aux = np.tile(1./lambda_mu_beta_aux, (2, 1)).T * \
				(beta_o_aux * m_o_aux + np.dot(phi_aux.T, data['xn']))

# Variational parameters
phi = tf.Variable(phi_aux, dtype=tf.float64)
lambda_pi = tf.Variable(lambda_pi_aux, dtype=tf.float64)
lambda_mu_beta = tf.Variable(lambda_mu_beta_aux, dtype=tf.float64)
lambda_mu_m = tf.Variable(lambda_mu_m_aux, dtype=tf.float64)

xn = tf.convert_to_tensor(xn , dtype=tf.float64)

# Reshapes
lambda_mu_beta_res = tf.reshape(lambda_mu_beta, [K, 1])
lambda_pi_res = tf.reshape(lambda_pi, [K, 1])

# Lower Bound definition
ELBO = tf_log_beta_function(lambda_pi)
ELBO = tf.sub(ELBO, tf_log_beta_function(alpha))
ELBO = tf.add(ELBO, tf.matmul(tf.sub(alpha, lambda_pi), tf.reshape(tf_dirichlet_expectation(lambda_pi), [K,1])))
ELBO = tf.add(ELBO, tf.mul(tf.cast(K/2., tf.float64), tf.log(tf.matrix_determinant(tf.mul(beta_o, Delta_o)))))
ELBO = tf.add(ELBO, tf.cast(K*(D/2.), tf.float64))
for k in range(K):
	a1 = tf.sub(lambda_mu_m[k,:], m_o)
	a2 = tf.matmul(Delta_o, tf.transpose(tf.sub(lambda_mu_m[k,:], m_o)))
	a3 = tf.mul(tf.div(beta_o, 2.), tf.matmul(a1, a2))
	a4 = tf.div(tf.mul(tf.cast(D, tf.float64), beta_o), tf.mul(tf.cast(2., tf.float64), lambda_mu_beta_res[k]))
	a5 = tf.mul(tf.cast(1/2., tf.float64), tf.log(tf.mul(tf.pow(lambda_mu_beta_res[k], 2), tf.matrix_determinant(Delta_o))))
	a6 = tf.add(a3, tf.add(a4, a5))
	ELBO = tf.sub(ELBO, a6)
	b1 = tf.transpose(phi[:,k])
	b2 = tf.sub(tf.cast(0., tf.float64), tf.log(phi[:,k]))
	b3 = tf.mul(tf.cast(1/2., tf.float64), tf.log(tf.div(tf.matrix_determinant(Delta_o), tf.mul(tf.cast(2., tf.float64), math.pi))))
	b4 = tf.sub(xn, lambda_mu_m[k,:])
	b5 = tf.matmul(Delta_o, tf.transpose(tf.sub(xn, lambda_mu_m[k,:])))
	b6 = tf.mul(tf.cast(1/2., tf.float64), tf.pack([tf.matmul(b4, b5)[i,i] for i in range(N)]))
	b7 = tf.div(tf.cast(D, tf.float64), tf.mul(tf.cast(2., tf.float64), lambda_mu_beta[k]))
	b8 = tf.add(b2, tf.sub(b3, tf.sub(b6, b7)))
	ELBO = tf.add(ELBO, tf.reduce_sum(tf.mul(b1, b8)))

init = tf.global_variables_initializer()
sess.run(init)

# Parameter updates
# alpha + np.sum(phi, axis=0)
assign_lambda_pi = lambda_pi.assign(tf.reshape(tf.add(alpha, tf.reduce_sum(phi, 0)), [2,]))
c1 = tf_dirichlet_expectation(lambda_pi)
for k in range(K):
	# c2 = xn[n,:] - lambda_mu_m[k,:]
	c2 = tf.sub(xn, lambda_mu_m[k,:])
	# c3 = np.dot(Delta_o,(xn[n,:]-lambda_mu_m[k,:]).T)
	c3 = tf.matmul(Delta_o, tf.transpose(tf.sub(xn, lambda_mu_m[k,:])))
	# c4 = -1./2*np.dot(c2, c3)
	c4 = tf.mul(tf.cast(-1/2., tf.float64), tf.matmul(c2, c3))
	# c5 = D/(2.*lambda_mu_beta[k])
	c5 = tf.div(tf.cast(D, tf.float64), tf.mul(tf.cast(2., tf.float64), lambda_mu_beta[k]))
	# c1[k] += c4-c5
	print('Shape c1: {}'.format(c1.get_shape()))
	aux = tf.add(c1[k], tf.sub(c4, c5))
	print('Shape aux: {}'.format(aux.get_shape()))
	c1[k] = aux
assign_phi = phi.assign(tf_exp_normalize(c1))
assign_lambda_mu_beta = lambda_mu_beta.assign(tf.add(beta_o, tf.reduce_sum(phi)))
# d1 = np.tile(1./lambda_mu_beta,(2,1)).T
d1 = tf.transpose(t.div(tf.cast(1., tf.float64), lambda_mu_beta))
# d2 = (m_o * beta_o + np.dot(phi.T, xn))
d2 = tf.add(tf.mul(m_o, beta_o), tf.matmul(tf.transpose(phi), xn))
assign_lambda_mu_m = lambda_mu_m.assign(tf.mul(d1, d2))

# Summaries definition
tf.summary.histogram('phi', phi)
tf.summary.histogram('lambda_pi', lambda_pi)
tf.summary.histogram('lambda_mu_m', lambda_mu_m)
tf.summary.histogram('lambda_mu_beta', lambda_mu_beta)
merged = tf.summary.merge_all()
file_writer = tf.summary.FileWriter('/tmp/tensorboard/', tf.get_default_graph())
run_calls = 0

# Main program
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	epoch = 0
	while epoch < MAX_EPOCHS:
		sess.run(assign_lambda_pi)
		sess.run(assign_phi)
		sess.run(assign_lambda_mu_beta)
		sess.run(assign_lambda_mu_m)
		m_mu_out, beta_mu_out, lambda_pi_out, phi_out = sess.run([lambda_mu_m, lambda_mu_beta, lambda_pi, phi])
		mer, lb = sess.run([merged, LB])
		printf('Epoch {}: pi={} mu={} beta={} ELBO={}'.format(epoch, pi_out, mu_out, beta_out, elbo))
		run_calls += 1
		file_writer.add_summary(mer, run_calls)
		if epoch > 0:
			inc = (old_lb-lb)/old_lb*100
			if inc < 1e-8:
				break
		old_lb = lb
		epoch += 1