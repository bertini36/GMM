# -*- coding: UTF-8 -*-

import math
import numpy as np
import tensorflow as tf
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

DEBUG = True
MAX_EPOCHS = 100
DATASET = 'data_k2_100.pkl'
K = 2
THRESHOLD =  1e-6

# np.random.seed(7)

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
	return tf.div(tf.add(tf.exp(tf.sub(aux, tf.reduce_max(aux))), np.finfo(np.float32).eps), \
				  tf.reduce_sum(tf.add(tf.exp(tf.sub(aux, tf.reduce_max(aux))), np.finfo(np.float32).eps)))

# Get data
with open('../../../data/{}'.format(DATASET), 'r') as inputfile:
	data = pkl.load(inputfile)
	xn = data['xn']
	xn_tf = tf.convert_to_tensor(xn , dtype=tf.float64)

plt.scatter(xn[:,0],xn[:,1], c=(1.*data['zn'])/max(data['zn']), cmap=cm.bwr)
plt.show()

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
				(beta_o_aux * m_o_aux + np.dot(phi_aux.T, xn))

# Variational parameters
phi_var = tf.Variable(phi_aux, dtype=tf.float64, name='phi_var')
lambda_pi_var = tf.Variable(lambda_pi_aux, dtype=tf.float64, name='lambda_pi_var')
lambda_mu_beta_var = tf.Variable(lambda_mu_beta_aux, dtype=tf.float64, name='lambda_mu_beta_var')
lambda_mu_m = tf.Variable(lambda_mu_m_aux, dtype=tf.float64, name='lambda_mu_m')

# Maintain numerical stability
lambda_pi = tf.nn.softplus(lambda_pi_var)
lambda_mu_beta  = tf.nn.softplus(lambda_mu_beta_var)
phi = tf.nn.softmax(phi_var)

# Reshapes
lambda_mu_beta_res = tf.reshape(lambda_mu_beta, [K, 1])

# Lower Bound definition
LB = tf_log_beta_function(lambda_pi)
LB = tf.sub(LB, tf_log_beta_function(alpha))
LB = tf.add(LB, tf.matmul(tf.sub(alpha, lambda_pi), tf.reshape(tf_dirichlet_expectation(lambda_pi), [K,1])))
LB = tf.add(LB, tf.mul(tf.cast(K/2., tf.float64), tf.log(tf.matrix_determinant(tf.mul(beta_o, Delta_o)))))
LB = tf.add(LB, tf.cast(K*(D/2.), tf.float64))
for k in range(K):
	a1 = tf.sub(lambda_mu_m[k,:], m_o)
	a2 = tf.matmul(Delta_o, tf.transpose(tf.sub(lambda_mu_m[k,:], m_o)))
	a3 = tf.mul(tf.div(beta_o, 2.), tf.matmul(a1, a2))
	a4 = tf.div(tf.mul(tf.cast(D, tf.float64), beta_o), tf.mul(tf.cast(2., tf.float64), lambda_mu_beta_res[k]))
	a5 = tf.mul(tf.cast(1/2., tf.float64), tf.log(tf.mul(tf.pow(lambda_mu_beta_res[k], 2), tf.matrix_determinant(Delta_o))))
	a6 = tf.add(a3, tf.add(a4, a5))
	LB = tf.sub(LB, a6)
	b1 = tf.transpose(phi[:,k])
	b2 = tf_dirichlet_expectation(lambda_pi)[k]
	b3 = tf.log(phi[:,k])
	b4 = tf.mul(tf.cast(1/2., tf.float64), tf.log(tf.div(tf.matrix_determinant(Delta_o), tf.mul(tf.cast(2., tf.float64), math.pi))))
	b5 = tf.sub(xn_tf, lambda_mu_m[k,:])
	b6 = tf.matmul(Delta_o, tf.transpose(tf.sub(xn_tf, lambda_mu_m[k,:])))
	b7 = tf.mul(tf.cast(1/2., tf.float64), tf.pack([tf.matmul(b5, b6)[i,i] for i in range(N)]))
	b8 = tf.div(tf.cast(D, tf.float64), tf.mul(tf.cast(2., tf.float64), lambda_mu_beta[k]))
	b9 = tf.sub(tf.sub(tf.add(tf.sub(b2, b3), b4), b7), b8)
	b1 = tf.reshape(b1, [1,N])
	b9 = tf.reshape(b9, [N,1])
	LB = tf.add(LB, tf.reshape(tf.matmul(b1, b9), [1]))

def compute_learning_rate(var, alpha_step):
	"""
	:param var: Var to optimize
	:param alpha_step: Initial learning rate
	"""
	# Obtaining the gradients
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha_step)
	grads_and_vars = optimizer.compute_gradients(-LB, var_list=[var])
	grads = sess.run(grads_and_vars)
	tmp_var = grads[0][1]
	tmp_grad = grads[0][0]

	# Gradient descent update
	fx = sess.run(-LB)
	tmp_mod = tmp_var - alpha_step * tmp_grad
	assign_op = var.assign(tmp_mod)
	sess.run(assign_op)
	fxgrad = sess.run(-LB)

	# Loop for problematic vars that produces Infs and Nans
	while np.isinf(fxgrad) or np.isnan(fxgrad):
		alpha_step /= 10.
		tmp_mod = tmp_var - alpha_step * tmp_grad
		assign_op = var.assign(tmp_mod)
		sess.run(assign_op)
		fxgrad = sess.run(-LB)

	m = tmp_grad**2
	c = 0.5
	tau = 0.2

	# The values update depart from the variable dimensions
	if var.name == 'lambda_pi_var:0' or var.name == 'lambda_mu_beta_var:0':
		for i in xrange(len(m)):
			while (fxgrad >= fx-alpha_step*c*m[i]):
				alpha_step *= tau
				tmp_mod = tmp_var - alpha_step * tmp_grad
				assign_op = var.assign(tmp_mod)
				sess.run(assign_op)
				fxgrad = sess.run(-LB)
				if alpha_step < 1e-10:
					alpha_step = 0
					break
	elif var.name == 'phi_var:0' or var.name == 'lambda_mu_m:0':
		for i in xrange(len(m)):
			for j in xrange(len(m[0])):
				while (fxgrad >= fx-alpha_step*c*m[i,j]):
					alpha_step *= tau
					tmp_mod = tmp_var - alpha_step * tmp_grad
					assign_op = var.assign(tmp_mod)
					sess.run(assign_op)
					fxgrad = sess.run(-LB)
					if alpha_step < 1e-10:
						alpha_step = 0
						break


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
	alpha_step = 1e10
	for epoch in xrange(MAX_EPOCHS):

		# Parameter updates with individual learning rates
		compute_learning_rate(lambda_pi_var, alpha_step)
		compute_learning_rate(phi_var, alpha_step)
		compute_learning_rate(lambda_mu_m, alpha_step)
		compute_learning_rate(lambda_mu_beta_var, alpha_step)
		
		# ELBO computation
		mer, lb, pi_out, phi_out, mu_out, beta_out = sess.run([merged, LB, lambda_pi, phi, lambda_mu_m, lambda_mu_beta])
		print('Epoch {}: Mus={} Precision={} Pi={} ELBO={}'.format(epoch, mu_out, beta_out, pi_out, lb))
		run_calls += 1
		file_writer.add_summary(mer, run_calls)

		# Break condition
		if epoch > 0: 
			if abs(lb-old_lb) < THRESHOLD:
				break
		old_lb = lb

	plt.scatter(xn[:,0], xn[:,1], c=np.array(1*[np.random.choice(K, 1, p=phi_out[n,:])[0] for n in xrange(N)]), cmap=cm.bwr)
	plt.show()