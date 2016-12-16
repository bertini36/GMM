import autograd.numpy as np
from autograd import grad, elementwise_grad
import argparse
import matplotlib.pyplot as plt
import math
import pickle as pkl
from utils import ag_dirichlet_expectation, ag_log_beta_function

machineprecion = 2.2204460492503131e-16

def initialize():
	phi = np.random.dirichlet(alpha, N)
	lambda_pi = alpha + np.sum(phi, axis=0)
	lambda_mu_beta = beta_o + np.sum(phi, axis=0)
	lambda_mu_m = np.tile(1./lambda_mu_beta,(2,1)).T * (beta_o * m_o + np.dot(phi.T, xn))
	return lambda_pi, phi, lambda_mu_m, lambda_mu_beta

def ELBO((lambda_pi, phi, lambda_mu_m, lambda_mu_beta)):

	lambda_pi_aux = np.log(1 + np.exp(lambda_pi) + machineprecion) 
	phi_aux = (np.exp(phi)+ machineprecion)/np.tile((np.exp(phi)+machineprecion).sum(axis=1),(K,1)).T
	lambda_mu_beta_aux = np.log(1 + np.exp(lambda_mu_beta) + machineprecion)
	ELBO = ag_log_beta_function(lambda_pi_aux)-ag_log_beta_function(alpha) \
			+ np.dot(alpha-lambda_pi_aux, ag_dirichlet_expectation(lambda_pi_aux)) \
			+ K/2.*np.log(np.linalg.det(beta_o*Delta_o))+K*D/2.

	for k in xrange(K):
		ELBO = ELBO - (beta_o/2.*np.dot((lambda_mu_m[k,:]-m_o),np.dot(Delta_o,(lambda_mu_m[k,:]-m_o).T)) \
				+ D*beta_o/(2.*lambda_mu_beta_aux[k])+1/2.*np.log(lambda_mu_beta_aux[k]))
		for n in xrange(N):
			ELBO = ELBO + (phi_aux[n,k]*(ag_dirichlet_expectation(lambda_pi_aux)[k]-np.log(phi_aux[n,k])+1/2.*np.log(1./(2.*math.pi))\
					-1/2.*np.dot((xn[n,:]-lambda_mu_m[k,:]),np.dot(Delta_o,(xn[n,:]-lambda_mu_m[k,:]).T))\
					-D/(2.*lambda_mu_beta_aux[k])))
	return -ELBO


def ELBO1((lambda_pi, phi)):

	lambda_pi_aux = np.log(1 + np.exp(lambda_pi) + machineprecion) 
	phi_aux = (np.exp(phi)+ machineprecion)/np.tile((np.exp(phi)+machineprecion).sum(axis=1),(K,1)).T
	lambda_mu_beta_aux = np.log(1 + np.exp(lambda_mu_beta) + machineprecion)
	ELBO = ag_log_beta_function(lambda_pi_aux)-ag_log_beta_function(alpha) \
			+ np.dot(alpha-lambda_pi_aux, ag_dirichlet_expectation(lambda_pi_aux)) \
			+ K/2.*np.log(np.linalg.det(beta_o*Delta_o))+K*D/2.

	for k in xrange(K):
		ELBO = ELBO - (beta_o/2.*np.dot((lambda_mu_m[k,:]-m_o),np.dot(Delta_o,(lambda_mu_m[k,:]-m_o).T)) \
				+ D*beta_o/(2.*lambda_mu_beta_aux[k])+1/2.*np.log(np.linalg.det(lambda_mu_beta_aux[k]*Delta_o)))
		for n in xrange(N):
			ELBO = ELBO +(phi_aux[n,k]*(ag_dirichlet_expectation(lambda_pi_aux)[k]-np.log(phi_aux[n,k])+1/2.*np.log(np.linalg.det(Delta_o)/(2.*math.pi))\
					-1/2.*np.dot((xn[n,:]-lambda_mu_m[k,:]),np.dot(Delta_o,(xn[n,:]-lambda_mu_m[k,:]).T))\
					-D/(2.*lambda_mu_beta_aux[k])))
	return -ELBO


def ELBO2((lambda_mu_m,lambda_mu_beta )):

	lambda_pi_aux = np.log(1 + np.exp(lambda_pi) + machineprecion) 
	phi_aux = (np.exp(phi)+ machineprecion)/np.tile((np.exp(phi)+machineprecion).sum(axis=1),(K,1)).T
	lambda_mu_beta_aux = np.log(1 + np.exp(lambda_mu_beta) + machineprecion)
	ELBO = ag_log_beta_function(lambda_pi_aux)-ag_log_beta_function(alpha) \
			+ np.dot(alpha-lambda_pi_aux, ag_dirichlet_expectation(lambda_pi_aux)) \
			+ K/2.*np.log(np.linalg.det(beta_o*Delta_o))+K*D/2.

	for k in xrange(K):
		ELBO = ELBO - (beta_o/2.*np.dot((lambda_mu_m[k,:]-m_o),np.dot(Delta_o,(lambda_mu_m[k,:]-m_o).T)) \
				+ D*beta_o/(2.*lambda_mu_beta_aux[k])+1/2.*np.log(np.linalg.det(lambda_mu_beta_aux[k]*Delta_o)))
		for n in xrange(N):
			ELBO = ELBO +(phi_aux[n,k]*(ag_dirichlet_expectation(lambda_pi_aux)[k]-np.log(phi_aux[n,k])+1/2.*np.log(np.linalg.det(Delta_o)/(2.*math.pi))\
					-1/2.*np.dot((xn[n,:]-lambda_mu_m[k,:]),np.dot(Delta_o,(xn[n,:]-lambda_mu_m[k,:]).T))\
					-D/(2.*lambda_mu_beta_aux[k])))
	return -ELBO


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Inference in the gaussian mixture data with unknown means')
	parser.add_argument('-maxIter', metavar='maxIter', type=int, default=100)
	parser.add_argument('-K', metavar='K', type=int, default=2)
	parser.add_argument('-filename', metavar='filename', type=str, default="data_means.pkl")
	parser.add_argument('-alpha', metavar='alpha', nargs='+', type=float, default=[1.]*2)
	parser.add_argument('-m_o', metavar='m_o', nargs='+', type=float, default=[0., 0.])
	parser.add_argument('-beta_o', metavar='beta_o', nargs='+', type=float, default=0.01)
	parser.add_argument('-Delta_o', metavar='Delta_o', nargs='+', type=float, default=[1., 0., 0., 1.])
	args = parser.parse_args()

	with open('data/' + 'data_means.pkl', 'r') as inputfile:
		data = pkl.load(inputfile)
	xn = data['xn']

	plt.scatter(xn[:,0],xn[:,1], c=(1.*data['zn'])/max(data['zn']))
	plt.show()

	N, D = xn.shape
	K = args.K
	alpha = args.alpha
	m_o = np.array(args.m_o)
	beta_o = args.beta_o
	Delta_o = np.array([args.Delta_o[0:D],args.Delta_o[D:2*D]]) 


	lambda_pi, phi, lambda_mu_m, lambda_mu_beta = initialize()

	#To fix the means and betas so that the algorithm has only to find the proportions
	#lambda_mu_m = np.array([[-20, 0],[2.5,-25],[2,-8],[0, 15]])
	#lambda_mu_beta = np.array([0.01, 0.01, 0.01, 0.01])
	
	# To Fix 
	#lambda_pi = np.array([250., 250., 250., 250.])
	#phi = np.ones((N,K))
	#for n in xrange(N):
	#	phi[n,data['zn'][n]] *=10.

	#print phi
	print "ELBO: ", ELBO((lambda_pi, phi, lambda_mu_m, lambda_mu_beta))

	for i in xrange(args.maxIter):
		aux = elementwise_grad(ELBO1)((lambda_pi, phi))
		lambda_pi -= 0.1*aux[0]
		phi -= 0.1*aux[1]

		aux = elementwise_grad(ELBO2)((lambda_mu_m, lambda_mu_beta ))
		lambda_mu_m  -= 0.001*aux[0]
		lambda_mu_beta -= 0.1*aux[1]

		print  "It: ", i, "ELBO: ", ELBO((lambda_pi, phi, lambda_mu_m, lambda_mu_beta))
	
	aux = (np.exp(phi)+ machineprecion)/np.tile((np.exp(phi)+machineprecion).sum(axis=1),(K,1)).T
	plt.scatter(xn[:,0], xn[:,1], c=np.array(1*[np.random.choice(K, 1, p=aux[n,:])[0] for n in xrange(N)]))
	plt.show()

