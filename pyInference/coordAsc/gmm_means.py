import numpy as np
import pickle as pkl
import argparse
import matplotlib.pyplot as plt
import math
from utils import dirichlet_expectation, exp_normalize, log_beta_function

def initialize(xn, K, alpha, m_o, beta_o, Delta_o):
	N, D = xn.shape
	phi = np.random.dirichlet(alpha, N)
	lambda_pi = alpha + np.sum(phi, axis=0)
	lambda_mu_beta = beta_o + np.sum(phi, axis=0)
	lambda_mu_m = np.tile(1./lambda_mu_beta,(2,1)).T * (beta_o * m_o + np.dot(phi.T, xn))
	return lambda_pi, phi, lambda_mu_m, lambda_mu_beta

def ELBO(xn, N, K, alpha, m_o, beta_o, Delta_o, lambda_pi, lambda_mu_m, lambda_mu_beta, phi):

	ELBO = log_beta_function(lambda_pi)-log_beta_function(alpha) \
			+ np.dot(alpha-lambda_pi, dirichlet_expectation(lambda_pi)) \
			+ K/2.*np.log(np.linalg.det(beta_o*Delta_o))+K*D/2.

	for k in xrange(K):
		ELBO -= beta_o/2.*np.dot((lambda_mu_m[k,:]-m_o),np.dot(Delta_o,(lambda_mu_m[k,:]-m_o).T)) \
				+D*beta_o/(2.*lambda_mu_beta[k])+1/2.*np.log(np.linalg.det(lambda_mu_beta[k]*Delta_o))	
		for n in xrange(N):
			ELBO += phi[n,k]*(dirichlet_expectation(lambda_pi)[k]-np.log(phi[n,k])+1/2.*np.log(np.linalg.det(Delta_o)/(2.*math.pi))\
					-1/2.*np.dot((xn[n,:]-lambda_mu_m[k,:]),np.dot(Delta_o,(xn[n,:]-lambda_mu_m[k,:]).T))\
					-D/(2.*lambda_mu_beta[k]))
	return ELBO


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

	with open('data/' + args.filename, 'r') as inputfile:
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

	lambda_pi, phi, lambda_mu_m, lambda_mu_beta = initialize(xn, K, alpha, m_o, beta_o, Delta_o)

	elbos = []
	for it in xrange(args.maxIter):
		lambda_pi = alpha + np.sum(phi, axis=0)

		Elogpi = dirichlet_expectation(lambda_pi)
		for n in xrange(N):
			aux = np.copy(Elogpi)
			for k in xrange(K):
				aux[k] += -1./2*np.dot((xn[n,:]-lambda_mu_m[k,:]),np.dot(Delta_o,(xn[n,:]-lambda_mu_m[k,:]).T))-D/(2.*lambda_mu_beta[k])
			phi[n,:] = exp_normalize(aux)
		
		Nk = np.sum(phi, axis=0)		
		lambda_mu_beta = beta_o + Nk
		lambda_mu_m = np.tile(1./lambda_mu_beta,(2,1)).T * (m_o * beta_o + np.dot(phi.T, xn))

		aux = ELBO(xn, N, K, alpha, m_o, beta_o, Delta_o, lambda_pi, lambda_mu_m, lambda_mu_beta, phi)
		print "It: ", it, "ELBO: ", aux, "Incr: ", (aux-elbos[-1])/abs(aux) if it>0. else " "
		if it>0: 
			if (aux-elbos[-1])/abs(aux)<1e-6:
				elbos.append(aux)
				break

		elbos.append(aux)

	plt.scatter(xn[:,0], xn[:,1], c=np.array(1*[np.random.choice(K, 1, p=phi[n,:])[0] for n in xrange(N)]))
	plt.show()


