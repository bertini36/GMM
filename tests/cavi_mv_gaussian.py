import numpy as np
import matplotlib.pyplot as plt

N = 100
mu = np.array([1., 1.])
cov = np.eye(2)*0.01
xn = np.random.multivariate_normal(mu, cov, N)
plt.scatter(xn[:,0], xn[:,1])
plt.show()


def ELBO(xn, N, K, alpha, m_o, beta_o, Delta_o, lambda_pi, lambda_mu_m, lambda_mu_beta, phi):

	ELBO = log_beta_function(lambda_pi)-log_beta_function(alpha) \
			+ np.dot(alpha-lambda_pi, dirichlet_expectation(lambda_pi)) \
			+ K/2.*np.log(np.linalg.det(beta_o*Delta_o))+K*D/2.

	for k in range(K):
		ELBO -= beta_o/2.*np.dot((lambda_mu_m[k,:]-m_o),np.dot(Delta_o,(lambda_mu_m[k,:]-m_o).T)) \
				+D*beta_o/(2.*lambda_mu_beta[k])+1/2.*np.log(np.linalg.det(lambda_mu_beta[k]*Delta_o))	
		for n in range(N):
			ELBO += phi[n,k]*(dirichlet_expectation(lambda_pi)[k]-np.log(phi[n,k])+1/2.*np.log(np.linalg.det(Delta_o)/(2.*math.pi))\
					-1/2.*np.dot((xn[n,:]-lambda_mu_m[k,:]),np.dot(Delta_o,(xn[n,:]-lambda_mu_m[k,:]).T))\
					-D/(2.*lambda_mu_beta[k]))
	return ELBO

def ELBO(xn, N, K, alpha, m_o, beta_o, Delta_o, lambda_pi, lambda_mu_m, lambda_mu_beta, phi):

	ELBO = log_beta_function(lambda_pi)-log_beta_function(alpha) \
		+ np.dot(alpha-lambda_pi, dirichlet_expectation(lambda_pi)) \
		+ K/2.*np.log(np.linalg.det(beta_o*Delta_o))+K*D/2.

	for k in range(K):
		ELBO -= beta_o/2.*np.dot((lambda_mu_m[k,:]-m_o),np.dot(Delta_o,(lambda_mu_m[k,:]-m_o).T)) \
				+D*beta_o/(2.*lambda_mu_beta[k])+1/2.*np.log(np.linalg.det(lambda_mu_beta[k]*Delta_o))	

		ELBO += np.dot(phi[:,k].T,(-np.log(phi[:,k])+1/2.*np.log(np.linalg.det(Delta_o)/(2.*math.pi))\
					-1/2.*np.dot((xn-lambda_mu_m[k,:]),np.dot(Delta_o,(xn-lambda_mu_m[k,:]).T))\
					-D/(2.*lambda_mu_beta[k])))
	return ELBO
