import autograd.numpy as np
import autograd.scipy as scipy
from autograd import grad, elementwise_grad
import math

from scipy.stats import wishart
from numpy.random import multivariate_normal, gamma, normal

def log_Normal((qmu, qsigma)):        
	return -1./2*np.log(np.linalg.det(2*math.pi*qsigma))-1./2*np.dot((x-qmu),np.dot(np.linalg.inv(qsigma),(x-qmu).T))

def fisher_pi():
	Fisher = np.zeros((d+2*d,d+2*d))
	for s in xrange(S):
		q_delta = wishart.rvs(df=nu, scale=W, size=1)
		q_mu = multivariate_normal(mu, np.linalg.inv(beta*q_delta), size=1)[0]
		global x 
		x = multivariate_normal(q_mu, np.linalg.inv(q_delta), size=1)
		score_gradient = elementwise_grad(log_Normal)
		aux = score_gradient((q_mu, np.linalg.inv(q_delta)))
		tmp = np.concatenate((aux[0].flatten(),aux[1].flatten()))
		Fisher = Fisher + np.outer(tmp, tmp)
	return Fisher/S

S = 1000
d = 2

nu = 10
W = np.array([[0.01, 0.0], [0.0, 0.01]])
mu =  np.array([0.0, 0.0])
beta = 0.1
fisher_pi()



def log_Normal((qmu, qsigma)):
	return -1./2*np.log(2*math.pi*np.prod(qsigma))-1./2*np.dot((x-qmu),np.dot(np.diag(1./qsigma**2),(x-qmu).T))

def fisher():
	Fisher = np.zeros((d+d,d+d))
	qmu =  np.array([0.0, 0.0]) #multivariate_normal(mu, np.diag(1./(beta*q_delta)), size=1)[0]
	qsigma = np.array([0.1, 0.1]) #np.array([np.random.gamma(a, b), np.random.gamma(a, b)])
	for s in xrange(S): 
		global x
		x = multivariate_normal(np.zeros(2), np.diag(np.array([0.1, 0.1])), size=1)[0]
		score_gradient = elementwise_grad(log_Normal)
		aux = score_gradient((qmu, qsigma))
		tmp = np.concatenate((aux[0].flatten(),aux[1].flatten()))
		Fisher = Fisher + np.outer(tmp, tmp)
	return Fisher/S


S = 1000
d = 2

Fisher = fisher()
print Fisher


def log_Normal((qmu, qsigma)):
	return -1./2*np.log(2.*math.pi*qsigma)-1./(2*qsigma**2)*(x-qmu)**2

def fisher():
	Fisher = np.zeros((2,2))
	for s in xrange(S):
		q_delta = np.random.gamma(a, b)
		q_mu = normal(mu, 1./(beta*q_delta))
		global x
		x = normal(q_mu,q_delta,size=1)
		score_gradient = elementwise_grad(log_Normal)
		aux = score_gradient((q_mu, q_delta))
		tmp = np.concatenate((aux[0].flatten(),aux[1].flatten()))
		Fisher = Fisher + np.outer(tmp, tmp)
	return Fisher/S


S = 1000
a = 2.
b = 2.
mu =  0.0
beta = 0.1

fisher()