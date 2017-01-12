# -*- coding: UTF-8 -*-

import numpy as np
import math
from scipy.special import psi, gammaln

N = 100
It = 50
xn = np.random.normal(5, 1, N)
m = 0.
beta = 0.0001
a = 0.001
b = 0.001

a_gamma = np.random.gamma(1, 1, 1)[0]
b_gamma = np.random.gamma(1, 1, 1)[0]
m_mu = np.random.normal(m,(beta)**(-1.),1)[0]
beta_mu = np.random.gamma(a_gamma, b_gamma, 1)[0]

def lowerbound(m_mu, beta_mu, a_gamma, g_gamma):
	LB = 0
	LB += 1./2*np.log(beta/beta_mu)+1./2*(m_mu**2+1./beta_mu)*(beta_mu-beta)-m_mu*(beta_mu*m_mu-beta*m)+1./2*(beta_mu*m_mu**2-beta*m**2)
	LB += a*np.log(b)-a_gamma*np.log(b_gamma)+gammaln(a_gamma)-gammaln(a)+(psi(a_gamma)-np.log(b_gamma))*(a-a_gamma)+a_gamma/b_gamma*(b_gamma-b)
	LB += N/2.*(psi(a_gamma)-np.log(b_gamma))-N/2.*np.log(2*math.pi)-1./2*a_gamma/b_gamma*sum(xn**2)+a_gamma/b_gamma*sum(xn)*m_mu-N/2.*a_gamma/b_gamma*(m_mu**2+1./beta_mu)
	return LB


it = 0
inc = 0.

while ((it<It) & ((it<2) | (inc>1e-10))):
	m_mu = (beta*m+a_gamma/b_gamma*sum(xn))/(beta+N*a_gamma/b_gamma)
	beta_mu = beta+N*a_gamma/b_gamma
	a_gamma = a + N/2.
	b_gamma = b + 1./2*sum(xn**2)-m_mu*sum(xn)+N/2.*(m_mu**2+1./beta_mu)

	lb =  lowerbound(m_mu, beta_mu, a_gamma, b_gamma) 
	
	if it>0:
		inc = 100*(lb_old-lb)/(lb_old)
		print "It: ", it, "Mean: ", m_mu, "Precision: ", a_gamma/b_gamma, "Lowerbound: ", lb, "Increase: ", inc
	else:
		inc = 0.
		print "It: ", it, "Mean: ", m_mu, "Precision: ", a_gamma/b_gamma, "Lowerbound: ", lb
	
	lb_old = lb
	it += 1
