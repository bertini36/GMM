# -*- coding: UTF-8 -*-

import math
import numpy as np
from scipy.special import psi, gammaln

N = 100
np.random.seed(7)
xn = np.random.normal(5, 1, N)

m = 0.
beta = 0.0001
a = 0.001
b = 0.001

a_gamma = 1.
b_gamma = 1.
m_mu = 10.
beta_mu = 1.


def lowerbound(m_mu, beta_mu, a_gamma):
    LB = 1. / 2 * np.log(beta / beta_mu)
    LB += 1. / 2 * (m_mu ** 2 + 1. / beta_mu) * (beta_mu - beta)
    LB -= m_mu * (beta_mu * m_mu - beta * m)
    LB += 1. / 2 * (beta_mu * m_mu ** 2 - beta * m ** 2)

    LB += a * np.log(b)
    LB -= a_gamma * np.log(b_gamma)
    LB += gammaln(a_gamma)
    LB -= gammaln(a)
    LB += (psi(a_gamma) - np.log(b_gamma)) * (a - a_gamma)
    LB += a_gamma / b_gamma * (b_gamma - b)

    LB += N / 2. * (psi(a_gamma) - np.log(b_gamma))
    LB -= N / 2. * np.log(2 * math.pi)
    LB -= 1. / 2 * a_gamma / b_gamma * sum(xn ** 2)
    LB += a_gamma / b_gamma * sum(xn) * m_mu
    LB -= N / 2. * a_gamma / b_gamma * (m_mu ** 2 + 1. / beta_mu)

    return LB


lb = lowerbound(m_mu, beta_mu, a_gamma)
print "Mean: ", m_mu, "Precision: ", a_gamma / b_gamma, "Lowerbound: ", lb
