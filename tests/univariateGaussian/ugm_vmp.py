# -*- coding: UTF-8 -*-

import numpy as np
import math
from scipy.special import psi, gammaln

N = 2
It = 50
xn = np.random.normal(5, 1, N)
m = 0.
beta = 0.0001
a = 0.001
b = 0.001

a_gamma = np.random.gamma(1, 1, 1)[0]
b_gamma = np.random.gamma(1, 1, 1)[0]
m_mu = np.random.normal(m, (beta) ** (-1.), 1)[0]
beta_mu = np.random.gamma(a_gamma, b_gamma, 1)[0]

m_gx = np.zeros(2)
m_xm = np.zeros((N, 2))
m_mx = np.zeros(2)
m_xg = np.zeros((N, 2))


def lowerbound(m_gx, m_xm, m_mx, m_xg, phi_m, phi_g):
    LB = 0
    m_mu = phi_m[0] / (-2 * phi_m[1])
    beta_mu = -2 * phi_m[1]
    a_gamma = phi_g[1] + 1
    b_gamma = -phi_g[0]

    for n in range(N):
        LB += np.dot(np.array([m_gx[0] * m_mx[0], -m_gx[0] / 2.]).T,
                     np.array([xn[n], xn[n] ** 2]))
    LB += N / 2. * (m_gx[1] - m_gx[0] * m_mx[1] - np.log(2 * math.pi))

    LB += np.dot((phi_mu - phi_m).T, m_mx) + 1. / 2 * (
    np.log(beta) - beta * m ** 2 + beta_mu * m_mu ** 2 - np.log(beta_mu))
    LB += np.dot((phi_gamma - phi_g).T, m_gx) + a * np.log(b) - gammaln(
        a) - a_gamma * np.log(b_gamma) + gammaln(a_gamma)

    return LB


phi_mu = np.array([beta * m, -beta / 2.])
phi_gamma = np.array([-b, a - 1])

it = 0
inc = 0.

while (it < It) & ((it < 2) | (inc > 1e-10)):
    m_gx = np.array([a_gamma / b_gamma, psi(a_gamma) - np.log(b_gamma)])
    for n in range(N):
        m_xm[n, :] = np.array([xn[n] * m_gx[0], -m_gx[0] / 2.])
    phi_m = phi_mu + np.sum(m_xm, axis=0)
    m_mu = phi_m[0] / (-2 * phi_m[1])
    beta_mu = -2 * phi_m[1]
    m_mx = np.array([m_mu, 1. / beta_mu + m_mu ** 2])
    for n in range(N):
        m_xg[n, :] = np.array(
            [-1. / 2 * (xn[n] ** 2 - 2 * xn[n] * m_mx[0] + m_mx[1]), 1. / 2])
    phi_g = phi_gamma + np.sum(m_xg, axis=0)
    a_gamma = phi_g[1] + 1
    b_gamma = -phi_g[0]

    lb = lowerbound(m_gx, m_xm, m_mx, m_xg, phi_m, phi_g)

    if it > 0:
        inc = 100 * (lb_old - lb) / (lb_old)
        print "It: ", it, "Mean: ", m_mu, "Precision: ", a_gamma / b_gamma, "Lowerbound: ", lb, "Increase: ", inc
    else:
        inc = 0.
        print "It: ", it, "Mean: ", m_mu, "Precision: ", a_gamma / b_gamma, "Lowerbound: ", lb

    lb_old = lb
    it += 1
