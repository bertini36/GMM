# -*- coding: UTF-8 -*-

"""
Autograd common functions
"""

from __future__ import absolute_import

import autograd.numpy as agnp
import autograd.scipy.special as agscipy
from sklearn.cluster import KMeans


def dirichlet_expectation(alpha):
    """
    Dirichlet expectation computation
    \Psi(\alpha) - \Psi(\sum_{i=1}^{K}(\alpha_{i}))
    """
    if len(alpha.shape) == 1:
        return agscipy.psi(alpha + agnp.finfo(agnp.float32).eps) \
               - agscipy.psi(agnp.sum(alpha))
    return agscipy.psi(alpha + agnp.finfo(agnp.float32).eps)\
           - agscipy.psi(agnp.sum(alpha, 1))[:, agnp.newaxis]


def log_beta_function(x):
    """
    Log beta function
    ln(\gamma(x)) - ln(\gamma(\sum_{i=1}^{N}(x_{i}))
    """
    return agnp.sum(agscipy.gammaln(x + agnp.finfo(agnp.float32).eps)) \
           - agscipy.gammaln(agnp.sum(x + agnp.finfo(agnp.float32).eps))


def init_kmeans(xn, N, K):
    """
    Init points assignations (lambda_phi) with Kmeans clustering
    """
    lambda_phi = 0.1 / (K - 1) * agnp.ones((N, K))
    labels = KMeans(K).fit(xn).predict(xn)
    for i, lab in enumerate(labels):
        lambda_phi[i, lab] = 0.9
    return lambda_phi


def log_(x):
    return agnp.log(x + agnp.finfo(agnp.float32).eps)


def softmax(x):
    """
    Softmax computation
    e^{x} / sum_{i=1}^{K}(e^x_{i})
    """
    e_x = agnp.exp(x - agnp.max(x))
    return (e_x + agnp.finfo(agnp.float32).eps) / \
           (e_x.sum(axis=0) + agnp.finfo(agnp.float32).eps)


def softplus(x):
    """
    Softplus computation
    """
    return agnp.log(1 + agnp.exp(x))
