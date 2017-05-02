# -*- coding: UTF-8 -*-

"""
Python inference common functions
"""

from __future__ import absolute_import

import numpy as np
from scipy.special import gammaln, psi


def dirichlet_expectation(alpha):
    """
    Dirichlet expectation computation
    \Psi(\alpha) - \Psi(\sum_{i=1}^{K}(\alpha_{i}))
    """
    if len(alpha.shape) == 1:
        return psi(alpha + np.finfo(np.float32).eps) - psi(np.sum(alpha))
    return psi(alpha + np.finfo(np.float32).eps)\
           - psi(np.sum(alpha, 1))[:, np.newaxis]


def dirichlet_expectation_k(alpha, k):
    """
    Dirichlet expectation computation
    \Psi(\alpha_{k}) - \Psi(\sum_{i=1}^{K}(\alpha_{i}))
    """
    return psi(alpha[k] + np.finfo(np.float32).eps) - psi(np.sum(alpha))


def log_beta_function(x):
    """
    Log beta function
    ln(\gamma(x)) - ln(\gamma(\sum_{i=1}^{N}(x_{i}))
    """
    return np.sum(gammaln(x + np.finfo(np.float32).eps)) - gammaln(
        np.sum(x + np.finfo(np.float32).eps))


def softmax(x):
    """
    Softmax computation
    e^{x} / sum_{i=1}^{K}(e^x_{i})
    """
    e_x = np.exp(x - np.max(x))
    return (e_x + np.finfo(np.float32).eps) / \
           (e_x.sum(axis=0) + np.finfo(np.float32).eps)


def log_(x):
    return np.log(x + np.finfo(np.float32).eps)
