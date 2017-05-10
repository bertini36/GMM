# -*- coding: UTF-8 -*-

"""
Tensorflow inference common functions
"""

from __future__ import absolute_import

import numpy as np
import tensorflow as tf


def dirichlet_expectation(alpha):
    """
    Dirichlet expectation computation
    \Psi(\alpha_{k}) - \Psi(\sum_{i=1}^{K}(\alpha_{i}))
    """
    return tf.subtract(tf.digamma(tf.add(alpha, np.finfo(np.float32).eps)),
                       tf.digamma(tf.reduce_sum(alpha)))


def dirichlet_expectation_k(alpha, k):
    """
    Dirichlet expectation computation
    \Psi(\alpha_{k}) - \Psi(\sum_{i=1}^{K}(\alpha_{i}))
    """
    return tf.subtract(tf.digamma(tf.add(alpha[k], np.finfo(np.float32).eps)),
                       tf.digamma(tf.reduce_sum(alpha)))


def log_beta_function(x):
    """
    Log beta function
    ln(\gamma(x)) - ln(\gamma(\sum_{i=1}^{N}(x_{i}))
    """
    return tf.subtract(
        tf.reduce_sum(tf.lgamma(tf.add(x, np.finfo(np.float32).eps))),
        tf.lgamma(tf.reduce_sum(tf.add(x, np.finfo(np.float32).eps))))


def softmax(x):
    """
    Softmax computation
    e^{x} / sum_{i=1}^{K}(e^x_{i})
    """
    return tf.div(tf.add(tf.exp(tf.subtract(x, tf.reduce_max(x))),
                         np.finfo(np.float32).eps),
                  tf.reduce_sum(
                      tf.add(tf.exp(tf.subtract(x, tf.reduce_max(x))),
                             np.finfo(np.float32).eps)))


def multilgamma(a, D, D_t):
    """
    ln multigamma Tensorflow implementation
    """
    res = tf.multiply(tf.multiply(D_t, tf.multiply(tf.subtract(D_t, 1),
                                                   tf.cast(0.25,
                                                           dtype=tf.float64))),
                      tf.log(tf.cast(np.pi, dtype=tf.float64)))
    res += tf.reduce_sum(tf.lgamma([tf.subtract(a, tf.div(
        tf.subtract(tf.cast(j, dtype=tf.float64),
                    tf.cast(1., dtype=tf.float64)),
        tf.cast(2., dtype=tf.float64))) for j in range(1, D + 1)]), axis=0)
    return res


def log_(x):
    return tf.log(tf.add(x, np.finfo(np.float32).eps))

