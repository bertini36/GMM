# -*- coding: UTF-8 -*-

"""
Tensorflow inference common functions
"""

import numpy as np
import tensorflow as tf


def dirichlet_expectation(alpha):
    """
    Dirichlet expectation computation
    \Psi(\alpha_{k}) - \Psi(\sum_{i=1}^{K}(\alpha_{i}))
    """
    return tf.subtract(tf.digamma(tf.add(alpha, np.finfo(np.float32).eps)),
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
