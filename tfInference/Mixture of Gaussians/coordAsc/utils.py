# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf

def tf_log_beta_function(x):
	return tf.sub(tf.reduce_sum(tf.lgamma(tf.add(x, np.finfo(np.float32).eps))), \
				  tf.lgamma(tf.add(x, np.finfo(np.float32).eps)))

def tf_dirichlet_expectation(alpha):
	if len(alpha.get_shape()) == 1:
		return tf.sub(tf.digamma(tf.add(alpha, np.finfo(np.float32).eps)), \
					  tf.digamma(tf.reduce_sum(alpha)))
	return tf.sub(tf.digamma(alpha), \
				  tf.digamma(tf.reduce_sum(alpha, 1))[:, tf.newaxis])