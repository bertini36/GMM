# -*- coding: UTF-8 -*-

"""
Coordinate Ascent Variational Inference
process to approximate an univariate gaussian
"""

from __future__ import absolute_import

import argparse
import math
from time import time

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

"""
Parameters:
    * maxIter: Max number of iterations
    * nElements: Number of data points to generate
    * verbose: Printing time, intermediate variational parameters, plots, ...
    
Execution:
    python ugm_cavi.py -nElements 1000 -verbose
"""

parser = argparse.ArgumentParser(description='CAVI in univariate gaussian')
parser.add_argument('-maxIter', metavar='maxIter', type=int, default=100)
parser.add_argument('-nElements', metavar='nElements', type=int, default=1000)
parser.add_argument('-verbose', dest='verbose', action='store_true')
parser.set_defaults(verbose=False)
args = parser.parse_args()

N = args.nElements
VERBOSE = args.verbose
DATA_MEAN = 7
THRESHOLD = 1e-6

sess = tf.Session()

# Data generation
xn_np = np.random.normal(DATA_MEAN, 1, N)
xn = tf.convert_to_tensor(xn_np, dtype=tf.float64)

if VERBOSE: init_time = time()

# Model hyperparameters
m_o = tf.Variable(0., dtype=tf.float64)
beta_o = tf.Variable(0.0001, dtype=tf.float64)
a_o = tf.Variable(0.001, dtype=tf.float64)
b_o = tf.Variable(0.001, dtype=tf.float64)

# Needed for variational initilizations
a_ini = np.random.gamma(1, 1, 1)[0]
b_ini = np.random.gamma(1, 1, 1)[0]

# Variational parameters
lambda_a = tf.Variable(a_ini, dtype=tf.float64)
lambda_b = tf.Variable(b_ini, dtype=tf.float64)
lambda_m = tf.Variable(np.random.normal(0., 0.0001 ** (-1.), 1)[0],
                       dtype=tf.float64)
lambda_beta = tf.Variable(np.random.gamma(a_ini, b_ini, 1)[0], dtype=tf.float64)

# Lower Bound definition
LB = tf.multiply(tf.cast(1. / 2, tf.float64),
                 tf.log(tf.div(beta_o, lambda_beta)))
LB = tf.add(LB, tf.multiply(tf.multiply(tf.cast(1. / 2, tf.float64),
                                        tf.add(tf.pow(lambda_m, 2),
                                               tf.div(tf.cast(1., tf.float64),
                                                      lambda_beta))),
                            tf.subtract(lambda_beta, beta_o)))
LB = tf.subtract(LB, tf.multiply(lambda_m,
                                 tf.subtract(tf.multiply(lambda_beta, lambda_m),
                                             tf.multiply(beta_o, m_o))))
LB = tf.add(LB, tf.multiply(tf.cast(1. / 2, tf.float64),
                            tf.subtract(tf.multiply(lambda_beta,
                                                    tf.pow(lambda_m, 2)),
                                        tf.multiply(beta_o, tf.pow(m_o, 2)))))
LB = tf.add(LB, tf.multiply(a_o, tf.log(b_o)))
LB = tf.subtract(LB, tf.multiply(lambda_a, tf.log(lambda_b)))
LB = tf.add(LB, tf.lgamma(lambda_a))
LB = tf.subtract(LB, tf.lgamma(a_o))
LB = tf.add(LB, tf.multiply(tf.subtract(tf.digamma(lambda_a), tf.log(lambda_b)),
                            tf.subtract(a_o, lambda_a)))
LB = tf.add(LB, tf.multiply(tf.div(lambda_a, lambda_b),
                            tf.subtract(lambda_b, b_o)))
LB = tf.add(LB,
            tf.multiply(tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64)),
                        tf.subtract(tf.digamma(lambda_a), tf.log(lambda_b))))
LB = tf.subtract(LB, tf.multiply(
    tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64)),
    tf.log(tf.multiply(tf.cast(2., tf.float64), math.pi))))
LB = tf.subtract(LB, tf.multiply(tf.cast(1. / 2, tf.float64),
                                 tf.multiply(tf.div(lambda_a, lambda_b),
                                             tf.reduce_sum(tf.pow(xn, 2)))))
LB = tf.add(LB,
            tf.multiply(tf.div(lambda_a, lambda_b),
                        tf.multiply(tf.reduce_sum(xn), lambda_m)))
LB = tf.subtract(LB, tf.multiply(
    tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64)),
    tf.multiply(tf.div(lambda_a, lambda_b), tf.add(tf.pow(lambda_m, 2),
                                                   tf.div(
                                                       tf.cast(1.,
                                                               tf.float64),
                                                       lambda_beta)))))

# Parameter updates
assign_lambda_m = lambda_m.assign(tf.div(tf.add(tf.multiply(beta_o, m_o),
                                                tf.multiply(
                                                    tf.div(lambda_a, lambda_b),
                                                    tf.reduce_sum(xn))),
                                         tf.add(beta_o,
                                                tf.multiply(
                                                    tf.cast(N, tf.float64),
                                                    tf.div(lambda_a,
                                                           lambda_b)))))

assign_lambda_beta = lambda_beta.assign(
    tf.add(beta_o,
           tf.multiply(tf.cast(N, tf.float64), tf.div(lambda_a, lambda_b))))

assign_lambda_a = lambda_a.assign(
    tf.add(a_o, tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64))))

assign_lambda_b = lambda_b.assign(tf.add(b_o, tf.add(
    tf.subtract(
        tf.multiply(tf.cast(1. / 2, tf.float64), tf.reduce_sum(tf.pow(xn, 2))),
        tf.multiply(lambda_m, tf.reduce_sum(xn))),
    tf.multiply(tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64)),
                tf.add(tf.pow(lambda_m, 2),
                       tf.div(tf.cast(1., tf.float64), lambda_beta))))))

# Summaries definition
tf.summary.histogram('lambda_m', lambda_m)
tf.summary.histogram('lambda_beta', lambda_beta)
tf.summary.histogram('lambda_a', lambda_a)
tf.summary.histogram('lambda_b', lambda_b)
merged = tf.summary.merge_all()
file_writer = tf.summary.FileWriter('/tmp/tensorboard/', tf.get_default_graph())


def main():

    if VERBOSE:
        plt.plot(xn_np, 'ro', markersize=3)
        plt.title('Simulated dataset')
        plt.show()

    # Inference
    init = tf.global_variables_initializer()
    sess.run(init)
    lbs = []
    n_iters = 0
    for _ in range(args.maxIter):

        # Parameter updates
        sess.run([assign_lambda_m, assign_lambda_beta, assign_lambda_a])
        sess.run(assign_lambda_b)
        m_out, beta_out, a_out, b_out = sess.run(
            [lambda_m, lambda_beta, lambda_a, lambda_b])

        # ELBO computation
        mer, lb = sess.run([merged, LB])
        lbs.append(lb)

        if VERBOSE:
            print('\n******* ITERATION {} *******'.format(n_iters))
            print('lambda_m: {}'.format(m_out))
            print('lambda_beta: {}'.format(beta_out))
            print('lambda_a: {}'.format(a_out))
            print('lambda_b: {}'.format(b_out))
            print('ELBO: {}'.format(lb))

        # Break condition
        if n_iters > 0 and abs(lb - lbs[n_iters - 1]) < THRESHOLD: break

        n_iters += 1
        file_writer.add_summary(mer, n_iters)

    if VERBOSE:
        plt.scatter(xn_np, mlab.normpdf(xn_np, m_out, a_out / b_out), s=5)
        plt.title('Result')
        plt.show()
        final_time = time()
        exec_time = final_time - init_time
        print('Time: {} seconds'.format(exec_time))
        print('Iterations: {}'.format(n_iters))
        print('ELBOs: {}'.format(lbs))


if __name__ == '__main__': main()
