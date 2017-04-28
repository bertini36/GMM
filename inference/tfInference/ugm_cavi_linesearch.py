# -*- coding: UTF-8 -*-

"""
Coordinate Ascent Variational Inference with Linesearch process
to approximate an univariate gaussian
"""

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
    python ugm_cavi_linesearch.py -nElements 1000 -verbose
"""

parser = argparse.ArgumentParser(
    description='CAVI Linesearch in univariate gaussian')
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
a_var = tf.Variable(a_ini, dtype=tf.float64)
b_var = tf.Variable(b_ini, dtype=tf.float64)
lambda_m = tf.Variable(np.random.normal(0., (0.0001) ** (-1.), 1)[0],
                       dtype=tf.float64)
beta_var = tf.Variable(np.random.gamma(a_ini, b_ini, 1)[0], dtype=tf.float64)

# Maintain numerical stability
lambda_a = tf.nn.softplus(a_var)
lambda_b = tf.nn.softplus(b_var)
lambda_beta = tf.nn.softplus(beta_var)

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
                            tf.subtract(
                                tf.multiply(lambda_beta, tf.pow(lambda_m, 2)),
                                tf.multiply(beta_o, tf.pow(m_o, 2)))))
LB = tf.add(LB, tf.multiply(a_o, tf.log(b_o)))
LB = tf.subtract(LB, tf.multiply(lambda_a, tf.log(lambda_b)))
LB = tf.add(LB, tf.lgamma(lambda_a))
LB = tf.subtract(LB, tf.lgamma(a_o))
LB = tf.add(LB, tf.multiply(tf.subtract(tf.digamma(lambda_a), tf.log(lambda_b)),
                            tf.subtract(a_o, lambda_a)))
LB = tf.add(LB,
            tf.multiply(tf.div(lambda_a, lambda_b), tf.subtract(lambda_b, b_o)))

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


def compute_learning_rate(var, alpha):
    """
    :param var: Var to optimize
    :param alpha: Initial learning rate
    """
    # Obtaining the gradients
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    grads_and_vars = optimizer.compute_gradients(-LB, var_list=[var])
    grads = sess.run(grads_and_vars)
    tmp_var = grads[0][1]
    tmp_grad = grads[0][0]

    # Gradient descent update
    fx = sess.run(-LB)
    tmp_mod = tmp_var - alpha * tmp_grad
    assign_op = var.assign(tmp_mod)
    sess.run(assign_op)
    fxgrad = sess.run(-LB)

    # Loop for problematic vars that produces Infs and Nans
    while np.isinf(fxgrad) or np.isnan(fxgrad):
        alpha /= 10.
        tmp_mod = tmp_var - alpha * tmp_grad
        assign_op = var.assign(tmp_mod)
        sess.run(assign_op)
        fxgrad = sess.run(-LB)

    m = tmp_grad ** 2
    c = 0.5
    tau = 0.2

    while fxgrad >= fx - alpha * c * m:
        alpha *= tau
        tmp_mod = tmp_var - alpha * tmp_grad
        assign_op = var.assign(tmp_mod)
        sess.run(assign_op)
        fxgrad = sess.run(-LB)
        if alpha < 1e-10:
            alpha = 0
            break


# Summaries definition
tf.summary.histogram('lambda_m', lambda_m)
tf.summary.histogram('lambda_beta', lambda_beta)
tf.summary.histogram('lambda_a', lambda_a)
tf.summary.histogram('lambda_a', lambda_b)
merged = tf.summary.merge_all()
file_writer = tf.summary.FileWriter('/tmp/tensorboard/', tf.get_default_graph())


def main():

    if VERBOSE:
        plt.plot(xn_np, 'ro', markersize=3)
        plt.title('Simulated dataset')
        plt.show()

    init = tf.global_variables_initializer()
    sess.run(init)
    alpha = 1e10
    lbs = []
    n_iters = 0
    for _ in range(args.maxIter):

        # Parameter updates with individual learning rates
        compute_learning_rate(a_var, alpha)
        compute_learning_rate(b_var, alpha)
        compute_learning_rate(lambda_m, alpha)
        compute_learning_rate(beta_var, alpha)

        # ELBO computation
        mer, lb, m_out, beta_out, a_out, b_out = sess.run(
            [merged, LB, lambda_m, lambda_beta, lambda_a, lambda_b])
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
