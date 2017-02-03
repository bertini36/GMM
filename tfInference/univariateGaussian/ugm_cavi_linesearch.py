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

parser = argparse.ArgumentParser(
    description='CAVI Linesearch in univariate gaussian')
parser.add_argument('-maxIter', metavar='maxIter', type=int, default=10000000)
parser.add_argument('-nElements', metavar='nElements', type=int, default=100)
parser.add_argument('--timing', dest='timing', action='store_true')
parser.add_argument('--no-timing', dest='timing', action='store_false')
parser.set_defaults(timing=False)
parser.add_argument('--getNIter', dest='getNIter', action='store_true')
parser.add_argument('--no-getNIter', dest='getNIter', action='store_false')
parser.set_defaults(getNIter=False)
parser.add_argument('--getELBOs', dest='getELBOs', action='store_true')
parser.add_argument('--no-getELBOs', dest='getELBOs', action='store_false')
parser.set_defaults(getELBOs=False)
parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--no-debug', dest='debug', action='store_false')
parser.set_defaults(debug=True)
parser.add_argument('--plot', dest='plot', action='store_true')
parser.add_argument('--no-plot', dest='plot', action='store_false')
parser.set_defaults(plot=True)
args = parser.parse_args()

N = args.nElements
MAX_ITERS = args.maxIter
DATA_MEAN = 7
THRESHOLD = 1e-6

sess = tf.Session()

# Data generation
xn_np = np.random.normal(DATA_MEAN, 1, N)
xn = tf.convert_to_tensor(xn_np, dtype=tf.float64)

if args.timing:
    init_time = time()

# Model hyperparameters
m = tf.Variable(0., dtype=tf.float64)
beta = tf.Variable(0.0001, dtype=tf.float64)
a = tf.Variable(0.001, dtype=tf.float64)
b = tf.Variable(0.001, dtype=tf.float64)

# Needed for variational initilizations
a_gamma_ini = np.random.gamma(1, 1, 1)[0]
b_gamma_ini = np.random.gamma(1, 1, 1)[0]

# Variational parameters
a_gamma_var = tf.Variable(a_gamma_ini, dtype=tf.float64)
b_gamma_var = tf.Variable(b_gamma_ini, dtype=tf.float64)
m_mu = tf.Variable(np.random.normal(0., (0.0001) ** (-1.), 1)[0],
                   dtype=tf.float64)
beta_mu_var = tf.Variable(np.random.gamma(a_gamma_ini, b_gamma_ini, 1)[0],
                          dtype=tf.float64)

# Maintain numerical stability
a_gamma = tf.nn.softplus(a_gamma_var)
b_gamma = tf.nn.softplus(b_gamma_var)
beta_mu = tf.nn.softplus(beta_mu_var)

# Lower Bound definition
LB = tf.multiply(tf.cast(1. / 2, tf.float64), tf.log(tf.div(beta, beta_mu)))
LB = tf.add(LB, tf.multiply(tf.multiply(tf.cast(1. / 2, tf.float64),
                                        tf.add(tf.pow(m_mu, 2),
                                               tf.div(tf.cast(1., tf.float64),
                                                      beta_mu))),
                            tf.subtract(beta_mu, beta)))
LB = tf.subtract(LB, tf.multiply(m_mu, tf.subtract(tf.multiply(beta_mu, m_mu),
                                                   tf.multiply(beta, m))))
LB = tf.add(LB, tf.multiply(tf.cast(1. / 2, tf.float64),
                            tf.subtract(tf.multiply(beta_mu, tf.pow(m_mu, 2)),
                                        tf.multiply(beta, tf.pow(m, 2)))))

LB = tf.add(LB, tf.multiply(a, tf.log(b)))
LB = tf.subtract(LB, tf.multiply(a_gamma, tf.log(b_gamma)))
LB = tf.add(LB, tf.lgamma(a_gamma))
LB = tf.subtract(LB, tf.lgamma(a))
LB = tf.add(LB, tf.multiply(tf.subtract(tf.digamma(a_gamma), tf.log(b_gamma)),
                            tf.subtract(a, a_gamma)))
LB = tf.add(LB, tf.multiply(tf.div(a_gamma, b_gamma), tf.subtract(b_gamma, b)))

LB = tf.add(LB,
            tf.multiply(tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64)),
                        tf.subtract(tf.digamma(a_gamma), tf.log(b_gamma))))
LB = tf.subtract(LB, tf.multiply(
    tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64)),
    tf.log(tf.multiply(tf.cast(2., tf.float64), math.pi))))
LB = tf.subtract(LB, tf.multiply(tf.cast(1. / 2, tf.float64),
                                 tf.multiply(tf.div(a_gamma, b_gamma),
                                             tf.reduce_sum(tf.pow(xn, 2)))))
LB = tf.add(LB,
            tf.multiply(tf.div(a_gamma, b_gamma),
                        tf.multiply(tf.reduce_sum(xn), m_mu)))
LB = tf.subtract(LB, tf.multiply(
    tf.div(tf.cast(N, tf.float64), tf.cast(2., tf.float64)),
    tf.multiply(tf.div(a_gamma, b_gamma), tf.add(tf.pow(m_mu, 2),
                                                 tf.div(
                                                     tf.cast(1.,
                                                             tf.float64),
                                                     beta_mu)))))


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
tf.summary.histogram('m_mu', m_mu)
tf.summary.histogram('beta_mu', beta_mu)
tf.summary.histogram('a_gamma', a_gamma)
tf.summary.histogram('b_gamma', b_gamma)
merged = tf.summary.merge_all()
file_writer = tf.summary.FileWriter('/tmp/tensorboard/', tf.get_default_graph())


def main():
    if args.plot:
        plt.plot(xn_np, 'go')
        plt.title('Simulated dataset')
        plt.show()

    init = tf.global_variables_initializer()
    run_calls = 0
    sess.run(init)
    alpha = 1e10
    lbs = []
    for i in xrange(MAX_ITERS):

        # Parameter updates with individual learning rates
        compute_learning_rate(a_gamma_var, alpha)
        compute_learning_rate(b_gamma_var, alpha)
        compute_learning_rate(m_mu, alpha)
        compute_learning_rate(beta_mu_var, alpha)

        # ELBO computation
        mer, lb, mu_out, beta_out, a_out, b_out = sess.run(
            [merged, LB, m_mu, beta_mu, a_gamma, b_gamma])
        if args.debug:
            print('Iter {}: Mean={} Precision={} ELBO={}'
                  .format(i, mu_out, a_out / b_out, lb))
        run_calls += 1
        file_writer.add_summary(mer, run_calls)

        # Break condition
        if i > 0:
            if abs(lb - lbs[i - 1]) < THRESHOLD:
                if args.getNIter:
                    n_iters = i + 1
                break
        lbs.append(lb)

    if args.plot:
        plt.scatter(xn_np, mlab.normpdf(xn_np, mu_out, a_out / b_out), s=5)
        plt.title('Results')
        plt.show()

    if args.timing:
        final_time = time()
        exec_time = final_time - init_time
        print('Time: {} seconds'.format(exec_time))

    if args.getNIter:
        print('Iterations: {}'.format(n_iters))

    if args.getELBOs:
        print('ELBOs: {}'.format(lbs))


if __name__ == '__main__': main()
