# -*- coding: UTF-8 -*-

"""
Coordinate Ascent Variational Inference with linesearch process to 
approximate a mixture of gaussians with common variance for all classes
"""

import argparse
import math
import pickle as pkl
from time import time

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from viz import create_cov_ellipse

"""
Parameters:
    * maxIter: Max number of iterations
    * dataset: Dataset path
    * k: Number of clusters
    * verbose: Printing time, intermediate variational parameters, plots, ...
"""

parser = argparse.ArgumentParser(
    description='CAVI with Linesearch in mixture of gaussians')
parser.add_argument('-maxIter', metavar='maxIter', type=int, default=10000)
parser.add_argument('-dataset', metavar='dataset', type=str,
                    default='../../../data/k2/data_k2_100.pkl')
parser.add_argument('-k', metavar='k', type=int, default=2)
parser.add_argument('--verbose', dest='verbose', action='store_true')
parser.add_argument('--no-verbose', dest='verbose', action='store_false')
parser.set_defaults(verbose=True)
args = parser.parse_args()

MAX_ITERS = args.maxIter
K = args.k
THRESHOLD = 1e-6
VERBOSE = args.verbose
PATH_IMAGE = 'img/gmm_means_cavi_linesearch'

sess = tf.Session()


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


def plot_iteration(ax_spatial, circs, sctZ, lambda_m, delta_o, xn, n_iters):
    """
    Plot the Gaussians in every iteration
    """
    if n_iters == 0:
        plt.scatter(xn[:, 0], xn[:, 1], cmap=cm.gist_rainbow, s=5)
        sctZ = plt.scatter(lambda_m[:, 0], lambda_m[:, 1],
                           color='black', s=5)
    else:
        for circ in circs: circ.remove()
        circs = []
        for k in range(K):
            cov = delta_o
            print('Cov: {}'.format(cov))
            circ = create_cov_ellipse(cov, lambda_m[k, :],
                                      color='r', alpha=0.3)
            circs.append(circ)
            ax_spatial.add_artist(circ)
        sctZ.set_offsets(lambda_m)
    plt.draw()
    plt.pause(0.001)
    return ax_spatial, circs, sctZ


# Get data
with open('{}'.format(args.dataset), 'r') as inputfile:
    data = pkl.load(inputfile)
    xn = data['xn']
    xn_tf = tf.convert_to_tensor(xn, dtype=tf.float64)
N, D = xn.shape

if VERBOSE: init_time = time()

# Model hyperparameters
alpha_o_aux = [1.0] * K
m_o_aux = np.array([0.0, 0.0])
beta_o_aux = 0.01
delta_o_aux = np.zeros((D, D), long)
np.fill_diagonal(delta_o_aux, 1)

# Priors (TF castings)
alpha_o = tf.convert_to_tensor([alpha_o_aux], dtype=tf.float64)
m_o = tf.convert_to_tensor([list(m_o_aux)], dtype=tf.float64)
beta_o = tf.convert_to_tensor(beta_o_aux, dtype=tf.float64)
delta_o = tf.convert_to_tensor(delta_o_aux, dtype=tf.float64)

# Initializations
lambda_phi_aux = np.random.dirichlet(alpha_o_aux, N)
lambda_pi_aux = alpha_o_aux + np.sum(lambda_phi_aux, axis=0)
lambda_beta_aux = beta_o_aux + np.sum(lambda_phi_aux, axis=0)
lambda_m_aux = np.tile(1. / lambda_beta_aux, (2, 1)).T * \
                  (beta_o_aux * m_o_aux + np.dot(lambda_phi_aux.T, xn))

# Variational parameters
lambda_phi_var = tf.Variable(lambda_phi_aux, dtype=tf.float64, name='phi_var')
lambda_pi_var = tf.Variable(lambda_pi_aux, dtype=tf.float64,
                            name='lambda_pi_var')
lambda_beta_var = tf.Variable(lambda_beta_aux, dtype=tf.float64,
                                 name='lambda_mu_beta_var')
lambda_m = tf.Variable(lambda_m_aux, dtype=tf.float64, name='lambda_mu_m')

# Maintain numerical stability
lambda_pi = tf.nn.softplus(lambda_pi_var)
lambda_beta = tf.nn.softplus(lambda_beta_var)
lambda_phi = tf.nn.softmax(lambda_phi_var)

# Reshapes
lambda_beta_res = tf.reshape(lambda_beta, [K, 1])

# Lower Bound definition
LB = log_beta_function(lambda_pi)
LB = tf.subtract(LB, log_beta_function(alpha_o))
LB = tf.add(LB, tf.matmul(tf.subtract(alpha_o, lambda_pi),
                          tf.reshape(dirichlet_expectation(lambda_pi),
                                     [K, 1])))
LB = tf.add(LB, tf.multiply(tf.cast(K / 2., tf.float64),
                            tf.log(tf.matrix_determinant(
                                tf.multiply(beta_o, delta_o)))))
LB = tf.add(LB, tf.cast(K * (D / 2.), tf.float64))
for k in range(K):
    a1 = tf.subtract(lambda_m[k, :], m_o)
    a2 = tf.matmul(delta_o, tf.transpose(tf.subtract(lambda_m[k, :], m_o)))
    a3 = tf.multiply(tf.div(beta_o, 2.), tf.matmul(a1, a2))
    a4 = tf.div(tf.multiply(tf.cast(D, tf.float64), beta_o),
                tf.multiply(tf.cast(2., tf.float64), lambda_beta_res[k]))
    a5 = tf.multiply(tf.cast(1 / 2., tf.float64), tf.log(
        tf.multiply(tf.pow(lambda_beta_res[k], 2),
                    tf.matrix_determinant(delta_o))))
    a6 = tf.add(a3, tf.add(a4, a5))
    LB = tf.subtract(LB, a6)
    b1 = tf.transpose(lambda_phi[:, k])
    b2 = dirichlet_expectation(lambda_pi)[k]
    b3 = tf.log(lambda_phi[:, k])
    b4 = tf.multiply(tf.cast(1 / 2., tf.float64), tf.log(
        tf.div(tf.matrix_determinant(delta_o),
               tf.multiply(tf.cast(2., tf.float64), math.pi))))
    b5 = tf.subtract(xn_tf, lambda_m[k, :])
    b6 = tf.matmul(delta_o, tf.transpose(tf.subtract(xn_tf, lambda_m[k, :])))
    b7 = tf.multiply(tf.cast(1 / 2., tf.float64),
                     tf.stack([tf.matmul(b5, b6)[i, i] for i in range(N)]))
    b8 = tf.div(tf.cast(D, tf.float64),
                tf.multiply(tf.cast(2., tf.float64), lambda_beta[k]))
    b9 = tf.subtract(tf.subtract(tf.add(tf.subtract(b2, b3), b4), b7), b8)
    b1 = tf.reshape(b1, [1, N])
    b9 = tf.reshape(b9, [N, 1])
    LB = tf.add(LB, tf.reshape(tf.matmul(b1, b9), [1]))


def compute_learning_rate(var, alpha_step):
    """
    Compute the optimal learning rate with linesearch
    :param var: Var to optimize
    :param alpha_step: Initial learning rate
    """
    # Obtaining the gradients
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha_step)
    grads_and_vars = optimizer.compute_gradients(-LB, var_list=[var])
    grads = sess.run(grads_and_vars)
    tmp_var = grads[0][1]
    tmp_grad = grads[0][0]

    # Gradient descent update
    fx = sess.run(-LB)
    tmp_mod = tmp_var - alpha_step * tmp_grad
    assign_op = var.assign(tmp_mod)
    sess.run(assign_op)
    fxgrad = sess.run(-LB)

    # Loop for problematic vars that produces Infs and Nans
    while np.isinf(fxgrad) or np.isnan(fxgrad):
        alpha_step /= 10.
        tmp_mod = tmp_var - alpha_step * tmp_grad
        assign_op = var.assign(tmp_mod)
        sess.run(assign_op)
        fxgrad = sess.run(-LB)

    tmp_grad = sess.run(tf.sqrt(tf.reduce_sum(tf.square(tmp_grad))))
    m = tmp_grad ** 2
    c = 0.5
    tau = 0.2

    while fxgrad >= fx - alpha_step * c * m:
        alpha_step *= tau
        tmp_mod = tmp_var - alpha_step * tmp_grad
        assign_op = var.assign(tmp_mod)
        sess.run(assign_op)
        fxgrad = sess.run(-LB)
        if alpha_step < 1e-10:
            alpha_step = 0
            break


# Summaries definition
tf.summary.histogram('lambda_phi', lambda_phi)
tf.summary.histogram('lambda_pi', lambda_pi)
tf.summary.histogram('lambda_m', lambda_m)
tf.summary.histogram('lambda_beta', lambda_beta)
merged = tf.summary.merge_all()
file_writer = tf.summary.FileWriter('/tmp/tensorboard/', tf.get_default_graph())


def main():

    # Plot configs
    if VERBOSE:
        plt.ion()
        fig = plt.figure(figsize=(10, 10))
        ax_spatial = fig.add_subplot(1, 1, 1)
        circs = []
        sctZ = None

    # Inference
    init = tf.global_variables_initializer()
    sess.run(init)
    alpha_step = 1e10
    lbs = []
    n_iters = 0
    for _ in range(MAX_ITERS):

        # Parameter updates with individual learning rates
        compute_learning_rate(lambda_pi_var, alpha_step)
        compute_learning_rate(lambda_phi_var, alpha_step)
        compute_learning_rate(lambda_m, alpha_step)
        compute_learning_rate(lambda_beta_var, alpha_step)

        # ELBO computation
        mer, lb, pi_out, phi_out, m_out, beta_out = sess.run(
            [merged, LB, lambda_pi, lambda_phi, lambda_m, lambda_beta])
        lbs.append(lb[0][0])

        if VERBOSE:
            print('\n******* ITERATION {} *******'.format(n_iters))
            print('lambda_pi: {}'.format(pi_out))
            print('lambda_beta: {}'.format(beta_out))
            print('lambda_m: {}'.format(m_out))
            print('lambda_phi: {}'.format(phi_out[0:9, :]))
            print('ELBO: {}'.format(lb))
            ax_spatial, circs, sctZ = plot_iteration(ax_spatial, circs, sctZ,
                                                     sess.run(lambda_m),
                                                     sess.run(delta_o),
                                                     xn, n_iters)

        # Break condition
        if n_iters > 0 and abs(lb - lbs[n_iters - 1]) < THRESHOLD:
            plt.savefig('{}.png'.format(PATH_IMAGE))
            break

        n_iters += 1
        file_writer.add_summary(mer, n_iters)

    if VERBOSE:
        print('\n******* RESULTS *******')
        for k in range(K):
            print('Mu k{}: {}'.format(k, m_out[k, :]))
        final_time = time()
        exec_time = final_time - init_time
        print('Time: {} seconds'.format(exec_time))
        print('Iterations: {}'.format(n_iters))
        print('ELBOs: {}'.format(lbs))


if __name__ == '__main__': main()
