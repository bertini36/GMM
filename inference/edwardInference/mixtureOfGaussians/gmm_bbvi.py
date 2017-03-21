# -*- coding: UTF-8 -*-

"""
Black Box Variational Inference
process to approximate a mixture of gaussians
"""

import argparse
import pickle as pkl
from time import time

import edward as ed
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from edward.models import Categorical, InverseGamma, \
    Mixture, MultivariateNormalDiag, Normal

"""
Parameters:
    * maxIter: Max number of iterations
    * dataset: Dataset path
    * k: Number of clusters
    * s: Number of samples
    * verbose: Printing time, intermediate variational parameters, plots, ...
"""

plt.style.use('ggplot')

parser = argparse.ArgumentParser(description='BBVI in mixture of gaussians')
parser.add_argument('-maxIter', metavar='maxIter', type=int, default=500)
parser.add_argument('-dataset', metavar='dataset',
                    type=str, default='../../../data/data_k2_100.pkl')
parser.add_argument('-k', metavar='k', type=int, default=2)
parser.add_argument('-s', metavar='s', type=int, default=30)
parser.add_argument('--verbose', dest='verbose', action='store_true')
parser.add_argument('--no-verbose', dest='verbose', action='store_false')
parser.set_defaults(verbose=True)
args = parser.parse_args()

MAX_ITERS = args.maxIter
K = args.k
VERBOSE = args.verbose


def main():

    # Get data
    with open('../../../data/k2/data_k2_100.pkl', 'r') as inputfile:
        data = pkl.load(inputfile)
        xn = np.array(data['xn'], dtype=np.float32)
    N, D = xn.shape

    if VERBOSE: init_time = time()

    # Probabilistic model
    mu = Normal(mu=tf.zeros([K, D]), sigma=tf.ones([K, D]))
    sigma = InverseGamma(alpha=tf.ones([K, D]), beta=tf.ones([K, D]))
    cat = Categorical(logits=tf.zeros([N, K]))
    components = [
        MultivariateNormalDiag(mu=tf.ones([N, 1]) * tf.gather(mu, k),
                               diag_stdev=tf.ones([N, 1]) * tf.gather(sigma, k))
        for k in range(K)]
    x = Mixture(cat=cat, components=components)

    # Variational model
    qmu = Normal(
        mu=tf.Variable(tf.random_normal([K, D])),
        sigma=tf.nn.softplus(tf.Variable(tf.zeros([K, D]))))
    qsigma = InverseGamma(
        alpha=tf.nn.softplus(tf.Variable(tf.random_normal([K, D]))),
        beta=tf.nn.softplus(tf.Variable(tf.random_normal([K, D]))))

    # Inference
    inference = ed.KLqp({mu: qmu, sigma: qsigma}, data={x: xn})
    inference.initialize(n_samples=args.s, n_iter=MAX_ITERS)

    sess = ed.get_session()
    init = tf.global_variables_initializer()
    init.run()

    n_iters = 0
    for _ in range(MAX_ITERS):
        info_dict = inference.update()
        if VERBOSE: inference.print_progress(info_dict)
        n_iters += 1

    if VERBOSE:
        print('\n******* RESULTS *******')
        for k in range(K):
            print('Mu k{}: {}'.format(k, sess.run(qmu.mean()[k])))
            print('Sigma k{}: {}'.format(k, sess.run(qsigma.mean()[k])))
        log_liks = []
        for _ in range(100):
            mu_sample = qmu.sample()
            sigma_sample = qsigma.sample()
            log_lik = []
            for k in range(K):
                x_post = Normal(mu=tf.ones([N, 1]) * tf.gather(mu_sample, k),
                                sigma=tf.ones([N, 1]) * tf.gather(sigma_sample,
                                                                  k))
                log_lik.append(tf.reduce_sum(x_post.log_prob(xn), 1))
            log_lik = tf.stack(log_lik)
            log_liks.append(log_lik)
        log_liks = tf.reduce_mean(log_liks, 0)
        clusters = tf.argmax(log_liks, 0).eval()
        plt.scatter(xn[:, 0], xn[:, 1], c=clusters, cmap=cm.gist_rainbow, s=5)
        plt.title('Predicted cluster assignments')
        plt.show()
        final_time = time()
        exec_time = final_time - init_time
        print('Time: {} seconds'.format(exec_time))
        print('Iterations: {}'.format(n_iters))

if __name__ == '__main__': main()
