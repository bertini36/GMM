# -*- coding: UTF-8 -*-

"""
Gradient Ascent Variational Inference
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
from edward.models import (Categorical, InverseGamma, Mixture,
                           MultivariateNormalDiag, Normal)

plt.style.use('ggplot')

parser = argparse.ArgumentParser(description='GAVI in mixture of gaussians')
parser.add_argument('-maxIter', metavar='maxIter', type=int, default=4000)
parser.add_argument('-dataset', metavar='dataset',
                    type=str, default='../../data/data_k2_100.pkl')
parser.add_argument('-k', metavar='k', type=int, default=2)
parser.add_argument('-s', metavar='s', type=int, default=30)
parser.add_argument('--timing', dest='timing', action='store_true')
parser.add_argument('--no-timing', dest='timing', action='store_false')
parser.set_defaults(timing=False)
parser.add_argument('--getELBOs', dest='getELBOs', action='store_true')
parser.add_argument('--no-getELBOs', dest='getELBOs', action='store_false')
parser.set_defaults(getELBO=False)
parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--no-debug', dest='debug', action='store_false')
parser.set_defaults(debug=True)
parser.add_argument('--plot', dest='plot', action='store_true')
parser.add_argument('--no-plot', dest='plot', action='store_false')
parser.set_defaults(plot=True)
args = parser.parse_args()

MAX_ITERS = args.maxIter
K = args.k


def main():
    # Data generation
    with open('../../data/data_k2_100.pkl', 'r') as inputfile:
        data = pkl.load(inputfile)
        xn = np.array(data['xn'], dtype=np.float32)
    N, D = xn.shape

    if args.timing:
        init_time = time()

    if args.plot:
        plt.scatter(xn[:, 0], xn[:, 1], c=(1. * data['zn']) / max(data['zn']),
                    cmap=cm.gist_rainbow, s=5)
        plt.title('Simulated dataset')
        plt.show()

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

    for _ in xrange(MAX_ITERS):
        info_dict = inference.update()
        if args.getELBO:
            inference.print_progress(info_dict)
        t = info_dict['t']
        if args.debug and t % inference.n_print == 0:
            print('Inferred cluster means: {}'.format(sess.run(qmu.value())))

    if args.plot:
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

    if args.timing:
        final_time = time()
        exec_time = final_time - init_time
        print('Time: {} seconds'.format(exec_time))


if __name__ == '__main__': main()
