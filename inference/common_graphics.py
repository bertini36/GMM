# -*- coding: UTF-8 -*-

"""
Script that executes:
    - Coordinate Ascent Variational Inference
    - Gradient Ascent Variational Inference
    - Sthocastic Coordinate Ascent Variational Inference
    - Sthocastic Gradient Ascent Variational Inference
with common configurations and plot:
    - Its elbo's evolutions
    - Its time per iteration
"""

from __future__ import absolute_import

import pickle as pkl
import subprocess

import matplotlib.pyplot as plt
import numpy as np

DATASET = '../data/synthetic/2D/k2/data_k2_1000.pkl'
MAX_ITER = 300
K = 2
BATCH_SIZE = 100
CALL = False
ALGORITHMS = ['CAVI', 'GAVI', 'SCAVI', 'SGAVI']


def main():

    if CALL:
        subprocess.call(['python', 'python/gmm_cavi.py', '-dataset', DATASET,
                         '-maxIter', str(MAX_ITER),
                         '-k', str(K), '-verbose', '-exportELBOs'])
        subprocess.call(['python', 'tensorflow/gmm_gavi.py', '-dataset',
                         DATASET, '-maxIter', str(MAX_ITER),
                         '-k', str(K), '-verbose', '-exportELBOs'])
        subprocess.call(['python', 'python/gmm_scavi.py', '-dataset', DATASET,
                         '-maxIter', str(MAX_ITER), '-k', str(K),
                         '-verbose', '-bs', str(BATCH_SIZE), '-exportELBOs'])
        subprocess.call(['python', 'tensorflow/gmm_sgavi.py', '-dataset',
                         DATASET, '-maxIter', str(MAX_ITER), '-k', str(K),
                         '-verbose', '-bs', str(BATCH_SIZE), '-exportELBOs'])

    with open('generated/cavi_elbos.pkl', 'r') as input:
        data = pkl.load(input)
        elbos_cavi = data['elbos']
        iter_time_cavi = data['iter_time']
    with open('generated/gavi_elbos.pkl', 'r') as input:
        data = pkl.load(input)
        elbos_gavi = data['elbos']
        iter_time_gavi = data['iter_time']
    with open('generated/scavi_elbos.pkl', 'r') as input:
        data = pkl.load(input)
        elbos_scavi = data['elbos']
        iter_time_scavi = data['iter_time']
    with open('generated/sgavi_elbos.pkl', 'r') as input:
        data = pkl.load(input)
        elbos_sgavi = data['elbos']
        iter_time_sgavi = data['iter_time']

    plt.style.use('seaborn-darkgrid')

    # Plot ELBOs
    plt.plot(np.arange(len(elbos_cavi)), elbos_cavi)
    plt.plot(np.arange(len(elbos_gavi)), elbos_gavi)
    plt.plot(np.arange(len(elbos_scavi)), elbos_scavi)
    plt.plot(np.arange(len(elbos_sgavi)), elbos_sgavi)
    plt.ylabel('ELBO')
    plt.xlabel('Iterations')
    plt.legend(ALGORITHMS, loc='lower right')
    plt.savefig('generated/elbos.png')
    plt.show()
    plt.gcf().clear()

    # Plot normalized ELBOs
    plt.style.use('seaborn-darkgrid')
    plt.plot(np.arange(len(elbos_cavi)), -np.log(np.absolute(elbos_cavi)))
    plt.plot(np.arange(len(elbos_gavi)), -np.log(np.absolute(elbos_gavi)))
    plt.plot(np.arange(len(elbos_scavi)), -np.log(np.absolute(elbos_scavi)))
    plt.plot(np.arange(len(elbos_sgavi)), -np.log(np.absolute(elbos_sgavi)))
    plt.ylabel('ELBO')
    plt.xlabel('Iterations')
    plt.legend(ALGORITHMS, loc='lower right')
    plt.savefig('generated/standarized_elbos.png')
    plt.show()
    plt.gcf().clear()

    # Plot iter time histogram
    times = np.array([iter_time_cavi, iter_time_gavi,
                      iter_time_scavi, iter_time_sgavi])
    pos = np.arange(len(ALGORITHMS))
    width = 0.95
    ax = plt.axes()
    ax.set_xticks(pos + (width / 13))
    ax.set_xticklabels(ALGORITHMS)
    plt.bar(pos, times, width, color='r')
    plt.savefig('generated/elbos_times.png')
    plt.show()


if __name__ == '__main__': main()
