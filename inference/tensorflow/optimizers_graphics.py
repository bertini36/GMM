# -*- coding: UTF-8 -*-

"""
Script that executes:
    - Sthocastic Gradient Ascent Variational Inference with Adam optimizer
    - Sthocastic Gradient Ascent Variational Inference with Adagrad optimizer
    - Sthocastic Gradient Ascent Variational Inference with Adadelta optimizer
    - Sthocastic Gradient Ascent Variational Inference with Rmsprop optimizer
with common configurations and plot:
    - Its elbo's evolutions
"""

from __future__ import absolute_import

import pickle as pkl
import subprocess

import matplotlib.pyplot as plt
import numpy as np

DATASET = '../../data/synthetic/2D/k4/data_k4_1000.pkl'
MAX_ITER = 300
K = 4
VERBOSE = False
BATCH_SIZE = 100
CALL = True
OPTIMIZERS = ['RMSPROP', 'ADAM', 'ADADELTA', 'ADAGRAD']


def main():

    if CALL:
        subprocess.call(['python', 'gmm_sgavi.py', '-dataset',
                         DATASET, '-maxIter', str(MAX_ITER), '-k', str(K),
                         '-verbose', '-bs', str(BATCH_SIZE), '-exportELBOs',
                         '-optimizer', 'rmsprop'])
        subprocess.call(['python', 'gmm_sgavi.py', '-dataset',
                         DATASET, '-maxIter', str(MAX_ITER), '-k', str(K),
                         '-verbose', '-bs', str(BATCH_SIZE), '-exportELBOs',
                         '-optimizer', 'adam'])
        subprocess.call(['python', 'gmm_sgavi.py', '-dataset',
                         DATASET, '-maxIter', str(MAX_ITER), '-k', str(K),
                         '-verbose', '-bs', str(BATCH_SIZE), '-exportELBOs',
                         '-optimizer', 'adadelta'])
        subprocess.call(['python', 'gmm_sgavi.py', '-dataset',
                         DATASET, '-maxIter', str(MAX_ITER), '-k', str(K),
                         '-verbose', '-bs', str(BATCH_SIZE), '-exportELBOs',
                         '-optimizer', 'adagrad'])

    with open('generated/sgavi_rmsprop_elbos.pkl', 'r') as input:
        data = pkl.load(input)
        elbos_rmsprop = data['elbos']
    with open('generated/sgavi_adam_elbos.pkl', 'r') as input:
        data = pkl.load(input)
        elbos_adam = data['elbos']
    with open('generated/sgavi_adadelta_elbos.pkl', 'r') as input:
        data = pkl.load(input)
        elbos_adadelta = data['elbos']
    with open('generated/sgavi_adagrad_elbos.pkl', 'r') as input:
        data = pkl.load(input)
        elbos_adagrad = data['elbos']

    plt.style.use('seaborn-darkgrid')

    # Plot ELBOs
    plt.plot(np.arange(len(elbos_rmsprop)), elbos_rmsprop)
    plt.plot(np.arange(len(elbos_adam)), elbos_adam)
    plt.plot(np.arange(len(elbos_adadelta)), elbos_adadelta)
    plt.plot(np.arange(len(elbos_adagrad)), elbos_adagrad)
    plt.ylabel('ELBO')
    plt.xlabel('Iterations')
    plt.legend(OPTIMIZERS, loc='lower right')
    plt.savefig('generated/elbos.png')
    plt.show()
    plt.gcf().clear()

    # Plot normalized ELBOs
    plt.style.use('seaborn-darkgrid')
    plt.plot(np.arange(len(elbos_rmsprop)), -np.log(np.absolute(elbos_rmsprop)))
    plt.plot(np.arange(len(elbos_adam)), -np.log(np.absolute(elbos_adam)))
    plt.plot(np.arange(len(elbos_adadelta)),
             -np.log(np.absolute(elbos_adadelta)))
    plt.plot(np.arange(len(elbos_adagrad)), -np.log(np.absolute(elbos_adagrad)))
    plt.ylabel('ELBO')
    plt.xlabel('Iterations')
    plt.legend(OPTIMIZERS, loc='lower right')
    plt.savefig('generated/standarized_elbos.png')
    plt.show()
    plt.gcf().clear()


if __name__ == '__main__': main()


