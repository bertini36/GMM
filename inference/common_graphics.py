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
    - Its memory usage
"""

from __future__ import absolute_import

import pickle as pkl
import subprocess

import matplotlib.pyplot as plt
import numpy as np

DATASET = '../data/synthetic/2D/k2/data_k2_1000.pkl'
MAX_ITER = 100
K = 2
VERBOSE = False
BATCH_SIZE = 100


def main():
    subprocess.call(['python', 'python/gmm_cavi.py', '-dataset', DATASET,
                     '-maxIter', str(MAX_ITER),
                     '-k', str(K), '-verbose', '-exportELBOs'])
    subprocess.call(['python', 'tensorflow/gmm_gavi.py', '-dataset', DATASET,
                     '-maxIter', str(MAX_ITER),
                     '-k', str(K), '-verbose', '-exportELBOs'])
    subprocess.call(['python', 'python/gmm_scavi.py', '-dataset', DATASET,
                     '-maxIter', str(MAX_ITER), '-k', str(K),
                     '-verbose', '-bs', str(BATCH_SIZE), '-exportELBOs'])
    subprocess.call(['python', 'tensorflow/gmm_sgavi.py', '-dataset', DATASET,
                     '-maxIter', str(MAX_ITER), '-k', str(K),
                     '-verbose', '-bs', str(BATCH_SIZE), '-exportELBOs'])

    with open('generated/cavi_elbos.pkl', 'r') as input:
        data = pkl.load(input)
        elbos_cavi = data['elbos']
    with open('generated/gavi_elbos.pkl', 'r') as input:
        data = pkl.load(input)
        elbos_gavi = data['elbos']
    with open('generated/scavi_elbos.pkl', 'r') as input:
        data = pkl.load(input)
        elbos_scavi = data['elbos']
    with open('generated/sgavi_elbos.pkl', 'r') as input:
        data = pkl.load(input)
        elbos_sgavi = data['elbos']

    # Plot ELBOs
    plt.style.use('seaborn-darkgrid')
    plt.plot(np.arange(len(elbos_cavi)), elbos_cavi)
    plt.plot(np.arange(len(elbos_gavi)), elbos_gavi)
    plt.plot(np.arange(len(elbos_scavi)), elbos_scavi)
    plt.plot(np.arange(len(elbos_sgavi)), elbos_sgavi)
    plt.ylabel('ELBO')
    plt.xlabel('Iterations')
    plt.savefig('elbos.png')
    plt.legend(['CAVI', 'GAVI', 'SCAVI', 'SGAVI'], loc='upper left')
    plt.show()


if __name__ == '__main__': main()
