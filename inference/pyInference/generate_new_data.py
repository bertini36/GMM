# -*- coding: UTF-8 -*-

"""
Generate and visualize new data with the inferred variational parameters
"""

import argparse
import pickle as pkl
import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import invwishart

"""
Parameters:
    * variationalParameters: Path to pkl that contains variational parameters
    * N: Number of examples to generate
    * K: If > -1 only generates samples of cluster K
    * output: Path to pkl file

Execution:
    python generate_new_data.py -variationalParameters vp.pkl -N 500 
                                -output generated_points.pkl
"""

parser = argparse.ArgumentParser(description='New data generator')
parser.add_argument('-variationalParameters', metavar='variationalParameters',
                    type=str, default='generated/variational_parameters.pkl')
parser.add_argument('-N', metavar='N', type=int, default=100)
parser.add_argument('-selectK', metavar='selectK', type=int, default=-1)
parser.add_argument('-output', metavar='output',
                    type=str, default='generated/new_data.pkl')
args = parser.parse_args()

N = args.N
INPUT = args.variationalParameters
OUTPUT = args.output


def main():
    try:
        if not ('.pkl' in INPUT): raise Exception('input_format')
        if not ('.pkl' in OUTPUT): raise Exception('output_format')

        # Get variational parameters
        with open('{}'.format(INPUT), 'r') as input:
            data = pkl.load(input)
            lambda_pi = data['lambda_pi']
            lambda_m = data['lambda_m']
            lambda_beta = data['lambda_beta']
            lambda_nu = data['lambda_nu']
            lambda_w = data['lambda_w']
            K = data['K']
            D = data['D']

        cn = []
        xn = []

        if args.selectK == -1:
            pi = np.random.dirichlet(lambda_pi)
            mus = []
            sigmas = []
            for k in range(K):
                mus.append(np.random.normal(lambda_m[k, :], 1 / lambda_beta[k]))
                sigmas.append(invwishart.rvs(lambda_nu[k], lambda_w[k, :, :]))
            for n in range(N):
                cn.append(np.random.choice(len(pi), p=pi))
                xn.append(np.random.multivariate_normal(mus[cn[n]],
                                                        sigmas[cn[n]]))
        else:
            mu = np.random.normal(lambda_m[args.selectK, :],
                                  1 / lambda_beta[args.selectK])
            sigma = invwishart.rvs(lambda_nu[args.selectK],
                                   lambda_w[args.selectK, :, :])
            for n in range(N):
                xn.append(np.random.multivariate_normal(mu, sigma))
        xn = np.asarray(xn)

        if D == 2:
            plt.scatter(xn[:, 0], xn[:, 1], c=cn, cmap=cm.gist_rainbow, s=5)
            plt.title('Generated data')
            plt.show()
        elif D == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xn[:, 0], xn[:, 1], xn[:, 2],
                       c=cn, cmap=cm.gist_rainbow, s=5)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.show()

        print('xn[0:10]: {}'.format(xn[0:10, :]))
        print('Length xn: {}'.format(len(xn)))
        print('Element length: {}'.format(len(xn[0])))

        with open(OUTPUT, 'w') as output:
            pkl.dump({'xn': xn}, output)

    except IOError:
        print('File not found!')
    except Exception as e:
        if e.args[0] == 'input_format':
            print('Input must be a pkl file')
        elif e.args[0] == 'output_format':
            print('Input must be a pkl file')
        else:
            print('Unexpected error: {}'.format(sys.exc_info()[0]))
            raise


if __name__ == '__main__': main()
