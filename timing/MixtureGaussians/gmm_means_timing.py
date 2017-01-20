# -*- coding: UTF-8 -*-

"""
Script to time different Mixture of Gaussians inferences
"""

import os
import csv
import sys
import string
import subprocess

PATH = '../tfInference/MixtureGaussians/KnownMeans/'

def main():

	with open('gmm_means_times.csv', 'wb') as csvfile:

		writer = csv.writer(csvfile, delimiter=';')
		writer.writerow(['Inference type', 'Dataset size', 'K', 'Time', 'Iterations', 'ELBO'])

		inferences = ['coordAsc/gmm_means_cavi', 'gradAsc/gmm_means_gavi']
		nelements = [100, 500, 1000]
		ks = [2, 4, 8]
		iterations = 10

		for inference in inferences:
			for n in nelements:
				for k in ks:
					total_time = 0
					total_iters = 0
					total_elbos = 0
					for i in xrange(iterations):
						script = '{}{}.py'.format(PATH, inference)
						output = subprocess.check_output(['python', script, '-dataset', '../data/data_k{}_{}.pkl'.format(str(k), str(n)),
														 '-k', str(k), '--timing', '--getNIter', '--getELBO', '--no-debug', '--no-plot'])
						time = output.split('\n')[0]
						time = time.split(': ')[1]
						time = float(time.split(' ')[0])
						iters = output.split('\n')[1]
						iters = int(iters.split(': ')[1])
						elbo = output.split('\n')[2]
						elbo = elbo.split(': [[')[1]
						elbo = float(elbo.split(']]')[0])
						total_time += time
						total_iters += iters
						total_elbos += elbo
					writer.writerow([inference, n, k, total_time/iterations, 
									total_iters/iterations, total_elbos/iterations])

if __name__ == '__main__': main()