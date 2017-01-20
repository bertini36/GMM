# -*- coding: UTF-8 -*-

"""
Script to time different Univariate Gaussian inferences
"""

import os
import csv
import sys
import string
import subprocess

PATH = '../tfInference/UnivariateGaussian/'

def main():

	with open('ugm_times.csv', 'wb') as csvfile:

		writer = csv.writer(csvfile, delimiter=';')
		writer.writerow(['Inference type', 'Dataset size', 'Time', 'Iterations', 'ELBO'])

		inferences = ['coordAsc/ugm_cavi', 'gradAsc/ugm_gavi']
		nelements = [100, 500, 1000]
		iterations = 10

		for inference in inferences:
			for n in nelements:
				total_time = 0
				total_iters = 0
				total_elbos = 0
				for i in xrange(iterations):
					script = '{}{}.py'.format(PATH, inference)
					output = subprocess.check_output(['python', script, '-nElements', str(n), 
													 '--timing', '--getNIter', '--getELBO', '--no-debug'])
					time = output.split('\n')[0]
					time = time.split(': ')[1]
					time = float(time.split(' ')[0])
					iters = output.split('\n')[1]
					iters = int(iters.split(': ')[1])
					elbo = output.split('\n')[2]
					elbo = float(elbo.split(': ')[1])
					total_time += time
					total_iters += iters
					total_elbos += elbo
				writer.writerow([inference, n, total_time/iterations, 
								 total_iters/iterations, total_elbos/iterations])

if __name__ == '__main__': main()