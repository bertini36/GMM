# -*- coding: UTF-8 -*-

"""
Script to time different Univariate Gaussian inferences
"""

import os
import sys
import string
import subprocess

PATH = '../tfInference/UnivariateGaussian/'
ITERATIONS = 10

def main():

	f_times = open('ugm_times.out', 'w')
	inferences = ['coordAsc/ugm_cavi', 'coordAsc/ugm_cavi_linesearch', 'gradAsc/ugm_gavi']

	for inference in inferences:
		for i in xrange(ITERATIONS):
			script = '{}{}.py'.format(PATH, inference)
			output = subprocess.check_output(['python', script, '--timing', '--getNIter', '--no-debug'])
			time = output.split('\n')[0]
			time = time.split(': ')[1]
			time = float(time.split(' ')[0])
			iters = output.split('\n')[1]
			iters = int(iters.split(': ')[1])

if __name__ == '__main__': main()