# -*- coding: UTF-8 -*-

"""
Script to time different Univariate Gaussian inferences
"""

import csv
import subprocess

PATH = '../../tfInference/UnivariateGaussian/'


def main():
    with open('csv/ugm_times.csv', 'wb') as csvfile:

        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(
            ['Inference type', 'Dataset size', 'Time', 'Iterations', 'ELBO'])

        inferences = ['coordAsc/ugm_cavi', 'gradAsc/ugm_gavi']
        nelements = [100, 500, 1000, 2000, 4000]
        iterations = 10

        for inference in inferences:
            script = '{}{}.py'.format(PATH, inference)
            for nelem in nelements:
                total_time = 0
                total_iters = 0
                total_elbos = 0
                for i in xrange(iterations):
                    output = subprocess.check_output(
                        ['python', script, '-nElements',
                         str(nelem), '--timing', '--getNIter',
                         '--getELBO', '--no-debug'])
                    time = float(
                        ((output.split('\n')[0]).split(': ')[1]).split(' ')[0])
                    iters = int((output.split('\n')[1]).split(': ')[1])
                    elbos = (
                        ((output.split('\n')[2]).split(': [')[1]).split(']')[
                            0]).split(', ')
                    elbos = [float(lb) for lb in elbos]
                    total_time += time
                    total_iters += iters
                    total_elbos += elbos[-1]
                writer.writerow([inference, nelem, total_time / iterations,
                                 total_iters / iterations,
                                 total_elbos / iterations])

            with open('csv/{}_elbos_1000.csv'
                      .format(inference.split('/')[1]),
                      'wb') as csvfile2:
                output = subprocess.check_output(
                    ['python', script, '-nElements', str(nelements[2]),
                     '--getELBO', '--no-debug'])
                elbos = ((output.split(': [')[1]).split(']')[0]).split(', ')
                writer2 = csv.writer(csvfile2, delimiter=';')
                writer2.writerow(['Iteration', 'ELBO'])
                for i, elbo in enumerate(elbos):
                    writer2.writerow([i, elbo])


if __name__ == '__main__': main()
