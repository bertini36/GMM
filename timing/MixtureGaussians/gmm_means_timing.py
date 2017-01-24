# -*- coding: UTF-8 -*-

"""
Script to time different Mixture of Gaussians inferences
"""

import csv
import subprocess

PATH = '../../tfInference/MixtureGaussians/KnownMeans/'


def main():
    with open('csv/gmm_means_times.csv', 'wb') as csvfile:

        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['Inference type', 'Dataset size',
                         'K', 'Time', 'Iterations', 'ELBO'])

        inferences = ['coordAsc/gmm_means_cavi', 'gradAsc/gmm_means_gavi']
        nelements = [100, 500, 1000]
        iterations = 1

        for inference in inferences:
            for nelem in nelements:
                script = '{}{}.py'.format(PATH, inference)
                for k in [2, 4, 8]:
                    total_time = 0
                    total_iters = 0
                    total_elbos = 0
                    for i in xrange(iterations):
                        output = subprocess.check_output(
                            ['python', script, '-dataset',
                             '../../data/data_k{}_{}.pkl'
                                 .format(str(k), str(nelem)),
                             '-k', str(k), '--timing', '--getNIter',
                             '--getELBOs', '--no-debug', '--no-plot'])
                        time = float(((output.split('\n')[0])
                                      .split(': ')[1]).split(' ')[0])
                        iters = int((output.split('\n')[1]).split(': ')[1])
                        elbos = [float(lb)
                                 for lb in (((output.split('\n')[2])
                                             .split('[')[1]).split(']')[0])
                                     .split(',')]
                        total_time += time
                        total_iters += iters
                        total_elbos += elbos[-1]
                    writer.writerow([inference, nelem, k, total_time/iterations,
                                     total_iters/iterations,
                                     total_elbos/iterations])

            with open('csv/{}_elbos_500.csv'
                      .format(inference.split('/')[1]),
                      'wb') as csvfile2:
                output = subprocess.check_output(
                    ['python', script, '-dataset',
                     '../../data/data_k{}_500.pkl'.format(str(2)),
                     '--getELBOs', '--no-debug', '--no-plot'])
                elbos = [float(lb)
                         for lb in (((output.split('\n')[2])
                                     .split('[')[1]).split(']')[0])
                             .split(',')]
                writer2 = csv.writer(csvfile2, delimiter=';')
                writer2.writerow(['Iteration', 'ELBO'])
                for i, elbo in enumerate(elbos):
                    writer2.writerow([i, elbo])


if __name__ == '__main__': main()
