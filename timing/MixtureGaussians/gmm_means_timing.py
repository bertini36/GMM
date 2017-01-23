# -*- coding: UTF-8 -*-

import csv
import subprocess

PATH = '../../tfInference/MixtureGaussians/KnownMeans/'


def main():
    """
    Script to time different Mixture of Gaussians inferences
    """
    with open('csv/gmm_means_times.csv', 'wb') as csvfile:

        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(
            ['Inference type', 'Dataset size', 'K', 'Time', 'Iterations',
             'ELBO'])

        inferences = ['coordAsc/gmm_means_cavi', 'gradAsc/gmm_means_gavi']
        nelements = [100, 500, 1000]
        iterations = 10

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
                             '../../data/data_k{}_{}.pkl'.format(str(k), str(nelem)),
                             '-k', str(k), '--timing', '--getNIter',
                             '--getELBO', '--no-debug', '--no-plot'])
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
                    writer.writerow([inference, nelem, k, total_time / iterations,
                                     total_iters / iterations,
                                     total_elbos / iterations])

            with open('csv/{}_elbos_1000.csv'
                      .format(inference.split('/')[1]),
                      'wb') as csvfile2:
                output = subprocess.check_output(
                    ['python', script, '-dataset',
                     '../../data/data_k{}_1000.pkl'.format(str(2)),
                     '--getELBO', '--no-debug'])
                elbos = ((output.split(': [')[1]).split(']')[0]).split(
                    ', ')
                writer2 = csv.writer(csvfile2, delimiter=';')
                writer2.writerow(['Iteration', 'ELBO'])
                for i, elbo in enumerate(elbos):
                    writer2.writerow([i, elbo])


if __name__ == '__main__': main()
