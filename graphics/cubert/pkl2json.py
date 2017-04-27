# -*- coding: UTF-8 -*-

"""
Data format to Cubert data visualization
"""

import argparse
import csv
import json
import pickle as pkl
import sys

from sklearn.preprocessing import MinMaxScaler

"""
Parameters:
    * input: Input path (PKL file with 3D points)
    * assignments: Cluster assignments (CSV file)
    * output: Output path (JSON file for Cubert visualization)

Execution:
    python pkl2json.py -input mallorca_nnint50_pca50.pkl 
                       -assignments assignments.csv 
                       -output mallorca_nnint50_pca50.json
"""

parser = argparse.ArgumentParser(description='pkl2Json')
parser.add_argument('-input', metavar='input', type=str, default='')
parser.add_argument('-assignments', metavar='assignments', type=str, default='')
parser.add_argument('-output', metavar='output', type=str, default='')
args = parser.parse_args()

INPUT = args.input
ASSIGNMENTS = args.assignments
OUTPUT = args.output


def main():
    try:
        if not ('.pkl' in INPUT): raise Exception('input_format')
        if not('.csv' in ASSIGNMENTS): raise Exception('assignments_format')
        if not ('.json' in OUTPUT): raise Exception('output_format')

        with open(INPUT, 'r') as input:
            data = pkl.load(input)
            xn = data['xn']
        N, D = xn.shape

        print('N: {}'.format(N))
        print('D: {}'.format(D))
        if D != 3:
            raise Exception('dimension_error')

        l_format = []
        scaler = MinMaxScaler(feature_range=(-100, 100))
        xn = scaler.fit_transform(xn)
        for row in xn:
            l_format.append({'x': '{}'.format(row[0]), 'y': '{}'.format(row[1]),
                             'z': '{}'.format(row[2])})

        with open(ASSIGNMENTS, 'rb') as assignments:
            reader = csv.reader(assignments, delimiter=';')
            reader.next()
            i = 0
            for row in reader:
                l_format[i]['cid'] = '{}'.format(float(row[0]))
                i += 1

        d_json = {'points': l_format}
        json.dump(d_json, open(OUTPUT, 'wb'))

    except IOError:
        print('File not found!')
    except Exception as e:
        if e.args[0] == 'input_format': print('Input must be a pkl file')
        if e.args[0] == 'assignments_format':
            print('Assignments must be a csv file')
        elif e.args[0] == 'output_format': print('Output must be a json file')
        elif e.args[0] == 'dimension_error': print('Data must be 3D')
        else:
            print('Unexpected error: {}'.format(sys.exc_info()[0]))
            raise


if __name__ == '__main__': main()
