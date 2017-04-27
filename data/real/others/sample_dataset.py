# -*- coding: UTF-8 -*-

"""
Data sampler
"""

import argparse
import csv
import random
import sys

"""
Parameters:
    * input: Input path (CSV with ; delimiter)
    * output: Output path (CSV with ; delimiter)
    * s: Number of samples

Execution:
    python sample_dataset.py -input mallorca.csv
                             -output /mallorca_subset.csv -s 500
"""

parser = argparse.ArgumentParser(description='Data sampler')
parser.add_argument('-input', metavar='input', type=str, default='')
parser.add_argument('-output', metavar='output', type=str, default='')
parser.add_argument('-s', metavar='s', type=int, default=100)
args = parser.parse_args()


def format_track(track):
    """
    Format track from String to points list
    :param track: Track as a string
    :return: Track as a Python list of points
    """
    new_track = []
    for point in track.split('[[')[1].split(']]')[0].split('], ['):
        new_track.append([float(n) for n in point.split(', ')])
    return new_track


def count_lines(input):
    """
    Count number of lines in the CSV
    :param input: Input file
    :output: Number of lines
    """
    reader = csv.reader(input)
    reader.next()
    n = 0
    for _ in reader: n += 1
    return n


def main():
    try:
        if not('.csv' in args.input): raise Exception('input_format')
        if not('.csv' in args.output): raise Exception('output_format')

        with open(args.input, 'rb') as input, open(args.output, 'wb') as output:
            lines = count_lines(input)
            writer = csv.writer(output, delimiter=';',
                                escapechar='', quoting=csv.QUOTE_NONE)
            input.seek(0)
            reader = csv.reader(input, delimiter=';')
            reader.next()
            writer.writerow(['Points'])

            print('Lines: {}'.format(lines))

            i_samples = random.sample(range(1, lines), args.s)
            for i, track in enumerate(reader):
                if i in i_samples:
                    writer.writerow([format_track(track[0])])

    except IndexError:
        print('CSV input file doesn\'t have the correct structure!')
    except IOError:
        print('File not found!')
    except Exception as e:
        if e.args[0] == 'input_format': print('Input must be a CSV file')
        elif e.args[0] == 'output_format': print('Output must be a PKL file')
        else:
            print('Unexpected error: {}'.format(sys.exc_info()[0]))
            raise


if __name__ == '__main__': main()
