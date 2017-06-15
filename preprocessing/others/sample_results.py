# -*- coding: UTF-8 -*-

"""
Results sampler
"""

import csv
import random
import sys

TRACKS_PATH = 'porto_int50.csv'
ASSIGNMENTS_PATH = 'assignments.csv'
NEW_TRACKS_PATH = 'porto_int50_subset.csv'
NEW_ASSIGNMENTS_PATH = 'assignments_subset.csv'
N_SAMPLES = 1500


def main():
    try:
        if not ('.csv' in TRACKS_PATH): raise Exception('input_format')
        if not ('.csv' in ASSIGNMENTS_PATH): raise Exception('input_format')
        if not ('.csv' in NEW_TRACKS_PATH): raise Exception('output_format')
        if not ('.csv' in NEW_ASSIGNMENTS_PATH):
            raise Exception('output_format')

        with open(TRACKS_PATH, 'rb') as tracks, \
                open(ASSIGNMENTS_PATH, 'rb') as assignments, \
                open(NEW_TRACKS_PATH, 'w') as new_tracks, \
                open(NEW_ASSIGNMENTS_PATH, 'w') as new_assignments:

            tracks_reader = csv.reader(tracks, delimiter=';')
            assignments_reader = csv.reader(assignments, delimiter=';')
            tracks_reader.next()
            assignments_reader.next()

            tracks_writer = csv.writer(new_tracks, delimiter=';', quotechar='',
                                       escapechar='\\', quoting=csv.QUOTE_NONE)
            assignments_writer = csv.writer(new_assignments, delimiter=';',
                                            quotechar='', escapechar='\\',
                                            quoting=csv.QUOTE_NONE)
            tracks_writer.writerow(['Points'])
            assignments_writer.writerow(['zn'])

            lines = 0
            for _ in assignments_reader: lines += 1
            assignments.seek(0)
            assignments_reader.next()

            i_samples = random.sample(range(1, lines), N_SAMPLES)
            for i, track in enumerate(tracks_reader):
                ass = assignments_reader.next()
                if i in i_samples:
                    tracks_writer.writerow([track[0]])
                    assignments_writer.writerow(ass)

    except IOError:
        print('File not found!')
    except Exception as e:
        if e.args[0] == 'input_format': print('Input must be a CSV file')
        elif e.args[0] == 'output_format': print('Output must be a CSV file')
        else:
            print('Unexpected error: {}'.format(sys.exc_info()[0]))
            raise


if __name__ == '__main__': main()
