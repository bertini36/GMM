# -*- coding: UTF-8 -*-

""" 
Filter outliers from Porto tracks dataset and CSV format
"""

import csv

DATA_DIRECTORY = 'train.csv'
USE_LIMITS = False
LIMIT_LAT = (41.10, 41.25)
LIMIT_LON = (-8.75, -8.50)


def main():
    with open(DATA_DIRECTORY, 'rb') as input, open('porto.csv', 'wb') as output:
        writer = csv.writer(output, delimiter=';')
        reader = csv.reader(input)
        reader.next()
        writer.writerow(['Points'])
        i = 0
        for row in reader:
            print('Track {}'.format(i))
            new_row = []
            if '[[' in row[8]:
                points = row[8].split('[[')[1].split(']]')[0].split('],[')
                bad = False
                for point in points:
                    p = point.split(',')
                    if USE_LIMITS:
                        if float(p[1]) < LIMIT_LAT[0] \
                                or float(p[1]) > LIMIT_LAT[1] \
                                or float(p[0]) < LIMIT_LON[0] \
                                or float(p[0]) > LIMIT_LON[1]:
                            bad = True
                            break
                    new_row.append([float(p[0]), float(p[1])])
                if not bad: writer.writerow([new_row])
            i += 1


if __name__ == '__main__': main()
