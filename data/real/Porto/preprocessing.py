# -*- coding: UTF-8 -*-

""" 
Filter outliers from Porto tracks dataset
"""

import csv

DATA_DIRECTORY = 'porto.csv'

with open(DATA_DIRECTORY, 'rb') as input, open('porto_new.csv'.format(DATA_DIRECTORY), 'wb') as output:
	writer = csv.writer(output)
	reader = csv.reader(input)
	writer.writerow(next(reader))
	i = 0
	for row in reader:
		print(i)
		if '[[' in row[8]:
			points = row[8].split('[[')[1].split(']]')[0].split('],[')
			bad = False
			for point in points:
				p = point.split(',')
				if float(p[1]) < 41.10 or float(p[1]) > 41.25 or float(p[0]) < -8.75 or float(p[0]) > -8.50:
					bad = True
					break
			if not bad: writer.writerow(row)
		i += 1
