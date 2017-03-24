# -*- coding: UTF-8 -*-

""" 
CSV generator from GPX files for Mallorca tracks dataset and filter outliers
"""

import os
import csv
import gpxpy

GPX_DIRECTORY = 'gpx/'
N_FILES = 200
# N_FILES = 1876

with open('tracks.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    writer.writerow(['Points'])
    files = os.listdir(GPX_DIRECTORY)
    for i in range(N_FILES):
        gpx = gpxpy.parse(open('{}{}'.format(GPX_DIRECTORY, files[i])))
        track = gpx.tracks[0]
        segment = track.segments[0]
        points = []
        bad = False
        for point in segment.points:
            if point.latitude < 39. or point.latitude > 40.5 \
                    or point.longitude < 2. or point.longitude > 4.:
                bad = True
                break
            points.append([point.latitude, point.longitude])
        if not bad: writer.writerow([points])
