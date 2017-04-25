# -*- coding: UTF-8 -*-

"""
CSV generator from GPX files for Mallorca tracks dataset and filter outliers
"""

import os
import csv
import gpxpy
import numpy as np

GPX_DIRECTORY = 'gpx/'
N_FILES = 1876
LIMIT_LAT = (39.0, 40.5)
LIMIT_LON = (2.0, 3.6)


def main():
    with open('mallorca2.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['Points'])
        files = os.listdir(GPX_DIRECTORY)
        for i in range(N_FILES):
            gpx = gpxpy.parse(open('{}{}'.format(GPX_DIRECTORY, files[i])))
            track = gpx.tracks[0]
            segment = track.segments[0]
            points = []
            bad = False
            time = 0
            pre_secs = None
            for point in segment.points:
                if point.latitude < LIMIT_LAT[0] \
                        or point.latitude > LIMIT_LAT[1] \
                        or point.longitude < LIMIT_LON[0] \
                        or point.longitude > LIMIT_LON[1]:
                    bad = True
                    break
                secs = str(point.time).split(' ')
                if len(secs) > 1:
                    secs = secs[1].split(':')
                    secs = float(secs[0]) * 60 * 60 + float(secs[1]) * 60 + float(secs[2])
                else: secs = 0
                if pre_secs is not None:
                    time += np.absolute(pre_secs-secs)
                pre_secs = secs
                points.append([point.latitude, point.longitude,
                               point.elevation, time])
            if not bad: writer.writerow([points])


if __name__ == '__main__': main()
