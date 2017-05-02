# -*- coding: UTF-8 -*-

""" 
CSV generator from GPX files for Mallorca tracks dataset and filter outliers
Points CSV: [Latitude, Longitude]
"""

import os
import csv
import gpxpy

GPX_DIRECTORY = 'gpx/'
N_FILES = 1876
LIMIT_LAT = (39.0, 40.5)
LIMIT_LON = (2.0, 3.6)


def main():
    with open('mallorca.csv', 'wb') as csvfile:
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
                if point.latitude < LIMIT_LAT[0] \
                        or point.latitude > LIMIT_LAT[1] \
                        or point.longitude < LIMIT_LON[0] \
                        or point.longitude > LIMIT_LON[1]:
                    bad = True
                    break
                points.append([point.latitude, point.longitude])
            if not bad: writer.writerow([points])


if __name__ == '__main__': main()