#!/usr/bin/env python

"convert three-dimensional data with labels from CSV to JSON"

import sys
import pandas as pd
import json

from sklearn.preprocessing import MinMaxScaler

try:
	input_file = sys.argv[1]
except IndexError:
	input_file = 'data.csv'
	
try:
	input_file = sys.argv[2]
except IndexError:
	output_file = 'data.json'
	
# load

d = pd.read_csv( input_file )
assert( set( d.columns ) == set([ 'cid', 'x', 'y', 'z' ]))

# scale 

scaler = MinMaxScaler( feature_range=( -100, 100 ))
d[[ 'x', 'y', 'z' ]] = scaler.fit_transform( d[[ 'x', 'y', 'z' ]])

# labels

tmp_cid = d.cid.copy()
new_label = 0
for label in sorted( d.cid.unique()):
	tmp_cid[d.cid == label] = new_label
	new_label += 1
d.cid = tmp_cid

# json
	
d_json = { 'points': json.loads( d.astype( str ).to_json( None, orient= 'records' )) }
json.dump( d_json, open( output_file, 'wb' ))

