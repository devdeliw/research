import pandas as pd
import numpy as np
from astropy.table import Table
from red_clump_riemann import Run_Riemann
import os

zeropoints = {
    'Region': ['NRCB1', 'NRCB2', 'NRCB3', 'NRCB4', 'NRCB5'],
    'F115W': [25.92, 25.95, 25.97, 25.98, 0],
    'F212N': [22.12, 22.15, 22.15, 22.23, 0], 
    'F323N': [0, 0, 0, 0, 21.14], 
    'F405N': [0, 0, 0, 0, 20.9], 
}

spisea_filters= pd.DataFrame(
    ['jwst,F115W', 'jwst,F212N', 'jwst,F323N', 'jwst,F405N'],
    index=['F115W', 'F212N', 'F323N', 'F405N'],
    columns = ['SPISEA'])

x_range_NRCB1 = { 
	'x1' : ['F115W', 'F212N', 'F323N', 'F405N'], 
	'F115W' : [None, [4.9, 9], [5.7, 10.1], [6.2, 11.6]], 
	'F212N' : [None, None, None, [1.2, 2.9]], 
	'F323N' : [None, None, None, [0.4, 3]]}

x_range_NRCB2 = { 
	'x1' : ['F115W', 'F212N', 'F323N', 'F405N'], 
	'F115W' : [None, [4.9, 9.3], [5.8, 10.2], [6.3, 11.35]], 
	'F212N' : [None, None, None, [1.2, 3]], 
	'F323N' : [None, None, None, [0.4, 3]]}

x_range_NRCB3 = { 
	'x1' : ['F115W', 'F212N', 'F323N', 'F405N'], 
	'F115W' : [None, [5.1, 9.6], [6.25, 10.4], [6.5, 11.8]], 
	'F212N' : [None, None, None, [1.3, 3]], 
	'F323N' : [None, None, None, [0.4, 3]]}

x_range_NRCB4 = { 
	'x1' : ['F115W', 'F212N', 'F323N', 'F405N'], 
	'F115W' : [None, [5, 9.5], [6, 10.2], [6.4, 11.8]], 
	'F212N' : [None, None, None, [1.2, 3]], 
	'F323N' : [None, None, None, [0.4, 3]]}

f115w_x1_f212n_x2 = { 
    'F115W' : [[(6.3, 21.3), (9, 25.2)], [(6.3, 22), (9, 25.9)]],
    'F212N' : [[(6.2, 15.5), (7.9, 16.3)], [(6.65, 15.2), (8.35, 16.0)]]}

f115w_x1_f323n_x2 = {
    'F115W' : [[(6.9, 21.45), (8.5, 23.65)], [(7.5, 21.35), (9.1, 23.55)]],
    'F323N' : [[(7.4, 13.65), (10, 15.05)], [(7.2, 14.35), (9.8, 15.75)]]}

f115w_x1_f405n_x2 = { 
    'F115W' : [[(7.85, 21), (10, 23.5)], [(8, 21.95), (10.15, 24.45)]],
    'F405N' : [[(7.9, 13.4), (10.5, 13.9)], [(8.4, 14), (11, 14.5)]]}

f212n_x1_f323n_x2 = {
    'F212N' : 0,
    'F323N' : 0}

f212n_x1_f405n_x2 = {
    'F212N' : [[(1.6, 14.9), (2.15, 15.85)], [(1.55, 15.5), (2.1, 16.45)]],
    'F405N' : [[(1.5, 13.35), (2.2, 13.75)], [(1.6, 13.9), (2.3, 14.3)]]}

f323n_x1_f405n_x2 = { 
    'F323N' : [[(0.5, 13.7), (2, 15.75)], [(0.5, 14.8), (2, 16.85)]],
    'F405N' : [[(0.5, 14.3), (2.6, 15)], [(0.7, 13), (2.8, 13.7)]]}

f115w_x1 = { 
    'F212N' : f115w_x1_f212n_x2,
    'F323N' : f115w_x1_f323n_x2,
    'F405N' : f115w_x1_f405n_x2}

f212n_x1 = { 
    'F323N' : f212n_x1_f323n_x2,
    'F405N' : f212n_x1_f405n_x2}

f323n_x1 = { 
    'F405N' : f323n_x1_f405n_x2}

cutoffs = pd.DataFrame({ 
    'F115W' : f115w_x1,
    'F212N' : f212n_x1,
    'F323N' : f323n_x1})

zeropoints = pd.DataFrame(zeropoints).set_index('Region')

x_range_NRCB1 = pd.DataFrame(x_range_NRCB1).set_index('x1')
x_range_NRCB2 = pd.DataFrame(x_range_NRCB2).set_index('x1')
x_range_NRCB3 = pd.DataFrame(x_range_NRCB3).set_index('x1')
x_range_NRCB4 = pd.DataFrame(x_range_NRCB4).set_index('x1')

fits ='~/research/work/catalogs/dr2/jwst_init_NRCB.fits'
catalog = Table.read(fits, format='fits')

# Add what you want to run. The indexing of each subarray is x1, x2, y
# For an x1 - x2 vs y CMD. 
script = [[['NRCB1', 'F115W'], ['NRCB1', 'F212N'], ['NRCB1', 'F115W']],
		  [['NRCB1', 'F115W'], ['NRCB1', 'F212N'], ['NRCB1', 'F212N']],
		  [['NRCB2', 'F115W'], ['NRCB2', 'F212N'], ['NRCB2', 'F115W']],
		  [['NRCB2', 'F115W'], ['NRCB2', 'F212N'], ['NRCB2', 'F212N']],
		  [['NRCB3', 'F115W'], ['NRCB3', 'F212N'], ['NRCB3', 'F115W']],
		  [['NRCB3', 'F115W'], ['NRCB3', 'F212N'], ['NRCB3', 'F212N']],
		  [['NRCB4', 'F115W'], ['NRCB4', 'F212N'], ['NRCB4', 'F115W']],
		  [['NRCB4', 'F115W'], ['NRCB4', 'F212N'], ['NRCB4', 'F212N']],
		  [['NRCB1', 'F115W'], ['NRCB5', 'F323N'], ['NRCB1', 'F115W']],
		  [['NRCB1', 'F115W'], ['NRCB5', 'F323N'], ['NRCB5', 'F323N']],
		  [['NRCB2', 'F115W'], ['NRCB5', 'F323N'], ['NRCB2', 'F115W']],
		  [['NRCB2', 'F115W'], ['NRCB5', 'F323N'], ['NRCB5', 'F323N']],
		  [['NRCB3', 'F115W'], ['NRCB5', 'F323N'], ['NRCB3', 'F115W']],
		  [['NRCB3', 'F115W'], ['NRCB5', 'F323N'], ['NRCB5', 'F323N']],
		  [['NRCB4', 'F115W'], ['NRCB5', 'F323N'], ['NRCB4', 'F115W']],
		  [['NRCB4', 'F115W'], ['NRCB5', 'F323N'], ['NRCB5', 'F323N']],
		  [['NRCB1', 'F115W'], ['NRCB5', 'F405N'], ['NRCB1', 'F115W']],
		  [['NRCB1', 'F115W'], ['NRCB5', 'F405N'], ['NRCB5', 'F405N']],
		  [['NRCB2', 'F115W'], ['NRCB5', 'F405N'], ['NRCB2', 'F115W']],
		  [['NRCB2', 'F115W'], ['NRCB5', 'F405N'], ['NRCB5', 'F405N']],
		  [['NRCB3', 'F115W'], ['NRCB5', 'F405N'], ['NRCB3', 'F115W']],
		  [['NRCB3', 'F115W'], ['NRCB5', 'F405N'], ['NRCB5', 'F405N']],
		  [['NRCB4', 'F115W'], ['NRCB5', 'F405N'], ['NRCB4', 'F115W']],
		  [['NRCB4', 'F115W'], ['NRCB5', 'F405N'], ['NRCB5', 'F405N']],
		  [['NRCB1', 'F212N'], ['NRCB5', 'F405N'], ['NRCB1', 'F212N']],
		  [['NRCB1', 'F212N'], ['NRCB5', 'F405N'], ['NRCB5', 'F405N']],
		  [['NRCB2', 'F212N'], ['NRCB5', 'F405N'], ['NRCB2', 'F212N']],
		  [['NRCB2', 'F212N'], ['NRCB5', 'F405N'], ['NRCB5', 'F405N']],
		  [['NRCB3', 'F212N'], ['NRCB5', 'F405N'], ['NRCB3', 'F212N']],
		  [['NRCB3', 'F212N'], ['NRCB5', 'F405N'], ['NRCB5', 'F405N']],
		  [['NRCB4', 'F212N'], ['NRCB5', 'F405N'], ['NRCB4', 'F212N']],
		  [['NRCB4', 'F212N'], ['NRCB5', 'F405N'], ['NRCB5', 'F405N']],
]

script = [[['NRCB1', 'F115W'], ['NRCB1', 'F212N'], ['NRCB1', 'F115W']]]

# ------------------------------------------------------------------- #

for i in range(len(script)): 

	catalog1filt = script[i][0][1]
	catalog2filt = script[i][1][1]
	catalogyfilt = script[i][2][1]

	region1 = script[i][0][0]
	region2 = script[i][1][0]
	regiony = script[i][2][0]

	filters = [spisea_filters['SPISEA'][catalog1filt], spisea_filters['SPISEA'][catalog2filt]]
	parallel_cutoff1 = cutoffs[catalog1filt][catalog2filt][catalogyfilt][0]
	parallel_cutoff2 = cutoffs[catalog1filt][catalog2filt][catalogyfilt][1]

	x_range = []

	if region1 == 'NRCB1': 
		x_range = x_range_NRCB1[catalog1filt][catalog2filt]
	if region1 == 'NRCB2': 
		x_range = x_range_NRCB2[catalog1filt][catalog2filt]
	if region1 == 'NRCB3': 
		x_range = x_range_NRCB3[catalog1filt][catalog2filt]
	if region1 == 'NRCB4': 
		x_range = x_range_NRCB4[catalog1filt][catalog2filt]

	catalog1zp = zeropoints[catalog1filt][region1]
	catalog2zp = zeropoints[catalog2filt][region2]

	ns = [10, 11, 12, 13, 14, 15] 
	n = 3
	image_path = f"/Users/devaldeliwala/research/work/plots&data/rc_analysis_v2_plots/{region1}/{catalog1filt}-{catalog2filt}/vs{catalogyfilt}/"
	hists = [True, False]

	if not os.path.isdir(image_path):
		os.makedirs(image_path)

	print("")
	print(f"##################################################")
	print(f"Current: {region1} {catalog1filt} - {region2} {catalog2filt} vs. {regiony} {catalogyfilt}")
	print(f"##################################################")
	print("")

	class_ = Run_(
	    catalog=catalog,
	    catalog1name=catalog1filt, 
	    catalog2name=catalog2filt, 
	    catalogyname=catalogyfilt, 
	    region1=region1, 
	    region2=region2, 
	    regiony=regiony, 
	    parallel_cutoff1=parallel_cutoff1, 
	    parallel_cutoff2=parallel_cutoff2, 
	    x_range=x_range, 
	    n=n, 
	    image_path=image_path, 
	    show_hists=hists[0], 
	    catalog1zp=catalog1zp, 
	    catalog2zp=catalog2zp 
	)	

	"""
	Run(
		catalog, catalog1filt, catalog2filt, catalogyfilt, 
		region1, region2, regiony, 
		parallel_cutoff1, parallel_cutoff2, 
		x_range, ns, image_path, hists[0], 
		catalog1zp, catalog2zp, 
	).run()
	Run(
		catalog, catalog1filt, catalog2filt, catalogyfilt, 
		region1, region2, regiony,  
		parallel_cutoff1, parallel_cutoff2, 
		x_range, ns, image_path, hists[1], 
		catalog1zp, catalog2zp, 
	).run()
	"""
	
	class_.run(ns = [10, 11, 12, 13, 14, 15])







