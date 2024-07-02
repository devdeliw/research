import pandas as pd
import numpy as np

from astropy.table import Table
from ast import literal_eval

from red_clump_riemann import Run_Riemann
from red_clump_rectangle import Run_Rectangle

import os


#-------------------------------------------------------------------------------------------------#
# set your base image paths here

# for the rectangle software:
image_path_rectangle = '/Users/devaldeliwala/research/work/plots&data/red_clump_analysis/rectangle_plots/'

# for the riemann software:
image_path_riemann = '/Users/devaldeliwala/research/work/plots&data/red_clump_analysis/riemann_plots/'

#-------------------------------------------------------------------------------------------------#

array = [0]

for i in range(40): 
	array.append(i)

array.pop(0)

# access script 
script = pd.read_csv("red_clump_analysis_script.csv").drop(array)

# ensure outputted lists aren't strings after pd.read_csv()
for i in ['x_range', 'parallel_cutoff1', 'parallel_cutoff2', 'xlim', 'ylim']: 
    script.loc[:, i] = script.loc[:, i].apply(lambda x: literal_eval(x))

# matched star catalog
fits ='/Users/devaldeliwala/research/work/catalogs/dr2/jwst_init_NRCB.fits'
catalog = Table.read(fits, format='fits')

# don't change, not necessary.
n = 10 

# if you want to show histograms under the curve fits
# default is set to True
hists = True

"""
`ns` provides the `n`s you wish to run the software on.
Each `n` determines how many segments to divide the RC bar by.
The algorithm will run through every `n` and afterward run 
the optimal `n` that minimizes the slope of the RC bar. 

"""

# change, if necessary.
ns = [8, 9, 10, 11, 12, 13, 14, 15]

for index, row in script.iterrows():
   
	region1 = row['region1']
	region2 = row['region2']
	regiony = row['regiony']

	catalog1name = row['catalog1']
	catalog2name = row['catalog2']
	catalogyname = row['catalogy']

	catalog1zp = row['catalog1zp']
	catalog2zp = row['catalog2zp']

	x_range = row['x_range']

	# If non-empty, will use rectangle software 
	xlim = row['xlim']
	ylim = row['ylim']

	parallel_cutoff1 = row['parallel_cutoff1']
	parallel_cutoff2 = row['parallel_cutoff2']

	# Checking if xlim is empty -- use riemann software
	if xlim == 0: 

		image_path = f'{image_path_riemann}/{region1}/{catalog1name}-{catalog2name}/vs{catalogyname}/'

		if not os.path.isdir(image_path):
			os.makedirs(image_path)

		class_ = Run_Riemann(
		    catalog = catalog,
		    catalog1name = catalog1name, 
		    catalog2name = catalog2name, 
		    catalogyname = catalogyname, 
		    region1 = region1, 
		    region2 = region2, 
		    regiony = regiony, 
		    parallel_cutoff1 = parallel_cutoff1, 
		    parallel_cutoff2 = parallel_cutoff2, 
		    x_range = x_range, 
		    n = n, 
		    image_path = image_path, 
		    show_hists = hists, 
		    catalog1zp = catalog1zp, 
		    catalog2zp = catalog2zp)

	# xlim is not NaN -- use rectangle software 
	else: 

		image_path = f'{image_path_rectangle}/{region1}/{catalog1name}-{catalog2name}/vs{catalogyname}/'
		
		if not os.path.isdir(image_path):
			os.makedirs(image_path)

		class_ = Run_Rectangle( 
			catalog = catalog, 
			catalog1name = catalog1name, 
			catalog2name = catalog2name, 
			catalogyname = catalogyname, 
			region1 = region1, 
			region2 = region2, 
			regiony = regiony, 
			xlim = xlim, 
			ylim = ylim, 
			n = n, 
			image_path = image_path, 
			show_hists = hists, 
			catalog1zp = catalog1zp, 
			catalog2zp = catalog2zp)

	# run the algorithm
	class_.run(ns = ns)








