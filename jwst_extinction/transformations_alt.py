import numpy as np
import pandas as pd
from flystar import match, transforms, plots, align
from flystar.starlists import StarList
from flystar.startables import StarTable
from astropy.table import Table, Column, vstack
import datetime
import copy
import os
import pdb
import time
import warnings
from astropy.utils.exceptions import AstropyUserWarning
import matplotlib.pyplot as plt
import math
from color_mag_diagram import *

df = Table.read('catalogs/catalog115j.csv')
df2 = Table.read('catalogs/catalog212j.csv')
df3 = Table.read('catalogs/catalog323j.csv')
df4 = Table.read('catalogs/catalog444j.csv')

msc = align.MosaicSelfRef([df, df2, df3, df4], iters=1,
                            dr_tol=[15], dm_tol=[15],
                            outlier_tol=[None], mag_lim= None,
                            trans_class=transforms.PolyTransform,
                            trans_args=[{'order': 1}],
                            use_vel=False,
                            ref_index = 0,
                            mag_trans=False,
                            weights='both,std',
                            init_guess_mode='name', verbose=False
                         )
msc.fit()
trans_list = msc.trans_args
stars_table = msc.ref_table

# -------------------------#

match115405 = []
match115212 = []
match115323 = []

for i in range(len(stars_table)): 
    if  (
            math.isnan(stars_table[i]['x'][0]) != True and 
            math.isnan(stars_table[i]['x'][3]) != True
        ): 
        match115405.append(i)

for i in range(len(stars_table)):
    if  (
            math.isnan(stars_table[i]['x'][0]) != True and 
            math.isnan(stars_table[i]['x'][1]) != True
        ): 
        match115212.append(i)

for i in range(len(stars_table)):
    if  (
            math.isnan(stars_table[i]['x'][0]) != True and
            math.isnan(stars_table[i]['x'][2]) != True
        ):
        match115323.append(i)

x115405, x115323, x115212       = [], [], []
x212115, x323115, x405115       = [], [], []
y115405, y115323, y115212       = [], [], []
y212115, y323115, y405115       = [], [], []

m115405, m115212, m115323       = [], [], []
me115405, me115212, me115323    = [], [], []
m212115, m323115, m405115       = [], [], []
me212115, me323115, me405115    = [], [], []


print('------------------------------------------------------------')
print('Number of matched stars between F115, ----, ----, F405: ' 
      + str(len(match115405)))
print('Number of matched stars between F115, F212, ----, ----: '
      + str(len(match115212)))
print('Number of matched stars between F115, ----, F323, ----: '
      + str(len(match115323)))
print('------------------------------------------------------------')

# --------------------------------#

for i in match115405: 
    x115405.append(stars_table[i]['x'][0])
    y115405.append(stars_table[i]['y'][0])
    m115405.append(stars_table[i]['m'][0])
    me115405.append(stars_table[i]['me'][0])
    x405115.append(stars_table[i]['x'][3])
    y405115.append(stars_table[i]['y'][3])
    m405115.append(stars_table[i]['m'][3])
    me405115.append(stars_table[i]['me'][3])
for i in match115323:
    x115323.append(stars_table[i]['x'][0])
    y115323.append(stars_table[i]['y'][0])
    m115323.append(stars_table[i]['m'][0])
    me115323.append(stars_table[i]['me'][0])
    x323115.append(stars_table[i]['x'][2])
    y323115.append(stars_table[i]['y'][2])
    m323115.append(stars_table[i]['m'][2])
    me323115.append(stars_table[i]['me'][2])
for i in match115212:
    x115212.append(stars_table[i]['x'][0])
    y115212.append(stars_table[i]['y'][0])
    m115212.append(stars_table[i]['m'][0])
    me115212.append(stars_table[i]['me'][0])
    x212115.append(stars_table[i]['x'][1])
    y212115.append(stars_table[i]['y'][1])
    m212115.append(stars_table[i]['m'][1])
    me212115.append(stars_table[i]['me'][1])

# ---------------------------------------#

# Plotting CMDs for each filter vs. F115W - F405N

fig, axis = plt.subplots(2, 1, figsize = (20,20))

arr_diff = np.subtract(m115405, m405115)

axis[0].plot(arr_diff, m115405, 'k+')
axis[0].set_xlabel('F115W - F405N')
axis[0].set_ylabel('F115W')

axis[1].plot(arr_diff, m405115, 'k+')
axis[1].set_xlabel('F115 - F405N')
axis[1].set_ylabel('F405N')

plt.savefig('/Users/devaldeliwala/research/jwst_extinction/media/cmd/mosaic_alt_115405.png')

# --------------------------------------------------------------------------------------#

fig, axis = plt.subplots(2, 1, figsize = (20,20))

arr_diff = np.subtract(m115212, m212115)
arr_diff2 = np.subtract(m115323, m323115)

axis[0].plot(arr_diff, m212115, 'k+')
axis[0].set_xlabel('F115W - F212N')
axis[0].set_ylabel('F212N')
axis[0].invert_yaxis()

axis[1].plot(arr_diff2, m323115, 'k+')
axis[1].set_xlabel('F115 - F323N')
axis[1].set_ylabel('F323N')
axis[1].invert_yaxis()

plt.savefig('/Users/devaldeliwala/research/jwst_extinction/media/cmd/mosaic_alt_115_212323.png')

# ---------------------------------------------------------------------------------------------#

# Plotting Centroid Positions of Stars that matched in each filter combination

fig, axis = plt.subplots(3, 1, figsize = (20, 30))

axis[0].plot(x115405, y115405, 'k+', label = '1.15µm stars')
axis[0].plot(x405115, y405115, 'b+', label = '4.05µm stars')
axis[0].set_title('matched ALT centroid pos between 1.15µm and 4.05µm')
axis[0].set_xlabel('x')
axis[0].set_ylabel('y')

axis[1].plot(x115212, y115212, 'k+', label = '1.15µm stars')
axis[1].plot(x212115, y212115, 'm+', label = '2.12µm stars')
axis[1].set_title('matched ALT centroid pos between 1.15µm and 2.12µm')
axis[1].set_xlabel('x')
axis[1].set_ylabel('y')

axis[2].plot(x115323, y115323, 'k+', label = '1.15µm stars')
axis[2].plot(x323115, y323115, 'r+', label = '3.23µm stars')
axis[2].set_title('matched centroid pos between 1.15µm and 3.23 µm')
axis[2].set_xlabel('x')
axis[2].set_ylabel('y')

plt.savefig('/Users/devaldeliwala/research/jwst_extinction/media/transforms/mosaic_ALT_centroid.png')

# -----------------------------------------------------------------------------------------------#

# Plotting Gaussian Mesh CMDs 

color_mag_diagram_rcbar(m115405, m405115, m115405, 'f115w', 'f405n', 'f115w',
                        [10.35, 25.1], [9.5, 24.3], [10.35, 24.1], [9.5, 23.3], 100) 

color_mag_diagram_rcbar(m115405, m405115, m405115, 'f115w', 'f405n', 'f405n',
                        [10.35, 14.55], [9.8, 14.6], [10.35, 13.95], [9.8, 14], 100)

color_mag_diagram_rcbar(m115212, m212115, m212115, 'f115w', 'f212n', 'f212n', 
                        [10, 17.7], [9, 17.6], [10, 16.8], [9, 16.7], 100)

color_mag_diagram_rcbar(m115323, m323115, m323115, 'f115w', 'f323n', 'f323n', 
                        [10.45, 15.5], [9.85, 15.75], [10.45, 14.9], [9.85, 15.15], 100)

# -----------------------------------------------------------------------------#

# Unsharp-masking algorithm to emphasize the Red Clump Bar


unsharp_mask(m115405, m405115, m115405, me115405, me405115, me115405,
             'jwst_f115w', 'jwst_f405n', 'jwst_f115w')
unsharp_mask(m115212, m212115, m212115, me115212, me212115, me212115,
             'jwst_f115w', 'jwst_f212n', 'jwst_f212n')
unsharp_mask(m115323, m323115, m323115, me115323, me323115, me323115,
             'jwst_f115w', 'jwst_f405n', 'jwstf323n')
unsharp_mask(m115405, m405115, m405115, me115405, me405115, me405115,
             'jwst__f_115W', 'jwst__f_405N', 'jwst_f405n')

#--------------------------------------------------------#
