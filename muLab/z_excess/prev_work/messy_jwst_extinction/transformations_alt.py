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

print(trans_list)
print(stars_table)



#-------------------------#

match115405     = []
match115212     = []
match115323     = []
match212323     = []
match_all       = [] 
match_115212323 = []

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
            math.isnan(stars_table[i]['x'][1]) != True and
            math.isnan(stars_table[i]['x'][2]) != True
        ):
        match212323.append(i)

for i in range(len(stars_table)):
    if  (
            math.isnan(stars_table[i]['x'][0]) != True and
            math.isnan(stars_table[i]['x'][2]) != True
        ):
        match115323.append(i)

for i in range(len(stars_table)):
    if  (
            math.isnan(stars_table[i]['x'][0]) != True and
            math.isnan(stars_table[i]['x'][1]) != True and
            math.isnan(stars_table[i]['x'][2]) != True and
            math.isnan(stars_table[i]['x'][3]) != True
        ):
        match_all.append(i)

for i in range(len(stars_table)):
    if  (
            math.isnan(stars_table[i]['x'][0]) != True and
            math.isnan(stars_table[i]['x'][1]) != True and
            math.isnan(stars_table[i]['x'][2]) != True
        ):
        match_115212323.append(i)



x115405, x115323, x115212       = [], [], []
x212115, x323115, x405115       = [], [], []
y115405, y115323, y115212       = [], [], []
y212115, y323115, y405115       = [], [], []
x212323, x323212                = [], []  
y212323, y323212                = [], []

m212323, m323212                = [], []
me212323, me323212              = [], []
m115405, m115212, m115323       = [], [], []
me115405, me115212, me115323    = [], [], []
m212115, m323115, m405115       = [], [], []
me212115, me323115, me405115    = [], [], []

m115_all, m212_all, m323_all, m405_all      = [], [], [], []
me115_all, me212_all, me323_all, me405_all  = [], [], [], []
m115_rest, m212_rest, m323_rest             = [], [], []
me115_rest, me212_rest, me323_rest          = [], [], []


print('------------------------------------------------------------')
print('Number of matched stars between F115, ----, ----, F405: ' 
      + str(len(match115405)))
print('Number of matched stars between F115, F212, ----, ----: '
      + str(len(match115212)))
print('Number of matched stars between F115, ----, F323, ----: '
      + str(len(match115323)))
print('Number of matched stars between ----, F212, F323, ----: '
      + str(len(match212323)))
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
for i in match212323:
    x212323.append(stars_table[i]['x'][1])
    y212323.append(stars_table[i]['y'][1])
    m212323.append(stars_table[i]['m'][1])
    me212323.append(stars_table[i]['me'][1])
    x323212.append(stars_table[i]['x'][2])
    y323212.append(stars_table[i]['y'][2])
    m323212.append(stars_table[i]['m'][2])
    me323212.append(stars_table[i]['me'][2])
for i in match_all:
    m115_all.append(stars_table[i]['m'][0])
    m212_all.append(stars_table[i]['m'][1])
    m323_all.append(stars_table[i]['m'][2])
    m405_all.append(stars_table[i]['m'][3])
    me115_all.append(stars_table[i]['me'][0])
    me212_all.append(stars_table[i]['me'][1])
    me323_all.append(stars_table[i]['me'][2])
    me405_all.append(stars_table[i]['me'][3])
for i in match_115212323:
    m115_rest.append(stars_table[i]['m'][0])
    m212_rest.append(stars_table[i]['m'][1])
    m323_rest.append(stars_table[i]['m'][2])
    me115_rest.append(stars_table[i]['me'][0])
    me212_rest.append(stars_table[i]['me'][1])
    me323_rest.append(stars_table[i]['me'][2])

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
arr_diff3 = np.subtract(m212323, m323212)

axis[0].plot(arr_diff3, m212323, 'k+')
axis[0].set_xlabel('F212W - F323N')
axis[0].set_ylabel('F212N')
axis[0].invert_yaxis()

axis[1].plot(arr_diff2, m323115, 'k+')
axis[1].set_xlabel('F115 - F323N')
axis[1].set_ylabel('F323N')
axis[1].invert_yaxis()

plt.savefig('/Users/devaldeliwala/research/jwst_extinction/media/cmd/mosaic_alt_212_323.png')

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

color_mag_diagram_rcbar(m115212, m212115, m115212, 'f115w', 'f212n', 'f115w',
                        [10, 27.55], [9.0, 26.55], [9.8, 26.3], [8.8, 25.3], 100) 

color_mag_diagram_rcbar(m115405, m405115, m405115, 'f115w', 'f405n', 'f405n',
                        [10.35, 14.55], [9.8, 14.6], [10.35, 13.95], [9.8, 14], 100)

color_mag_diagram_rcbar(m212323, m323212, m212323, 'f212n', 'f323n', 'f212n', 
                        [1.55, 17.9], [0.9, 17.4], [1.55, 17.1], [0.9, 16.6], 100)

color_mag_diagram_rcbar(m115323, m323115, m323115, 'f115w', 'f323n', 'f323n', 
                        [10.4, 15.1], [9.8, 15.45], [10.4, 14.35], [9.8, 14.7], 100)

# -----------------------------------------------------------------------------#

# Unsharp-masking algorithm to emphasize the Red Clump Bar


unsharp_mask(m115212, m212115, m115212, me115212, me212115, me115212,
             'jwst_f115w', 'jwst_f212n', 'jwst_f115w')
unsharp_mask(m212323, m323212, m212323, me212323, me323212, me212323,
             'jwst_f212n', 'jwst_f323n', 'jwst_f212n')
unsharp_mask(m115323, m323115, m323115, me115323, me323115, me323115,
             'jwst_f115w', 'jwst_f405n', 'jwstf323n')
unsharp_mask(m115405, m405115, m405115, me115405, me405115, me405115,
             'jwst__f_115W', 'jwst__f_405N', 'jwst_f405n')

#--------------------------------------------------------#

# Generating color-color diagrams 

color_color_diagram(m115_all, m405_all, m212_all, m323_all, 'jwst_f115w',
                    'jwst_f405n', 'jwst_f212n', 'jwst_f323n', [9.3, 0.6],
                    [10.3, 1.4], [9.3, 1.7], [10.3, 2.5], 100)

color_color_diagram(m115_rest, m212_rest, m212_rest, m323_rest, 'jwst_f115w',
                    'jwst_f212n', 'jwst_f212n', 'jwst_f323n', [7.6, 0.2],
                    [6.7, 0.9], [7.6, 1.2], [6.7, 1.9], 100)

color_color_diagram(m115_rest, m323_rest, m212_rest, m323_rest, 'jwst_f115w',
                    'jwst_f323n', 'jwst_f212n', 'jwst_f323n', [9, 1.2], 
                    [8, 0.5], [9, 2.2], [8, 1.5], 100)

