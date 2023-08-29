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

# Fritz 2011

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
                                init_guess_mode='name', verbose=False)
msc.fit()
trans_list = msc.trans_args
stars_table = msc.ref_table

#-------------------------#

match115405     = []
match115405_212 = []
match115405_323 = []

for i in range(len(stars_table)): 
    if  (
            math.isnan(stars_table[i]['x'][0]) != True and 
            math.isnan(stars_table[i]['x'][3]) != True
        ): 
        match115405.append(i)

for i in range(len(stars_table)):
    if  (
            math.isnan(stars_table[i]['x'][0]) != True and 
            math.isnan(stars_table[i]['x'][3]) != True and 
            math.isnan(stars_table[i]['x'][1]) != True
        ): 
        match115405_212.append(i)

for i in range(len(stars_table)):
    if  (
            math.isnan(stars_table[i]['x'][0]) != True and
            math.isnan(stars_table[i]['x'][3]) != True and
            math.isnan(stars_table[i]['x'][2]) != True
        ):
        match115405_323.append(i)

x115, y115, m115405, me115405 = [], [], [] , []
m115405212, me115405212 = [], []
m115405323, me115405323 = [], []

x212, y212, m212, me212 = [], [], [], []
x323, y323, m323, me323 = [], [], [], []

x405, y405, m405115, me405115 = [], [], [], []
m405115212, me405115212 = [], []
m405115323, me405115323 = [], []

print('Number of matched stars between F115, ----, ----, F405: ' 
      + str(len(match115405)))
print('Number of matched stars between F115, F212, ----, F405: '
      + str(len(match115405_212)))
print('Number of matched stars between F115, ----, F323, F405: '
      + str(len(match115405_323)))

#--------------------------------#

for i in match115405: 
    x115.append(stars_table[i]['x'][0])
    y115.append(stars_table[i]['y'][0])
    m115405.append(stars_table[i]['m'][0])
    me115405.append(stars_table[i]['me'][0])
    x405.append(stars_table[i]['x'][3])
    y405.append(stars_table[i]['y'][3])
    m405115.append(stars_table[i]['m'][3])
    me405115.append(stars_table[i]['me'][3])
for i in match115405_212: 
    x115.append(stars_table[i]['x'][0])
    y115.append(stars_table[i]['y'][0])
    m115405212.append(stars_table[i]['m'][0])
    me115405212.append(stars_table[i]['me'][0])
    x405.append(stars_table[i]['x'][3])
    y405.append(stars_table[i]['y'][3])
    m405115212.append(stars_table[i]['m'][3])
    me405115212.append(stars_table[i]['me'][3])
    x212.append(stars_table[i]['x'][1])
    y212.append(stars_table[i]['y'][1])
    m212.append(stars_table[i]['m'][1])
    me212.append(stars_table[i]['me'][1])
for i in match115405: 
    x115.append(stars_table[i]['x'][0])
    y115.append(stars_table[i]['y'][0])
    m115405323.append(stars_table[i]['m'][0])
    me115405323.append(stars_table[i]['me'][0])
    x405.append(stars_table[i]['x'][3])
    y405.append(stars_table[i]['y'][3])
    m405115323.append(stars_table[i]['m'][3])
    me405115323.append(stars_table[i]['me'][3])
    x323.append(stars_table[i]['x'][2])
    y323.append(stars_table[i]['y'][2])
    m323.append(stars_table[i]['m'][2])
    me323.append(stars_table[i]['me'][2])

#---------------------------------------#

# Plotting CMDs for each filter vs. F115W - F405N

fig, axis = plt.subplots(2, 1, figsize = (20,20))

arr_diff = np.subtract(m115405, m405115)

axis[0].plot(arr_diff, m115405, 'k+')
axis[0].set_xlabel('F115W - F405N')
axis[0].set_ylabel('F115W')

axis[1].plot(arr_diff, m405115, 'k+')
axis[1].set_xlabel('F115 - F405N')
axis[1].set_ylabel('F405N')

plt.savefig('/Users/devaldeliwala/research/jwst_extinction/media/cmd/mosaic_115405.png')

#--------------------------------------------------------------------------------------#

fig, axis = plt.subplots(2, 1, figsize = (20,20))

arr_diff = np.subtract(m115405212, m405115212)
arr_diff2 = np.subtract(m115405323, m405115323)

axis[0].plot(arr_diff, m212, 'k+')
axis[0].set_xlabel('F115W - F405N')
axis[0].set_ylabel('F212N')

axis[1].plot(arr_diff2, m323, 'k+')
axis[1].set_xlabel('F115 - F405N')
axis[1].set_ylabel('F323N')

plt.savefig('/Users/devaldeliwala/research/jwst_extinction/media/cmd/mosaic_115405_212323.png')


