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

x115, y115, m115, me115 = [], [], [], []
x212, y212, m212, me212 = [], [], [], []
x323, y323, m323, me323 = [], [], [], []
x405, y405, m405, me405 = [], [], [], []

print('Number of matched stars between F115, ----, ----, F405: ' 
      + str(len(match115405)))
print('Number of matched stars between F115, F212, ----, F405: '
      + str(len(match115405_212)))
print('Number of matched stars between F115, ----, F323, F405: '
      + str(len(match115405_323)))

for i in match115405: 
    x115.append(stars_table[i]['x'][0])

print(x115)


