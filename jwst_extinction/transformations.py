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

df = Table.read('catalog115j.csv')
df2 = Table.read('catalog212j.csv')
df3 = Table.read('catalog323j.csv')
df4 = Table.read('catalog444j.csv')

msc = align.MosaicToRef(df, [df2,df3,df4], iters=1,
                        dr_tol=[7], dm_tol=[7],
                                outlier_tol=[None], mag_lim= None,
                                trans_class=transforms.PolyTransform,
                                trans_args=[{'order': 1}],
                                use_vel=False,
                                use_ref_new=False,
                                update_ref_orig=False,
                                mag_trans=False,
                                weights='both,std',
                                init_guess_mode='name', verbose=False)
msc.fit()
trans_list = msc.trans_args
stars_table = msc.ref_table

print(stars_table)
