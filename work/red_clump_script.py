import pandas as pd
import numpy as np
from astropy.table import Table

zeropoints = {
    'Region': ['NRCB1', 'NRCB2', 'NRCB3', 'NRCB4', 'NRCB5'],
    'F115W': [25.92, 25.95, 25.97, 25.98, 0],
    'F212N': [22.12, 22.15, 22.15, 22.23, 0], 
    'F323N': [0, 0, 0, 0, 21.14], 
    'F405N': [0, 0, 0, 0, 20.9], 
}

spisea_filters = {
    'Filter': ['F115W', 'F212N', 'F323N', 'F405N'], 
    'SPISEA': ['jwst,F115W', 'jwst,F212N', 'jwst,F323N', 'jwst,F405N']
}

f115w_x1_f212n_x2 = { 
    'F115W' : [(6.3, 21.3), (9, 25.2), (6.3, 22), (9, 25.9)],
    'F212N' : [(6.2, 15.5), (7.9, 16.3), (6.65, 15.2), (8.35, 16.0)] 
}

f115w_x1_f323n_x2 = {
    'F115W' : [(6.9, 21.7), (8.5, 23.9), (7.5, 21.1), (9.1, 23.3)],
    'F323N' : [(7.4, 13.75), (9.2, 14.8), (7.1, 14.8), (8.9, 15.85)]
}

f115w_x1_f405n_x2 = { 
    'F115W' : [(7.85, 21), (10, 23.5), (8, 21.95), (10.15, 24.45)],
    'F405N' : [(7.9, 13.3), (10.5, 13.8), (8.4, 14.1), (11, 14.6)]
}

f212n_x1_f323n_x2 = {
    'F212N' : 0,
    'F323N' : 0
}

f212n_x1_f405n_x2 = {
    'F212N' : [(1.6, 14.9), (2.15, 15.85), (1.55, 15.5), (2.1, 16.45)],
    'F405N' : [(1.5, 13.15), (2.2, 13.55), (1.6, 14.1), (2.3, 14.5)]
}

f323n_x1_f405n_x2 = { 
    'F323N' : [(0.5, 13.7), (2, 15.75), (0.5, 14.8), (2, 16.85)],
    'F405N' : [(0.5, 14.3), (2.6, 15), (0.7, 13), (2.8, 13.7)]
}

f115w_x1 = { 
    'F212N' : f115w_x1_f212n_x2,
    'F323N' : f115w_x1_f323n_x2,
    'F405N' : f115w_x1_f405n_x2
}

f212n_x1 = { 
    'F323N' : f212n_x1_f323n_x2,
    'F405N' : f212n_x1_f405n_x2
}

f323n_x1 = { 
    'F405N' : f323n_x1_f405n_x2
}

x1 = pd.DataFrame({ 
    'F115W' : f115w_x1,
    'F212N' : f212n_x1,
    'F323N' : f323n_x1
})

zeropoints = pd.DataFrame(zeropoints)
spisea_filters = pd.DataFrame(spisea_filters)

fits ='~/research/work/catalogs/dr2/jwst_init_NRCB.fits'
catalog = Table.read(fits, format='fits')









