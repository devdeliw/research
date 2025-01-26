import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from dbscan import LocateRC
from astropy.table import Table 
from scipy.stats import gaussian_kde 
from catalog_helper_functions import get_matches 



class Lift(LocateRC): 
    """
    Locates the Red Clump (RC) cluster from a CMD with a 
    well-defined RC bar, and "lifts" the same stars to higher wavelengths.
    (i.e. finds the same stars in different wavelength catalogs). 

    This is useful because gradient ascent performs poorly on ill-defined 
    RC stars when DBSCAN does not work well. This way, DBSCAN can be skipped
    on harder-to-define-RC CMD's and gradient ascent can be implemented directly. 

    For large catalogs this is expensive. 
    I try to vectorize operations when possible. 

    Args: 
        mag1 (array-like): magnitudes for first wavelength catalog.
        mag2 (array-like): magnitudes for second wavelength catalog.
        magy (array-like): magnitudes for wavelength catalog on y axis.
        mag1name, mag2name, magyname (str): names for respective catalogs.
        catalog (astropy Table): starcluster catalog with every wavelength. 
        region (str): region name for image directory.
        image_dir (str): base directory to place plots.

    """

    def __init__(
        self, 
        mag1, mag2, magy, 
        mag1name, mag2name, magyname, 
        catalog, region, image_dir='./images/LIFT/', 
    ): 
        self.mag1 = mag1
        self.mag2 = mag2 
        self.magy = magy 
        self.mag1name = mag1name 
        self.mag2name = mag2name 
        self.magyname = magyname 
        self.catalog = catalog
        self.image_dir = f'{image_dir}{region}/{mag1name}-{mag2name}/'
        self.region = region
        return 

    def locate_rc(self, render_rc=False): 
        rc_color, rc_mag = self.isolate(render_rc=render_rc)

        # Finding indices of magnitude list that correspond to RC 
        x_idxs = np.where(np.in1d(
            np.subtract(self.mag1, self.mag2), 
            rc_color
        ))[0]

        y_idxs = np.where(np.in1d(
            self.magy, 
            rc_mag, 
        ))[0] 

        # Final RC star indices 
        self.idxs = np.intersect1d(x_idxs, y_idxs) 
        return 
    
    def lift(self, F115W=True, F212N=True, F323N=True, F405N=True):
        # Gets magnitude lists for RC stars in other wavelength catalogs

        # Calculate `mag1`-`mag2` vs. `magy` RC idxs
        self.locate_rc() 

        # Calculate indices in main catalog that contain the RC indices 
        full_idxs = np.where(
            np.isin(
                self.catalog['m'], 
                self.mag1[self.idxs]
            )
        )[0]
        rc_catalog = self.catalog[full_idxs]['m']

        f115w_idx = 0 
        if self.region == 'NRCB2': 
            f115w_idx = 1
        if self.region == 'NRCB3': 
            f115w_idx = 2
        if self.region == 'NRCB4': 
            f115w_idx = 3

        # Extract other wavelength's identical RC stars
        red_clump_data = {}
        if F115W: 
            red_clump_data['F115W'] = rc_catalog[:, f115w_idx] 
        if F212N:
            red_clump_data['F212N'] = rc_catalog[:, f115w_idx+4]
        if F323N: 
            red_clump_data['F323N'] = rc_catalog[:, 9]
        if F405N: 
            red_clump_data['F405N'] = rc_catalog[:, 8]

        return pd.DataFrame(red_clump_data) 








if __name__ == '__main__': 
    catalog_dir = './fits/'
    file_name = 'jwst_init_NRCB.fits'

    catalog = Table.read(
        os.path.join(catalog_dir, file_name),
        format='fits',
    )

    mf115w, mf212n, mef115w, mef212n = get_matches(
        catalog, 
        'F115W', 'NRCB1', 
        'F212N', 'NRCB1', 
    )

    data = Lift(
        mf115w, mf212n, mf115w,
        'F115W', 'F212N', 'F115W', 
        region='NRCB1', catalog=catalog
    ).lift()
    



