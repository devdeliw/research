import os 
import numpy as np 
import pandas as pd 

from astropy.table import Table
from catalog_helper_functions import get_matches
from scipy.stats import gaussian_kde 
from scipy.optimize import curve_fit 

from slope import RidgeTracing 
from dbscan import LocateRC 


class Render(RidgeTracing, LocateRC): 
    """
    Generates a color-magnitude diagram, locates the Red Clump (RC) 
    cluster via density-based clustering, then runs a gradient ascent 
    algorithm to calculate the slope of the RC cluster.

    The color-magnitude is defined as 
    `mag1` - `mag2` vs. `magy`. 

    Args: 
        mag1 (array-like): magnitudes for first wavelength catalog.
        mag2 (array-like): magnitudes for second wavelength catalog.
        magy (array-like): magnitudes for wavelength catalog on y axis.
        mag1name, mag2name, magyname (str): names for respective catalogs. 
    
    """

    def __init__(
        self, 
        mag1, mag2, magy, 
        mag1name, mag2name, magyname, 
        region
    ): 
        self.mag1 = mag1 
        self.mag2 = mag2 
        self.magy = magy 
        self.mag1name = mag1name 
        self.mag2name = mag2name 
        self.magyname = magyname 
        self.region = region
        return 

    def dbscan_ansatz(self, num_ansatz=20, eps=0.1, min_samples=50):
        """
        Generates starting ansatz coordinates within the located 
        RC cluster for the gradient ascent algorithm. 

        This is accomplished via generating `num_ansatz` vertical tiles 
        across the RC bar in color-space and choosing a random point 
        in each tile. 

        Args: 
            num_ansatz (int): number of ansatz starting points for slope calc. 
            eps (float): min distance for nearby point in same cluster.
            min_samples (int): min number points to form cluster.

        """ 

        rc_color, rc_mag = LocateRC(
            self.mag1, self.mag2, self.magy,
            self.mag1name, self.mag2name, self.magyname,
            self.region, 
        ).isolate(eps=eps, min_samples=min_samples, render_rc=True) 

        color_range = max(rc_color) - min(rc_color) 

        # Vertical bin boundaries 
        bin_width = color_range / num_ansatz 
        start = min(rc_color) 
        bins = []
        for _ in range(num_ansatz): 
            bins.append([start, start+bin_width])
            start = start+bin_width
        
        ansatz_points = []
        for bin in bins: 
            # indices of stars within bin boundaries
            interior_idxs = (rc_color > bin[0]) & (rc_color < bin[1])

            bin_color = np.array(rc_color)[interior_idxs] 
            bin_mag = np.array(rc_mag)[interior_idxs] 

            # select random coordinate
            idx = np.random.randint(0, len(bin_color))
            ansatz_points.append([bin_color[idx], bin_mag[idx]])

        self.rc_x = rc_color 
        self.rc_y = rc_mag 

        return ansatz_points 

    def gradient_ascent(
        self, 
        num_ansatz=40,
        eps=0.1, 
        min_samples=50,
    ): 

        ansatz_points = self.dbscan_ansatz(num_ansatz, eps, min_samples)

        # calculate rc cluster bounding box
        rc_bbox = pd.DataFrame(
            {
                'color': [min(self.rc_x), max(self.rc_x)], 
                'mag': [min(self.rc_y), max(self.rc_y)]
            }
        )

        # calculate slope, intercept of rc cluster via gradient ascent
        slope, intercept = RidgeTracing(
            self.mag1, self.mag2, self.magy,
            self.mag1name, self.mag2name, self.magyname, 
            self.region, rc_bbox,  
            N_color_bins=100, N_mag_bins=100, 
        ).slope(ansatz=ansatz_points)

        return slope, intercept 
















if __name__ == '__main__': 
    catalog_dir = './fits/'
    file_name = 'jwst_init_NRCB.fits'

    catalog = Table.read(
    os.path.join(catalog_dir, file_name), 
        format='fits'
    )

    mf115w, mf212n, mef115w, mef212n = get_matches(
        catalog, 
        'F115W', 'NRCB1', 
        'F212N', 'NRCB1', 
    )

    slope, intercept = Render(
            mf115w, mf212n, mf115w,
            'F115W', 'F212N', 'F115W', 
    ).gradient_ascent(eps=0.1, min_samples=50) 



















    



