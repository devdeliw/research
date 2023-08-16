import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np
from flystar import starlists, transforms, startables, align
from collections import Counter
from scipy.spatial import cKDTree as KDT
from astropy.table import Column, Table
import matplotlib.gridspec as gridspec
import itertools
import copy
import scipy.signal
from scipy.spatial import distance
import math
from flystar import match
import pdb
import astropy.coordinates as coord
import astropy.units as u
from astropy.io import ascii
from astropy.coordinates import SkyCoord
import sys
from scipy.stats import gaussian_kde, kde
from spisea import synthetic, evolution, atmospheres, reddening, ifmr
from spisea.imf import imf, multiplicity
import pdb
import pylab as py
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import mpl_scatter_density

class SPISEA_CMD:
    def __init__(self, catalog1, catalog2, catalog1_name, catalog2_name,
                 dr_tol, dm_tol, y_axis_m1):

        """
        Finds star matches between two different catalogs using flystar.match. No
        transformations are done as it is assumed both catalogs lie on the same
        coordinate plane and magnitude system.

        As a result, using catalogs in similar wavelength ranges will yield more
        accurate matches.

        Afterwards builds a color-magnitude diagram between the two catalogs

        Parameters:
        -----------
        catalog1    : Pandas DataFrame
                catalog of first starlist

        catalog2    : Pandas DataFrame
                catalog of second starlist

        catalog1_name: String
                    name of first catalog

        catalog2_name: String
                    name of second catalog

        dr_tol      : float
                how close stars have to be to match

        dm_tol      : float
                how close in delta-magnitude stars have to be to match

        y_axis_m1   : boolean
                chooses whether y_axis on the CMD is m1, or m2

        idxs1       : Numpy Array
                list of indices in first catalog that matched

        idxs2       : Numpy Array
                list of indices in second catalog that matched

        """

        self.catalog1 = catalog1
        self.catalog2 = catalog2
        self.catalog1_name = catalog1_name
        self.catalog2_name = catalog2_name
        self.dr_tol = dr_tol
        self.dm_tol = dm_tol
        self.y_axis_m1 = y_axis_m1

    def match(self):

        df, df2 = self.catalog1, self.catalog2

        x1 = df['x']
        y1 = df['y']
        m1 = df['m']
        m1e = df['me']
        x2 = df2['x']
        y2 = df2['y']
        m2 = df2['m']
        m2e = df2['me']

        idxs1, idxs2, dr, dm = match.match(x1, y1, m1, x2, y2, m2, self.dr_tol,
                                           self.dm_tol, verbose = False)

        m1_err, m2_err = [], []

        for i in idxs1:
            m1_err += [m1e[i]]
        for i in idxs2:
            m2_err += [m2e[i]]

        m1_err = np.array(m1_err)
        m2_err = np.array(m2_err)

        return idxs1, idxs2, dr, dm, m1, m2, m1_err, m2_err

    def cmd(self, idxs1, idxs2, m1, m2):

        """
        Generates color-magnitude diagram (CMD) between two catalogs

        m1 - m2 v. m2 if y_axis_m1 = False
        m1 - m2 v. m1 if y_axis_m1 = True

        Parameters:
        -----------

        y_axis_m1: boolean
            True if CMD has m1 on y_axis, False if CMD has m2 on y_axis

        catalog1_name: string
            name of catalog1

        catalog2_name: string
            name of catalog2

        """

        m1_match, m2_match = [], []

        for i in idxs1:
            m1_match += [m1[i]]
        for i in idxs2:
            m2_match += [m2[i]]

        arr_diff = np.subtract(m1_match, m2_match)

        fig, axis = plt.subplots(2, 1, figsize = (20,20))

        if self.y_axis_m1 == False:
            xy = np.vstack([arr_diff, m2_match])
            z = gaussian_kde(xy)(xy)

            # Make the Plot
            axis[0].set_xlabel(self.catalog1_name + " - " + self.catalog2_name)
            axis[0].set_ylabel(self.catalog2_name)

            axis[0].scatter(arr_diff, m2_match, c = z, s = 1)
            axis[0].invert_yaxis()

            nbins=600
            k = kde.gaussian_kde([arr_diff, m2_match])
            xi, yi = np.mgrid[min(arr_diff):max(arr_diff):nbins*1j,
                              min(m2_match):max(m2_match):nbins*1j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))

            axis[1].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
            axis[1].set_xlabel(self.catalog1_name + " - " + self.catalog2_name)
            axis[1].set_ylabel(self.catalog2_name)
            axis[1].invert_yaxis()

            x_name = self.catalog1_name + self.catalog2_name
            y_name = self.catalog2_name
            plt.savefig(f"/Users/devaldeliwala/research/jwst_extinction/img/cmd/cmd_{x_name}_{y_name}_SPISEA.png")

        else:
            xy = np.vstack([arr_diff, m1_match])
            z = gaussian_kde(xy)(xy)

            axis[0].set_xlabel(self.catalog1_name + " - " + self.catalog2_name)
            axis[0].set_ylabel(self.catalog1_name)

            axis[0].scatter(arr_diff, m1_match, c = z, s = 1)
            axis[0].invert_yaxis()

            nbins=600
            k = kde.gaussian_kde([arr_diff, m1_match])
            xi, yi = np.mgrid[min(arr_diff):max(arr_diff):nbins*1j,
                              min(m1_match):max(m1_match):nbins*1j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))

            axis[1].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
            axis[1].set_xlabel(self.catalog1_name + " - " + self.catalog2_name)
            axis[1].set_ylabel(self.catalog1_name)
            axis[1].invert_yaxis()

            x_name = self.catalog1_name + self.catalog2_name
            y_name = self.catalog1_name
            plt.savefig(f"/Users/devaldeliwala/research/jwst_extinction/img/cmd/cmd_{x_name}_{y_name}_SPISEA.png")

        return m1_match, m2_match


    def theoretical_iso(self, logAge, AKs, dist, metallicity, filt_list,
                          iso_dir): 

        """
        Generates theoretical isochrone based on the age, distance,
        metallicity to a starcluster. Uses the MIST evolution model and the
        Fritz+11 extinction law. 

        Parameters:
        -----------

        logAge      :   float
                age of starcluster in log years: np.log10(10**9) -- 1 billion year old cluster

        Aks         :   float 
                total extinction in Ks (2.2µm band)

        distance    :   float   
                distance to starcluster in parsecs

        metallicity :   float
                metallicity of starcluster

        filt_list   :   list
                list of starclusters filters used -- see SPISEA documentation

        iso_dir     :   string
                directory to place generated theoretical isochrones

        """

        evo_model = evolution.MISTv1()
        atm_func  = atmospheres.get_merged_atmosphere
        red_law   = reddening.RedLawFritz11(scale_lambda=2.166)

        AKs2 = AKs  + 0.3
        AKs3 = AKs2 + 0.3
        AKs4 = AKs3 + 0.3
        AKs5 = AKs4 + 0.3

        my_iso = synthetic.IsochronePhot(logAge, AKs, dist, metallicity,
                                    evo_model=evo_model, atm_func=atm_func,
                                    red_law=red_law, filters=filt_list,
                                    iso_dir=iso_dir)

        my_iso2 = synthetic.IsochronePhot(logAge, AKs2, dist, metallicity,
                                    evo_model=evo_model, atm_func=atm_func,
                                    red_law=red_law, filters=filt_list,
                                    iso_dir=iso_dir)

        my_iso3 = synthetic.IsochronePhot(logAge, AKs3, dist, metallicity,
                                    evo_model=evo_model, atm_func=atm_func,
                                    red_law=red_law, filters=filt_list,
                                    iso_dir=iso_dir)

        my_iso4 = synthetic.IsochronePhot(logAge, AKs4, dist, metallicity,
                                  evo_model=evo_model, atm_func=atm_func,
                                    red_law=red_law, filters=filt_list,
                                    iso_dir=iso_dir)

        my_iso5 = synthetic.IsochronePhot(logAge, AKs5, dist, metallicity,
                                    evo_model=evo_model, atm_func=atm_func,
                                    red_law=red_law, filters=filt_list,
                                    iso_dir=iso_dir)


        file_name = filt_list[0] + filt_list[1]
        dfy = pd.DataFrame(my_iso.points['phase'])
        dfy.to_csv(f"spisea_iso{file_name}.csv")

        print('The columns in the isochrone table are: {0}'.format(my_iso.points.keys()))

        return


df = pd.read_csv("catalog115w.csv", delimiter = ",")
df2 = pd.read_csv("catalog212n.csv", delimiter = ",")

#---------------------------------------------------#

cmd = SPISEA_CMD(df, df2, "jwst_115w", "jwst_212n", dr_tol = 15, dm_tol = 15,
                   y_axis_m1 = True)

cmd.theoretical_iso(np.log(10**9), 2, 8000, -0.3, ['jwst,F212N', 'jwst,F115W'],
                    "/Users/devaldeliwala/research/jwst_extinction/img/isochrones")
