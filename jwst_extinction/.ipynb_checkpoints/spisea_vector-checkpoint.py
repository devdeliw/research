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
            plt.savefig(f"/Users/devaldeliwala/research/jwst_extinction/media/cmd/cmd_{x_name}_{y_name}_SPISEA.png")

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
            plt.savefig(f"/Users/devaldeliwala/research/jwst_extinction/media/cmd/cmd_{x_name}_{y_name}_SPISEA.png")

        return m1_match, m2_match


    def extinction_vector(self, logAge, AKs, AKs_step, dist, metallicity, filt_list,
                          iso_dir): 

        """
        Generates theoretical isochrone based on the age, distance,
        metallicity to a starcluster. Uses the MIST evolution model and the
        Fritz+11 extinction law. 

        Afterwards plots the increasing extinction isochrones against a CMD
        that should follow along with the slope of the RC Bar

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
                [catalog1_name (smaller wavelength), catalog2_name (bigger
                wavelength)]

        iso_dir     :   string
                directory to place generated theoretical isochrones

        """

        evo_model = evolution.MISTv1()
        atm_func  = atmospheres.get_merged_atmosphere
        red_law   = reddening.RedLawFritz11(scale_lambda=2.166)
        print(filt_list)

        AKs2 = AKs  + AKs_step
        AKs3 = AKs2 + AKs_step
        AKs4 = AKs3 + AKs_step
        AKs5 = AKs4 + AKs_step

        # Generating Isochrones
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
        """
        idx = np.where( abs(my_iso.points['mass'] - 1.0) == min(abs(my_iso.points['mass'] - 1.0)) )[0]
        filter_1 = np.round(my_iso.points[idx[0]][''+my_iso.points.keys()[9]], decimals=3)
        filter_2 = np.round(my_iso.points[idx[0]][''+my_iso.points.keys()[8]], decimals=3)

        idx2 = np.where( abs(my_iso2.points['mass'] - 1.0) == min(abs(my_iso2.points['mass'] - 1.0)) )[0]
        filter1_2 = np.round(my_iso2.points[idx2[0]][''+my_iso2.points.keys()[9]], decimals=3)
        filter2_2 = np.round(my_iso2.points[idx2[0]][''+my_iso2.points.keys()[8]], decimals=3)

        idx3 = np.where( abs(my_iso3.points['mass'] - 1.0) == min(abs(my_iso3.points['mass'] - 1.0)) )[0]
        filter1_3 = np.round(my_iso3.points[idx3[0]][''+my_iso3.points.keys()[9]], decimals=3)
        filter2_3 = np.round(my_iso3.points[idx3[0]][''+my_iso3.points.keys()[8]], decimals=3)

        idx4 = np.where( abs(my_iso4.points['mass'] - 1.0) == min(abs(my_iso4.points['mass'] - 1.0)) )[0]
        filter1_4 = np.round(my_iso4.points[idx4[0]][''+my_iso4.points.keys()[9]], decimals=3)
        filter2_4 = np.round(my_iso4.points[idx4[0]][''+my_iso4.points.keys()[8]], decimals=3)

        idx5 = np.where( abs(my_iso5.points['mass'] - 1.0) == min(abs(my_iso5.points['mass'] - 1.0)) )[0]
        filter1_5 = np.round(my_iso5.points[idx5[0]][''+my_iso5.points.keys()[9]], decimals=3)
        filter2_5 = np.round(my_iso5.points[idx5[0]][''+my_iso5.points.keys()[8]], decimals=3)

        print('1 M_sun: {0} = {1} mag, {2} = {3}\
              mag'.format(self.catalog1_name, filter_1, self.catalog2_name,
              filter_2))

        # Making Initial Mass Function (IMF)
        
        # Make multiplicity object Here, we use the MultiplicityUnresolved object,
        # based on Lu+13. This means that star systems will be unresolved, i.e.,
        # that all components of a star system are combined into a single "star" in the cluster
        imf_multi = multiplicity.MultiplicityUnresolved()

        # Make IMF object; we'll use a broken power law with the parameters from Kroupa+01

        # NOTE: when defining the power law slope for each segment of the IMF, we define
        # the entire exponent, including the negative sign. For example, if dN/dm $\propto$ m^-alpha,
        # then you would use the value "-2.3" to specify an IMF with alpha = 2.3.

        massLimits = np.array([0.2, 0.5, 1, 120]) # Define boundaries of each mass segement
        powers = np.array([-1.3, -2.3, -2.3]) # Power law slope associated with each mass segment
        my_imf = imf.IMF_broken_powerlaw(massLimits, powers, imf_multi)

        # Define total cluster mass
        mass = 10**5.

        # Make cluster object
        cluster = synthetic.ResolvedCluster(my_iso, my_imf, mass)

        # Look at the cluster CMD, compared to input isochrone. Note the impact of
        # multiple systems on the photometry
        clust = cluster.star_systems
        iso = my_iso.points

        # Plotting Synthetic Isochrones
        fig, axis = plt.subplots(2,1, figsize = (20,20))

        axis[0].plot(clust[''+my_iso.points.keys()[9]]
                     - clust[''+my_iso.points.keys()[8]], 
                     clust[''+my_iso.points.keys()[9]], 'k.', ms=5, alpha=0.1, label='__nolegend__')
        axis[0].plot(iso[''+my_iso.points.keys()[9]]
                     - iso[''+my_iso.points.keys()[8]],
                     iso[''+my_iso.points.keys()[9]],
               'r-', label='Isochrone')
        axis[0].set_xlabel(self.catalog1_name + " - " + self.catalog2_name)
        axis[0].set_ylabel(self.catalog1_name)
        axis[0].invert_yaxis()
        axis[0].legend()

        axis[1].plot(clust[''+my_iso.points.keys()[9]]
                     - clust[''+my_iso.points.keys()[8]],
                     clust[''+my_iso.points.keys()[8]], 'k.', ms=5, alpha=0.1, label='__nolegend__')
        axis[1].plot(iso[''+my_iso.points.keys()[9]]
                     - iso[''+my_iso.points.keys()[8]],
                     iso[''+my_iso.points.keys()[8]],
               'r-', label='Isochrone')
        axis[1].set_xlabel(self.catalog1_name + " - " + self.catalog2_name)
        axis[1].set_ylabel(self.catalog2_name)
        axis[1].invert_yaxis()
        axis[1].legend()

        file_name = self.catalog1_name + self.catalog2_name
        plt.savefig(f"/Users/devaldeliwala/research/jwst_extinction/media/isochrones/syn_isochrone_{file_name}.png")


        # Creating Isochrone-CMD Plot
    
        idxs1, idxs2, dr, dm, m1, m2, m1_err, m2_err = self.match()
        m1_match, m2_match = self.cmd(idxs1, idxs2, m1, m2)
        
        arr_diff = np.subtract(m1_match, m2_match)
        
        fig, axis = plt.subplots(2, 1, figsize = (20,20))

        axis[0].plot(my_iso.points[''+my_iso.points.keys()[9]]
                - my_iso.points[''+my_iso.points.keys()[8]], 
            my_iso.points[''+my_iso.points.keys()[9]], 'r-', label='_nolegend_')
        axis[0].plot(my_iso.points[''+my_iso.points.keys()[9]][idx]
                - my_iso.points[''+my_iso.points.keys()[8]][idx], 
           my_iso.points[''+my_iso.points.keys()[9]][idx], 'b*', ms=15, label='1 $M_\odot$')

        axis[0].plot(my_iso2.points[''+my_iso2.points.keys()[9]]
                - my_iso2.points[''+my_iso2.points.keys()[8]], 
            my_iso2.points[''+my_iso2.points.keys()[9]], 'r-', label='_nolegend_')
        axis[0].plot(my_iso2.points[''+my_iso2.points.keys()[9]][idx2]
                - my_iso2.points[''+my_iso2.points.keys()[8]][idx2], 
           my_iso2.points[''+my_iso2.points.keys()[9]][idx2], 'b*', ms=15, label='_nolegend_')

        axis[0].plot(my_iso3.points[''+my_iso3.points.keys()[9]]
                - my_iso3.points[''+my_iso3.points.keys()[8]], 
            my_iso3.points[''+my_iso3.points.keys()[9]], 'r-', label='_nolegend_')
        axis[0].plot(my_iso3.points[''+my_iso3.points.keys()[9]][idx3]
             - my_iso3.points[''+my_iso3.points.keys()[8]][idx3], 
           my_iso3.points[''+my_iso3.points.keys()[9]][idx3], 'b*', ms=15, label='_nolegend_')

        axis[0].plot(my_iso4.points[''+my_iso4.points.keys()[9]]
            - my_iso4.points[''+my_iso4.points.keys()[8]], 
            my_iso4.points[''+my_iso4.points.keys()[9]], 'r-', label='_nolegend_')
        axis[0].plot(my_iso4.points[''+my_iso4.points.keys()[9]][idx4]
            - my_iso4.points[''+my_iso4.points.keys()[8]][idx4], 
           my_iso4.points[''+my_iso4.points.keys()[9]][idx4], 'b*', ms=15, label='_nolegend_')

        axis[0].plot(my_iso5.points[''+my_iso5.points.keys()[9]]
                - my_iso5.points[''+my_iso5.points.keys()[8]], 
            my_iso5.points[''+my_iso5.points.keys()[9]], 'r-', label='_nolegend_')
        axis[0].plot(my_iso5.points[''+my_iso5.points.keys()[9]][idx5]
                - my_iso5.points[''+my_iso5.points.keys()[8]][idx5], 
           my_iso5.points[''+my_iso5.points.keys()[9]][idx5], 'b*', ms=15, label='_nolegend_')

        nbins=600
        k = kde.gaussian_kde([arr_diff, m1_match])
        xi, yi = np.mgrid[min(arr_diff):max(arr_diff):nbins*1j,
                              min(m1_match):max(m1_match):nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        axis[0].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
        axis[0].invert_yaxis()

        #----------------------------------------------------#

        axis[1].plot(my_iso.points[''+my_iso.points.keys()[9]]
                - my_iso.points[''+my_iso.points.keys()[8]],
            my_iso.points[''+my_iso.points.keys()[8]], 'r-', label='_nolegend_')
        axis[1].plot(my_iso.points[''+my_iso.points.keys()[9]][idx]
                - my_iso.points[''+my_iso.points.keys()[8]][idx],
           my_iso.points[''+my_iso.points.keys()[8]][idx], 'b*', ms=15, label='1 $M_\odot$')

        axis[1].plot(my_iso2.points[''+my_iso2.points.keys()[9]]
                - my_iso2.points[''+my_iso2.points.keys()[8]],
            my_iso2.points[''+my_iso2.points.keys()[8]], 'r-', label='_nolegend_')
        axis[1].plot(my_iso2.points[''+my_iso2.points.keys()[9]][idx2]
                - my_iso2.points[''+my_iso2.points.keys()[8]][idx2],
           my_iso2.points[''+my_iso2.points.keys()[8]][idx2], 'b*', ms=15, label='_nolegend_')

        axis[1].plot(my_iso3.points[''+my_iso3.points.keys()[9]]
                - my_iso3.points[''+my_iso3.points.keys()[8]],
            my_iso3.points[''+my_iso3.points.keys()[8]], 'r-', label='_nolegend_')
        axis[1].plot(my_iso3.points[''+my_iso3.points.keys()[9]][idx3]
             - my_iso3.points[''+my_iso3.points.keys()[8]][idx3],
           my_iso3.points[''+my_iso3.points.keys()[8]][idx3], 'b*', ms=15, label='_nolegend_')

        axis[1].plot(my_iso4.points[''+my_iso4.points.keys()[9]]
            - my_iso4.points[''+my_iso4.points.keys()[8]],
            my_iso4.points[''+my_iso4.points.keys()[8]], 'r-', label='_nolegend_')
        axis[1].plot(my_iso4.points[''+my_iso4.points.keys()[9]][idx4]
            - my_iso4.points[''+my_iso4.points.keys()[8]][idx4],
           my_iso4.points[''+my_iso4.points.keys()[8]][idx4], 'b*', ms=15, label='_nolegend_')

        axis[1].plot(my_iso5.points[''+my_iso5.points.keys()[9]]
                - my_iso5.points[''+my_iso5.points.keys()[8]],
            my_iso5.points[''+my_iso5.points.keys()[8]], 'r-', label='_nolegend_')
        axis[1].plot(my_iso5.points[''+my_iso5.points.keys()[9]][idx5]
                - my_iso5.points[''+my_iso5.points.keys()[8]][idx5],
           my_iso5.points[''+my_iso5.points.keys()[8]][idx5], 'b*', ms=15, label='_nolegend_')
            
        k = kde.gaussian_kde([arr_diff, m2_match])
        xi, yi = np.mgrid[min(arr_diff):max(arr_diff):nbins*1j,
                              min(m2_match):max(m2_match):nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    
        axis[1].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
        axis[1].invert_yaxis()
        
        #--------------------------------------------------------------#
    
        file_name = self.catalog1_name + self.catalog2_name
        plt.savefig(f"/Users/devaldeliwala/research/jwst_extinction/media/isochrones/isochrone_cmd_{file_name}.png")
        return 
"""


df3 = pd.read_csv("catalogs/catalog323n.csv", delimiter = ",")
df4 = pd.read_csv("catalogs/catalog444w.csv", delimiter = ",")


spisea2 = SPISEA_CMD(df3, df4, "jwst_323n", "jwst_405n", dr_tol = 15, dm_tol = 15,
                   y_axis_m1 = True)

spisea2.extinction_vector(np.log(10**9), 2, 1.5, 8000, -0.3, ['j', 'jwst,F212N'],
                    "/Users/devaldeliwala/research/jwst_extinction/media/isochrones")



