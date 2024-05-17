import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import math
import mpl_scatter_density
import pickle 
import pdb
import flystar

from scipy.spatial import cKDTree as KDT
from matplotlib.patches import Rectangle
from numpy.polynomial.polynomial import polyfit
from scipy.stats import norm, tstd
from astropy.table import * 
from astropy.modeling import models, fitting
from flystar import match, transforms, plots, align, starlists
from flystar.starlists import StarList
from flystar.startables import StarTable
from scipy.stats import gaussian_kde, kde
from spisea import synthetic, evolution, atmospheres, reddening, ifmr
from spisea.imf import imf, multiplicity


class color_magnitude_diagram: 
    def __init__(self, catalog1, catalog2, catalogy, 
                catalog1name, catalog2name, catalogyname, 
                image_path
    ): 

        """
        
        Generates a color-magnitude diagram (CMD) 
        catalog1 (mag) - catalog2 (mag) vs. catalogy (mag)

        Note catalogy must either equal catalog1 or catalog2

        By convention, the x-axis of CMDs are of the form 
        smaller λ - larger λ

        Hence, catalog1 should be the catalog of smaller λ

        ----------
        Functions: 
        ----------

        # match(self, dr_tol, dm_tol) 

        Finds matching stars between catalog1 and catalog2 within a certain 
        dr_tol (radius tolerancee) and dm_tol (magnitude tolerance). 
        
        Assumes catalogy = catalog1, or catalogy = catalog2 
        (i.e. you are generating a CMD between two wavelengths)

        No transformations are done and it is assumed both catalogs exist on 
        the same coordinate basis and magnitude system

        # generate(self, color_by_density = False): 

        Generates CMD of catalog1 - catalog2 (mag) vs. catalogy (mag). 
        Places image in image_path/

        if color_by_density == True:
            makes plot color points by their density 
            outputs a fully colored mask CMD as well


        # unsharp_mask(mag1, mag2, magy, mag1err, mag2err, magyerr, 
                       mag1_filt, mag2_filt, magy_filt, 
                       magerr_lim_max = 1.0, mask_width = 0.6, 
                       binsize_mag = 0.1, binsize_clr = 0.05, 
                       fig_dimensions = 'default', hess_extent = None, 
                       fig_path = "", data_path = "", 
                       vmini = None, vmaxi = None, vminf = None, vmaxf = None, 
                       recalc = True): 

        Apply the image-sharpening technique unsharp masking to
        the color-magnitude diagram (CMD) of the NSC to show the RC star
        region cleanly. This procedure replicates the one used by De Marchi
        et al, 2016. (https://doi.org/10.1093/mnras/stv2528)

        Makes magy vs. (mag1 - mag2) CMD

        Details of parameters are shown @ the function. 

        -----------------
        Class Parameters: 
        -----------------
            catalog1        : Pandas DataFrame
                            catalog of first starlist of smaller λ

            catalog2        : Pandas DataFrame
                            catalog of second starlist of larger λ

            catalogy        : Pandas DataFrame
                            catalog of starlist going on y axis
                            could be same as catalog1 or catalog2

            catalog1name   : String
                            name of first catalog

            catalog2name   : String
                            name of second catalog

            catalogyname    : String
                            name of y catalog

            image_path      : String
                            directory to place image files

        """

        self.catalog1 = catalog1
        self.catalog2 = catalog2 
        self.catalogy = catalogy 
        self.catalog1name = catalog1name
        self.catalog2name = catalog2name
        self.catalogyname = catalogyname
        self.image_path = image_path

    def match(self, dr_tol, dm_tol, verbose = False): 

        """

        Parameters: 
        -----------

        dr_tol  : float
                radius tolerance for stars to match
        
        dm_tol  : float
                magnitude tolerance for stars to match

        verbose : Boolean
                if you want to know # stars that matched 
                and some other info 
        """

        df, df2 = self.catalog1, self.catalog2

        x1 = df['x']   # x centroid position of catalog1 stars
        y1 = df['y']   # y centroid position of catalog1 stars 
        m1 = df['m']   # vega magnitude of catalog1 stars
        me1 = df['me'] # error in vega magnitude of catalog1 stars
        x2 = df2['x']
        y2 = df2['y']
        m2 = df2['m']
        me2 = df2['me']

        idxs1, idxs2, dr, dm = flystar.match.match(x1, y1, m1, x2, y2, m2, 
                                                  dr_tol, dm_tol, 
                                                  verbose = verbose)

        self.idxs1 = idxs1 # indexes of matched stars in catalog1
        self.idxs2 = idxs2 # indexes of matched stars in catalog2

        m1_error, m2_error = me1[idxs1], me2[idxs2]
        m1_matched, m2_matched = m1[idxs1], m2[idxs2]

        m1_error = np.array(m1_error)
        m2_error = np.array(m2_error)

        if verbose: 
            title = [self.catalog1name, self.catalog2name, "# matched"]
            print("{:>7} | {:>7} | {:>7}".format(title[0], title[1], title[2]))
            print("-----------------------------")
            print(f"{len(m1):>7} | {len(m2):>7} | {len(m1_matched):>7}")
            print("-----------------------------")

        return idxs1, idxs2, m1_matched, m2_matched, m1_error, m2_error

    def generate(self, color_by_density, dr_tol, dm_tol): 

        # catalog1 - catalog2 (mag) vs. catalogy (mag)

        #   if color_by_density == True:
        #       makes plot color points by their density 
        #       outputs a fully colored mask CMD as well

        idxs1, idxs2, m1_matched, m2_matched, m1_error, m2_error = color_magnitude_diagram.match(self, dr_tol, dm_tol)

        m1_matched = np.array(m1_matched)
        m2_matched = np.array(m2_matched)
        m_difference = np.subtract(m1_matched, m2_matched) # x axis

        if color_by_density: 
            fig, axis = plt.subplots(2, 1, figsize = (20, 20))
            nbins = 200

            check = False

            if self.catalogyname == self.catalog1name: 
                xy = np.vstack([m_difference, m1_matched])
                z = gaussian_kde(xy)(xy)

                axis[0].scatter(m_difference, m1_matched, c = z, s = 1)
                axis[0].set_xlabel(self.catalog1name + " - " + self.catalog2name)
                axis[0].set_ylabel(self.catalog1name)
                axis[0].invert_yaxis() # by convention
                axis[0].set_title(f"{self.catalog1name} - {self.catalog2name} vs. {self.catalog1name} / {len(m_difference)} stars")

                file_name = self.catalog1name + "-" + self.catalog2name + "_"\
                            + self.catalog1name + "_" + "density"

                k = kde.gaussian_kde([m_difference, m1_matched])
                xi, yi = np.mgrid[min(m_difference):max(m_difference):nbins*1j,
                                  min(m1_matched):max(m1_matched):nbins*1j]
                zi = k(np.vstack([xi.flatten(), yi.flatten()]))

                axis[1].pcolormesh(xi, yi, zi.reshape(xi.shape), shading = 'auto')
                axis[1].set_xlabel(self.catalog1name + " - " + self.catalog2name)
                axis[1].set_ylabel(self.catalog1name)
                axis[1].invert_yaxis() # by convention

                plt.savefig(f"{self.image_path}{file_name}.png")
                check = True

            if self.catalogyname == self.catalog2name: 

                xy = np.vstack([m_difference, m2_matched])
                z = gaussian_kde(xy)(xy)

                axis[0].scatter(m_difference, m2_matched, c = z, s = 1)
                axis[0].set_xlabel(self.catalog1name + " - " + self.catalog2name)
                axis[0].set_ylabel(self.catalog2name)
                axis[0].invert_yaxis() # by convention
                axis[0].set_title(f"{self.catalog1name} - {self.catalog2name} vs. {self.catalog2name} / {len(m_difference)} stars")

                file_name = self.catalog1name + "-" + self.catalog2name + "_"\
                            + self.catalog2name + "_" + "density"

                k = kde.gaussian_kde([m_difference, m2_matched])
                xi, yi = np.mgrid[min(m_difference):max(m_difference):nbins*1j,
                                  min(m2_matched):max(m2_matched):nbins*1j]
                zi = k(np.vstack([xi.flatten(), yi.flatten()]))

                axis[1].pcolormesh(xi, yi, zi.reshape(xi.shape), shading = 'auto')
                axis[1].set_xlabel(self.catalog1name + " - " + self.catalog2name)
                axis[1].set_ylabel(self.catalog1name)
                axis[1].invert_yaxis() # by convention

                plt.savefig(f"{self.image_path}{file_name}.png")  
                check = True

            if not check: 
                raise Exception("`catalogy` must equal `catalog1` or `catalog2`; ensure their names are the same too.")

        else: 
            fig, axis = plt.subplots(1, 1, figsize = (20, 10))

            check = False

            if self.catalogyname == self.catalog1name: 
                plt.scatter(m_difference, m1_matched, c = 'k', s = 0.5)
                plt.xlabel(self.catalog1name + " - " + self.catalog2name)
                plt.ylabel(self.catalog1name)
                plt.gca().invert_yaxis()
                plt.title(f"{self.catalog1name} - {self.catalog2name} vs. {self.catalog1name} / {len(m_difference)} stars") 

                file_name = self.catalog1name + "-" + self.catalog2name + "_"\
                            + self.catalog1name + "_" + "plain"

                plt.savefig(f"{self.image_path}{file_name}.png")
                check = True
                
            if self.catalogyname == self.catalog2name: 
                plt.scatter(m_difference, m2_matched, c = 'k', s = 0.5)
                plt.xlabel(self.catalog1name + " - " + self.catalog2name)
                plt.ylabel(self.catalog2name)
                plt.gca().invert_yaxis() 
                plt.title(f"{self.catalog1name} - {self.catalog2name} vs. {self.catalog2name} / {len(m_difference)} stars") 

                file_name = self.catalog1name + "-" + self.catalog2name + "_"\
                            + self.catalog2name + "_" + "plain"

                plt.savefig(f"{self.image_path}{file_name}.png")
                check = True
                
            if not check: 
                raise Exception("`catalogy` must equal `catalog1` or `catalog2`; ensure their names are the same too.")

        return m1_matched, m2_matched, m_difference