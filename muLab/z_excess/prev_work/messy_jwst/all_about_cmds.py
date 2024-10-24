import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import image
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import numpy as np
from collections import Counter
from scipy.spatial import cKDTree as KDT
from astropy.table import Column, Table
import itertools
import copy
import scipy.signal
from scipy.spatial import distance
import math
import sys
import pdb
import pickle as pickle
from astropy.convolution import convolve, Gaussian2DKernel
import astropy.coordinates as coord
import astropy.units as u
from astropy.io import ascii
import mpl_scatter_density
from scipy.stats import gaussian_kde, kde
from spisea import synthetic, evolution, atmospheres, reddening, ifmr
from spisea.imf import imf, multiplicity
import pdb
import pylab as py
from flystar import match, transforms, plots, align
from flystar import starlists
from flystar.starlists import StarList
from flystar.startables import StarTable
from astropy.table import Table, Column, vstack
import flystar
import datetime
import copy
import os
import pdb
import time
import warnings
from astropy.utils.exceptions import AstropyUserWarning




class CMD: 
    def __init__(self, catalog1, catalog2, catalog1_name, catalog2_name,
                 dr_tol, dm_tol, cmd_directory, data_directory): 

        """
        Finds star matches between two catalogs using flystar.match. No
        transformations are done as it is assumed both catalogs exist on the
        same coordinate plane and magnitude system.

        Catalogs in a similar wavelength will yield better results. 

        Afterwards builds a CMDs between two catalogs with the x axis (smaller
        λ - larger λ)

        Parameters:
        -----------
        catalog1        : Pandas DataFrame
                catalog of first starlist

        catalog2        : Pandas DataFrame
                catalog of second starlist

        catalog1_name   : String
                    name of first catalog

        catalog2_name   : String
                    name of second catalog

        dr_tol          : float
                how close stars have to be to match

        dm_tol          : float
                how close in delta-magnitude stars have to be to match


        idxs1           : Numpy Array
                list of indices in first catalog that matched

        idxs2           : Numpy Array
                list of indices in second catalog that matched

        cmd_directory   : String
                string indicating directory to place resultant cmd images

        data_directory  : String
                string indicating directory to place resultant data from
                algorithm
        """

        self.catalog1 = catalog1
        self.catalog2 = catalog2
        self.catalog1_name = catalog1_name
        self.catalog2_name = catalog2_name
        self.dr_tol = dr_tol
        self.dm_tol = dm_tol
        self.cmd_directory = cmd_directory
        self.data_directory = data_directory


    def cmd_match(self):
        
        df, df2 = self.catalog1, self.catalog2

        x1 = df['x']        # x centroid position of catalog1 stars
        y1 = df['y']        # y centroid position of catalog1 stars 
        m1 = df['m']        # vega magnitude of catalog1 stars
        me1 = df['me']      # error in vega magnitude of catalog1 stars
        x2 = df2['x']
        y2 = df2['y']
        m2 = df2['m']
        me2 = df2['me']

        idxs1, idxs2, dr, dm = flystar.match.match(x1, y1, m1, x2, y2, m2,
                                           self.dr_tol, self.dm_tol,
                                           verbose = False
                               )
        self.idxs1 = idxs1
        self.idxs2 = idxs2

        """
        idxs1: indexes of stars in catalog1 that matched with stars in
        catalog2

        idxs2: indexes of stars in catalog2 that matched with stars in
        catalog1
        """

        m1_error = []
        m2_error = []
        m1_matched = []
        m2_matched = []
        
        for i in idxs1: 
            m1_error += [me1[i]]
            m1_matched += [m1[i]]
            
        for i in idxs2: 
            m2_error += [me2[i]]
            m2_matched += [m2[i]]

        m1_error = np.array(m1_error)
        m2_error = np.array(m2_error)

        print(f"{len(m1)} total stars from {self.catalog1_name}")
        print(f"{len(m2)} total stars from {self.catalog2_name}")
        print(f"{len(m1_matched)} stars matched between {self.catalog1_name} and {self.catalog2_name}")

        return idxs1, idxs2, m1_matched, m2_matched, m1_error, m2_error

    def color_mag_diagram(self): 

        """
        Generates a color-magnitude diagram (CMD) between two catalogs.
        Places resultant color-magnitude diagrams in {cmd_directory}/

        """

        m1_matched = []
        m2_matched = []

        idxs1, idxs1, m1_matched, m2_matched, m1_error, m2_error = CMD.cmd_match(self)

        # array to be defined on x axis (m1 - m2)
        m_difference = np.subtract(m1_matched, m2_matched) 

        # generating first CMD (m1 vs. m1 - m2)
        
        fig, axis = plt.subplots(1, 1, figsize = (20, 10))

        xy = np.vstack([m_difference, m1_matched])
        z = gaussian_kde(xy)(xy)

        plt.xlabel(self.catalog1_name + " - " + self.catalog2_name)
        plt.ylabel(self.catalog1_name)

        plt.scatter(m_difference, m1_matched, c=z, s=1) 
        plt.gca().invert_yaxis()

        file_name = self.catalog1_name + "-" + self.catalog2_name + "_"\
        + self.catalog1_name

        plt.savefig(f"{self.cmd_directory}/cmd_{file_name}.png")

        # generating second CMD (m2 vs. m1 - m2)
        fig, axis = plt.subplots(1, 1, figsize = (20, 10))

        xy = np.vstack([m_difference, m2_matched])
        z = gaussian_kde(xy)(xy)

        plt.xlabel(self.catalog1_name + " - " + self.catalog2_name)
        plt.ylabel(self.catalog2_name)

        plt.scatter(m_difference, m2_matched, c=z, s=1)
        plt.gca().invert_yaxis()

        file_name = self.catalog1_name + "-" + self.catalog2_name + "_"\
        + self.catalog2_name

        plt.savefig(f"{self.cmd_directory}/cmd_{file_name}.png")

        return m1_matched, m2_matched, m_difference

    def unsharp_mask(mag1, mag2, magy, mag1err, mag2err, magyerr,
                     mag1_filt, mag2_filt, magy_filt,
                     magerr_lim_max = 1.0,
                     mask_width = 0.6,
                     binsize_mag = 0.1,
                     binsize_clr = 0.05,
                     fig_dimensions = 'default',
                     hess_extent = None,
                     fig_path = "" , data_path = "", 
                     vmini = None, vmaxi = None,
                     vminf = None, vmaxf = None,
                     recalc = True):

        """
        Apply the image-sharpening technique unsharp masking to
        the color-magnitude diagram (CMD) of the NSC to show the RC star
        region cleanly. This procedure replicates the one used by De Marchi
        et al, 2016. (https://doi.org/10.1093/mnras/stv2528)

        Makes magy vs. (mag1 - mag2) CMD

        Parameters:
        --------------
        mag1, mag2, magy     : arrays
            Magnitudes in the appropriate filters, in apparent mags

        mag1err, mag2err, magyerr: arrays
            magnitude errors in the appropriate filters

        mag1_filt, mag2_filt, magy_filt: str
            Names of mag1 and mag2 and magy filters

        magerr_lim_max  : float
           Maximum photometric error. Stars with photometric error more than
           magerr_lim_max in trim_err_cols columns will be masked.

        mask_width      : float
           Standard deviation of mask gaussian kernel, in mags;
           for unsharp mask

        binsize_mag     : float
           Size of the bin in magnitudes for the histogram in the CMD
           along the y-axis (filter 2 magnitudes)

        binsize_clr     : float
           Size of the bin in magnitudes for the histogram in the CMD
           along the x-axis (filter 1 - filter 2 magnitudes)

        fig_dimensions  : 'default' or two-element tuple
           Specifies the size of the figures; 'default' produces
           figures of size (10,10)

        hess_extent     : None or four-element array
           Specifies x-limits and y-limits of the hess diagram;
           Should be given in order of [xmin, xmax, ymin, ymax]

        fig_path        : string
           Path to directory that figures will be stored in

        data_path       : string
           Path to directory that data will be stored in

        vmin            : float
           Hess diagram minimum value

        vmax            : float
           Hess diagram maximum value

        clean_catalog   : bool
           Whether to apply photometric error cuts to the
           HST catalog before generating the Hess diagram

        recalc          : bool
           Recalculate modified Hess diagram if it already exists

        """

        unsharp_mask_extent = copy.deepcopy(hess_extent)
        final_cmd_extent = copy.deepcopy(hess_extent)

        outname = 'Unsharp_hess_{0}_{1}.pickle'.format(mag1_filt, mag2_filt)

        mag1 = np.array(mag1)
        mag2 = np.array(mag2)
        magy = np.array(magy)
        mag1err = np.array(mag1err)
        mag2err = np.array(mag2err)
        magyerr = np.array(magyerr)

        if (not os.path.exists(data_path + outname)) or (recalc == True):
            # Clean out stars with large photometric errors and
            # stars not found in relevant filters
            
            good = np.where( (np.isfinite(mag1)) & (np.isfinite(mag2)) &
                            (mag1err <= magerr_lim_max) & (mag2err <= magerr_lim_max) &
                            (magyerr <= magerr_lim_max) &
                            (mag1err > 0) & (mag2err > 0))

            print('{1} of {0} stars considered'.format(len(mag1), len(good[0])))
            mag1 = np.array(mag1)[good[0]]
            mag2 = np.array(mag2)[good[0]]
            mag1err = np.array(mag1err)[good[0]]
            mag2err = np.array(mag2err)[good[0]]
            magy = np.array(magy)[good[0]]
            magyerr = np.array(magyerr)[good[0]]

            # Compute color: mag1 - mag2
            clr_arr = abs(mag1 - mag2)
            clrerr_arr = np.hypot(mag1err, mag2err)

            # Assign mag2 to be the 'color'
            mag_arr = magy
            magerr_arr = magyerr

            # Define magnitude bin limits and the color bin limits
            # for histograms (include value extremes plus error)
            print(good)
            mag_min = np.min(mag_arr) - np.max(magerr_arr)
            mag_max = np.max(mag_arr) + np.max(magerr_arr)

            clr_min = np.min(clr_arr) - np.max(clrerr_arr)
            clr_max = np.max(clr_arr) + np.max(clrerr_arr)

            # Bins for magnitude array and bins for color array
            bins_mag = np.arange(mag_min, mag_max, binsize_mag)
            bins_clr = np.arange(clr_min, clr_max, binsize_clr)

            #-----------------------------------------------------------------#
            # Construct binned CMD of stars, where each star magnitude
            # is a 2D Guassian of width equal to photometric error
            #-----------------------------------------------------------------#

            print('Making error-weighted color-magnitude diagram.')
            #obs_pdf = np.zeros((len(mag_arr),len(bins_mag)-1,len(bins_clr)-1))
            tmp_obs_pdf = np.zeros((1000,len(bins_mag)-1,len(bins_clr)-1))
            obs_pdf_sum = np.sum(tmp_obs_pdf,axis = 0)

            # Determine the 2D Gaussian distribution of each star
            # in batches
            adjust = 0

            for ii in range(len(mag_arr)):
                # For each bin:
                mag = mag_arr[ii]                      # Magnitude of bin
                magerr = magerr_arr[ii]                # Magnitude error of bin
                clr = clr_arr[ii]                      # Color of bin
                clrerr = clrerr_arr[ii]                # Color error of bin

                # Normal continuous random variable distributions
                mag_var = scipy.stats.norm(loc=mag, scale = magerr)
                clr_var = scipy.stats.norm(loc=clr, scale = clrerr)

                # Cumulative density functions
                mag_cdf = mag_var.cdf(bins_mag)
                clr_cdf = clr_var.cdf(bins_clr)

                # Probability distributions for bin
                mag_pdf_binned = np.diff(mag_cdf)
                clr_pdf_binned = np.diff(clr_cdf)
                pdf = np.outer(mag_pdf_binned, clr_pdf_binned)

                # Add star to binned CMD array
                tmp_obs_pdf[ii-adjust] = pdf

                if ii% 1000 == 0:
                    obs_pdf_sum = obs_pdf_sum + np.sum(tmp_obs_pdf, axis=0)

                    # Reset the temporary pdf
                    tmp_obs_pdf = np.zeros((1000,len(bins_mag)-1,len(bins_clr)-1))
                    adjust += 1000
                    print('Done {0} of {1}'.format(ii, len(mag_arr)))

            # Error-weighted CMD, summed over the individual stars. Need to
            # get the contribution from the last set of stars (not even mod1000)
            obs_pdf_sum = obs_pdf_sum + np.sum(tmp_obs_pdf, axis=0)

            # Let's make sure we have the contribution from all stars (within 1 star)
            #assert np.isclose(np.sum(obs_pdf_sum), len(mag1), atol=1)

            #-----------------------------------------------------------------#
            # Plot the observed probability density functions (Hess diagram)
            #-----------------------------------------------------------------#

            # Limits of the plot; can be specified in the initiation of the
            # function using hess_extent

            if hess_extent == None:
                # If not specified, determine boundaries from color and
                # magnitude histograms
                clr_min = bins_clr[0]
                clr_max = bins_clr[-1]
                mag_min = bins_mag[0]
                mag_max = bins_mag[-1]

                hess_limits = (clr_min, clr_max, mag_max, mag_min)

            else:
                # Use given boundaries
                clr_min = hess_extent[0]
                clr_max = hess_extent[1]
                mag_max = hess_extent[2]
                mag_min = hess_extent[3]

                ind_clr_min = int(np.argmin(np.absolute(bins_clr - clr_min)))
                ind_clr_max = int(np.argmin(np.absolute(bins_clr - clr_max)))

                ind_mag_min = int(np.argmin(np.absolute(bins_mag - mag_min)))
                ind_mag_max = int(np.argmin(np.absolute(bins_mag - mag_max)))

                hess_limits = (bins_clr[ind_clr_min], bins_clr[ind_clr_max],
                                   bins_mag[ind_mag_max], bins_mag[ind_mag_min])

            if fig_dimensions == 'default':
                fig_dimensions = (20,10)

            # Initialize figure for unedited Hess diagram
            fig1, ax1 = plt.subplots(1, 1, figsize = fig_dimensions)

            # Plot the Hess diagram
            if hess_extent == None:
                hess_plot = ax1.imshow(obs_pdf_sum, extent = hess_limits,
                                    vmin = vmini, vmax = vmaxi)
            else:
                hess_plot = ax1.imshow(obs_pdf_sum[ind_mag_min:ind_mag_max, ind_clr_min:ind_clr_max],
                                           extent = hess_limits,
                                           vmin = vmini, vmax = vmaxi)
            # Label plot
            ax1.set_xlabel('{0} - {1} (mag)'.format(mag1_filt, mag2_filt))
            ax1.set_ylabel('{0} (mag)'.format(magy_filt))
            cbar_hess = plt.colorbar(hess_plot, ax = ax1)
            cbar_hess.set_label('Number of stars per bin')
            ax1.axis('tight')
            fig1.savefig(fig_path + 'error_weighted_hess_diagram_{0}_{1}_{2}.png'.format(mag1_filt, 
                                                                                         mag2_filt,
                                                                                         magy_filt)
                        )

            #-----------------------------------------------------------------#
            # Generate the unsharp mask
            #-----------------------------------------------------------------#
            # Create astropy kernel object of desired width based on the magnitude
            # binsize
            sigma_mask = mask_width / binsize_mag
            gauss = Gaussian2DKernel(sigma_mask)

            # Create mask by convolving gaussian kernel with pdf
            mask = convolve(obs_pdf_sum, gauss)

            #-----------------------------------------------------------------#
            # Plot the unsharp mask
            #-----------------------------------------------------------------#

            # Limits of the plot; can be specified in the initiation of the
            # function using unsharp_mask_extent

            if unsharp_mask_extent == None:
                clr_min = bins_clr[0]
                clr_max = bins_clr[-1]
                mag_min = bins_mag[0]
                mag_max = bins_mag[-1]

                unsharp_mask_limits = (clr_min, clr_max, mag_max, mag_min)
            else:
                clr_min = unsharp_mask_extent[0]
                clr_max = unsharp_mask_extent[1]
                mag_max = unsharp_mask_extent[2]
                mag_min = unsharp_mask_extent[3]

                ind_clr_min = int(np.argmin(np.absolute(bins_clr - clr_min)))
                ind_clr_max = int(np.argmin(np.absolute(bins_clr - clr_max)))

                ind_mag_min = int(np.argmin(np.absolute(bins_mag - mag_min)))
                ind_mag_max = int(np.argmin(np.absolute(bins_mag - mag_max)))

                unsharp_mask_limits = (bins_clr[ind_clr_min], bins_clr[ind_clr_max],
                                           bins_mag[ind_mag_max], bins_mag[ind_mag_min])

            # Plot values
            if fig_dimensions == 'default':
                fig_dimensions = (20,10)

            # Initialize plot
            fig2, ax2 = plt.subplots(1, 1, figsize = fig_dimensions)

            if unsharp_mask_extent == None:
                mask_plot = ax2.imshow(mask, extent = unsharp_mask_limits,
                                vmin = vmini, vmax = vmaxi)
            else:
                mask_plot = ax2.imshow(mask[ind_mag_min:ind_mag_max,
                                    ind_clr_min:ind_clr_max],
                                    extent = unsharp_mask_limits,
                                    vmin = vmini, vmax = vmaxi)

            ax2.set_xlabel('{0} - {1} (mag)'.format(mag1_filt, mag2_filt))
            ax2.set_ylabel('{0} (mag)'.format(magy_filt))
            cbar_mask = plt.colorbar(mask_plot, ax = ax2)
            cbar_mask.set_label('Number of stars per bin')
            ax2.axis('tight')

            fig2.savefig(fig_path + 'unsharp_mask_{0}_{1}_{2}.png'.format(mag1_filt,
                                                                          mag2_filt,
                                                                          magy_filt)
                        )

            #-----------------------------------------------------------------#
            # Subtract mask from original binned CMD (Hess diagram) to get
            # the final, sharpened CMD
            #-----------------------------------------------------------------#

            print('Generating final pdf.')
            pdf_final = obs_pdf_sum - mask

            #-----------------------------------------------------------------#
            # Plot the final, sharpened binned CMD
            #-----------------------------------------------------------------#

            # Limits of the plot; can be specified in the initiation of the
            # function using final_cmd_extent

            if final_cmd_extent == None:
                clr_min = bins_clr[0]
                clr_max = bins_clr[-1]
                mag_min = bins_mag[0]
                mag_max = bins_mag[-1]

                final_cmd_limits = (clr_min, clr_max,mag_max, mag_min)

            else:
                clr_min = final_cmd_extent[0]
                clr_max = final_cmd_extent[1]
                mag_max = final_cmd_extent[2]
                mag_min = final_cmd_extent[3]

                ind_clr_min = int(np.argmin(np.absolute(bins_clr - clr_min)))
                ind_clr_max = int(np.argmin(np.absolute(bins_clr - clr_max)))

                ind_mag_min = int(np.argmin(np.absolute(bins_mag - mag_min)))
                ind_mag_max = int(np.argmin(np.absolute(bins_mag - mag_max)))

                final_cmd_limits = (bins_clr[ind_clr_min], bins_clr[ind_clr_max],
                                        bins_mag[ind_mag_max], bins_mag[ind_mag_min])

            # Plot values
            if fig_dimensions == 'default':
                fig_dimensions = (20,10)

            # Initialize plot
            fig3, ax3 = plt.subplots(1, 1, figsize = fig_dimensions)

            if final_cmd_extent == None:
                final_pdf_plot = ax3.imshow(pdf_final, extent = final_cmd_limits,
                                        vmin = vminf, vmax = vmaxf)
            else:
                final_pdf_plot = ax3.imshow(pdf_final[ind_mag_min:ind_mag_max,
                                    ind_clr_min:ind_clr_max],
                                    extent = final_cmd_limits,
                                    vmin = vminf, vmax = vmaxf)
            ax3.set_xlabel('{0} - {1} (mag)'.format(mag1_filt, mag2_filt))
            ax3.set_ylabel('{0} (mag)'.format(magy_filt))
            cbar_final = plt.colorbar(final_pdf_plot, ax = ax3)
            cbar_final.set_label('Number of stars per bin')
            ax3.axis('tight')

            fig3.savefig(fig_path + 'unsharp_final_cmd_{0}_{1}_{2}.png'.format(mag1_filt,
                                                                               mag2_filt,
                                                                               magy_filt)
                        )

            # Save the final probability distribution function data as a
            # pickle file

            _out = open(data_path + outname, 'wb')

            pickle.dump(pdf_final, _out)
            pickle.dump(bins_mag, _out)
            pickle.dump(bins_clr, _out)

            _out.close

        return outname


    def save_hist(hist, magbins, clrbins, outName):
        """
        Helper function to save the histogram bins
        and 2D Hess diagram (hist) with name
        outName

        Parameters:
        _____________
        hist    : 2D numpy array
           2D image-sharpened probability density
           function

        magbins : 1D numpy array
           bin boundaries for magnitudes

        clrbins : 1D numpy array
           bin boundaries for colors

        outName : string
           path + filename of the pickle file
           of the histogram

        """

        _out = open(outName, 'wb')

        pickle.dump(hist, _out)
        pickle.dump(magbins, _out)
        pickle.dump(clrbins, _out)

        _out.close()

        return

    def generate_hess_diagram(self): 

        idxs1, idxs2, m1, m2, m1_error, m2_error = CMD.cmd_match(self)

        # Hess Diagram m1 vs. (m1 - m2)
        hess1 = CMD.unsharp_mask(m1, m2, m1, m1_error, m2_error, m1_error, 
                                 self.catalog1_name, self.catalog2_name,
                                 self.catalog1_name, 
                                 fig_path = self.cmd_directory, 
                                 data_path = self.data_directory)

        # Hess Diagram m2 vs. (m1 - m2)
        hess1 = CMD.unsharp_mask(m1, m2, m2, m1_error, m2_error, m2_error,
                                 self.catalog1_name, self.catalog2_name,
                                 self.catalog2_name,
                                 fig_path = self.cmd_directory,
                                 data_path = self.data_directory)


class Isochrones: 
    def __init__(self, catalog1, catalog2, 
                 filt1_name, filt2_name, 
                 dr_tol, dm_tol, 
                 logAge, AKs, AKs_step, dist, 
                 metallicity, mass, 
                 iso_dir, cmd_directory, data_directory): 

        """ 
        Generates theoretical isochrone(s) based on the age, metallicity, and
        distance of a starcluster. Uses the MIST evolution moden and the
        Fritz+11 extinction Law. Implemented using SPISEA (Hosek+18 et al.)

        Afterwards plots isochrones of increasing extinction (AKs) against
        a color-magnitude diagram. Theoretically the isochrones should `follow
        along` with the slope of the RC cluster on the color-magnitude diagram.

        Parameters: 
        -----------

        catalog1        : Pandas DataFrame
                catalog of first starlist wavelength

        catalog2        : Pandas DataFrame
                catalog of second starlist wavelength

        filt1_name   : String
                name of first catalog -- see SPISEA documentation 

        filt2_name   : String
                name of second catalog -- see SPISEA documentation

        dr_tol          : float
                how close stars have to be between two catalogs to match

        dm_tol          : float
                how close in delta-magnitude stars between two catalogs have to
                be to match

        logAge          : float
                age of starcluster in log years: np.log(10**9) -- 1 billion
                year old starcluster

        AKs             : float
                extinction in Ks band (~2.2µm band)

        AKs_step        : float
                step value increase of following isochrones to align with Red
                Clump cluster on CMD

        dist            : float
                distance to starcluster in parsecs

        metallicity     : float
                metallicity of starcluster

        iso_dir         : String
                directory to place generated theoretical isochrones

        mass            : float
                total starcluster mass (e.g. 10**5)

        """

        self.catalog1 = catalog1
        self.catalog2 = catalog2
        self.filt1_name = filt1_name
        self.filt2_name = filt2_name
        self.dr_tol = dr_tol
        self.dm_tol = dm_tol
        self.dist = dist
        self.logAge = logAge
        self.AKs = AKs
        self.AKs_step = AKs_step
        self.metallicity = metallicity
        self.iso_dir = iso_dir
        self.cmd_directory = cmd_directory
        self.data_directory = data_directory

    def generate_isochrones(self): 

        evo_model = evolution.MISTv1()
        atm_func  = atmospheres.get_merged_atmosphere
        red_law   = reddening.RedLawFritz11(scale_lambda=2.166)
        filt_list = [self.filt1_name, self.filt2_name]

        AKs2 = self.AKs  + self.AKs_step
        AKs3 = AKs2 + self.AKs_step
        AKs4 = AKs3 + self.AKs_step
        AKs5 = AKs4 + self.AKs_step

        # Generating Isochrones
        

        my_iso = synthetic.IsochronePhot(self.logAge, self.AKs, self.dist,
                                         self.metallicity,
                                         evo_model=evo_model, atm_func=atm_func,
                                         red_law=red_law, filters=filt_list,
                                         iso_dir=self.iso_dir)

        my_iso2 = synthetic.IsochronePhot(self.logAge, AKs2, self.dist, self.metallicity,
                                          evo_model=evo_model, atm_func=atm_func,
                                          red_law=red_law, filters=filt_list,
                                          iso_dir=self.iso_dir)

        my_iso3 = synthetic.IsochronePhot(self.logAge, AKs3, self.dist, self.metallicity,
                                          evo_model=evo_model, atm_func=atm_func,
                                          red_law=red_law, filters=filt_list,
                                          iso_dir=self.iso_dir)

        my_iso4 = synthetic.IsochronePhot(self.logAge, AKs4, self.dist, self.metallicity,
                                          evo_model=evo_model, atm_func=atm_func,
                                          red_law=red_law, filters=filt_list,
                                          iso_dir=self.iso_dir)

        my_iso5 = synthetic.IsochronePhot(self.logAge, AKs5, self.dist, self.metallicity,
                                          evo_model=evo_model, atm_func=atm_func,
                                          red_law=red_law, filters=filt_list,
                                          iso_dir=self.iso_dir)

        file_name = filt_list[0] + "_" + filt_list[1]

        dfy = pd.DataFrame(my_iso.points['phase'])
        dfy.to_csv(f"{self.iso_dir}" + f"spisea_iso{file_name}.csv")

        print('The columns in the isochrone table are:\
              {0}'.format(my_iso.points.keys()))


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
              mag'.format(self.filt1_name, filter_1, self.filt2_name,
              filter_2))
            
         
        return my_iso, my_iso2, my_iso3, my_iso4, my_iso5, idx, idx2, idx3, idx4, idx5


    def synthetic_isochrones(self): 

        my_iso, my_iso2, my_iso3, my_iso4, my_iso5, idx, idx2, idx3, idx4, idx5 = Isochrones.generate_isochrones(self)

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
        # synthetic isochrone for (m1 vs. m1 - m2)
        fig, axis = plt.subplots(1,1, figsize = (20,10))

        axis.plot(clust[''+my_iso.points.keys()[8]]
                     - clust[''+my_iso.points.keys()[9]], 
                     clust[''+my_iso.points.keys()[8]], 'k.', ms=5, alpha=0.1, label='__nolegend__')
        axis.plot(iso[''+my_iso.points.keys()[8]]
                     - iso[''+my_iso.points.keys()[9]],
                     iso[''+my_iso.points.keys()[8]],
               'r-', label='Isochrone')
        axis.set_xlabel(self.filt1_name + " - " + self.filt2_name)
        axis.set_ylabel(self.filt1_name)
        axis.invert_yaxis()
        axis.legend()

        file_name = "synthetic_iso_" + self.filt1_name + "-" + self.filt2_name + "_" + self.filt1_name
        plt.savefig(f"{self.iso_dir}" + f"{file_name}.png")
        
        # synthetic isochrone for (m2 vs. m1 - m2)
        fig, axis = plt.subplots(1,1, figsize = (20,10))

        axis.plot(clust[''+my_iso.points.keys()[8]]
                     - clust[''+my_iso.points.keys()[9]],
                     clust[''+my_iso.points.keys()[9]], 'k.', ms=5, alpha=0.1, label='__nolegend__')
        axis.plot(iso[''+my_iso.points.keys()[8]]
                     - iso[''+my_iso.points.keys()[9]],
                     iso[''+my_iso.points.keys()[9]],
               'r-', label='Isochrone')
        axis.set_xlabel(self.filt1_name + " - " + self.filt2_name)
        axis.set_ylabel(self.filt2_name)
        axis.invert_yaxis()
        axis.legend()

        file_name = "synthetic_iso_" + self.filt1_name + "-" + self.filt2_name + "_" + self.filt2_name
        plt.savefig(f"{self.iso_dir}" + f"{file_name}.png")

        return

    def cmd_isochrones(self): 

        CMD_class = CMD(self.catalog1, self.catalog2, 
                        self.filt1_name, self.filt2_name, 
                        self.dr_tol, self.dm_tol, 
                        self.cmd_directory, self.data_directory)

        my_iso, my_iso2, my_iso3, my_iso4, my_iso5, idx, idx2, idx3, idx4, idx5 = Isochrones.generate_isochrones(self)

        m1_match, m2_match, arr_diff = CMD_class.color_mag_diagram()

        # plotting first isochrone-cmd plot (m1 vs. m1 - m2)
        fig, axis = plt.subplots(1, 1, figsize = (20, 10))
        print(my_iso.points.keys()[8])

        axis.plot(my_iso.points[''+my_iso.points.keys()[8]]
                - my_iso.points[''+my_iso.points.keys()[9]],
            my_iso.points[''+my_iso.points.keys()[8]], 'r-', label='_nolegend_')
        axis.plot(my_iso.points[''+my_iso.points.keys()[8]][idx]
                - my_iso.points[''+my_iso.points.keys()[9]][idx],
           my_iso.points[''+my_iso.points.keys()[8]][idx], 'b*', ms=15, label='1 $M_\odot$')

        axis.plot(my_iso2.points[''+my_iso2.points.keys()[8]]
                - my_iso2.points[''+my_iso2.points.keys()[9]],
            my_iso2.points[''+my_iso2.points.keys()[8]], 'r-', label='_nolegend_')
        axis.plot(my_iso2.points[''+my_iso2.points.keys()[8]][idx2]
                - my_iso2.points[''+my_iso2.points.keys()[9]][idx2],
           my_iso2.points[''+my_iso2.points.keys()[8]][idx2], 'b*', ms=15, label='_nolegend_')

        axis.plot(my_iso3.points[''+my_iso3.points.keys()[8]]
                - my_iso3.points[''+my_iso3.points.keys()[9]],
            my_iso3.points[''+my_iso3.points.keys()[8]], 'r-', label='_nolegend_')
        axis.plot(my_iso3.points[''+my_iso3.points.keys()[8]][idx3]
             - my_iso3.points[''+my_iso3.points.keys()[9]][idx3],
           my_iso3.points[''+my_iso3.points.keys()[8]][idx3], 'b*', ms=15, label='_nolegend_')

        axis.plot(my_iso4.points[''+my_iso4.points.keys()[8]]
            - my_iso4.points[''+my_iso4.points.keys()[9]],
            my_iso4.points[''+my_iso4.points.keys()[8]], 'r-', label='_nolegend_')
        axis.plot(my_iso4.points[''+my_iso4.points.keys()[8]][idx4]
            - my_iso4.points[''+my_iso4.points.keys()[9]][idx4],
           my_iso4.points[''+my_iso4.points.keys()[8]][idx4], 'b*', ms=15, label='_nolegend_')

        axis.plot(my_iso5.points[''+my_iso5.points.keys()[8]]
                - my_iso5.points[''+my_iso5.points.keys()[9]],
            my_iso5.points[''+my_iso5.points.keys()[8]], 'r-', label='_nolegend_')
        axis.plot(my_iso5.points[''+my_iso5.points.keys()[8]][idx5]
                - my_iso5.points[''+my_iso5.points.keys()[9]][idx5],
           my_iso5.points[''+my_iso5.points.keys()[8]][idx5], 'b*', ms=15, label='_nolegend_')

        nbins=600
        k = kde.gaussian_kde([arr_diff, m1_match])
        xi, yi = np.mgrid[min(arr_diff):max(arr_diff):nbins*1j,
                              min(m1_match):max(m1_match):nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        axis.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
        axis.invert_yaxis()
        axis.set_xlabel(f"{self.filt1_name} - {self.filt2_name}")
        axis.set_ylabel(f"{self.filt1_name}")

        file_name = "spisea_" + self.filt1_name + "-" + self.filt2_name + "_" + self.filt1_name
        plt.savefig(f"{self.cmd_directory}" + f"{file_name}.png")

        # plotting second isochrone-cmd plot (m2 vs. m1 - m2)
        fig, axis = plt.subplots(1, 1, figsize = (20, 10))

        axis.plot(my_iso.points[''+my_iso.points.keys()[8]]
                - my_iso.points[''+my_iso.points.keys()[9]],
            my_iso.points[''+my_iso.points.keys()[9]], 'r-', label='_nolegend_')
        axis.plot(my_iso.points[''+my_iso.points.keys()[8]][idx]
                - my_iso.points[''+my_iso.points.keys()[9]][idx],
           my_iso.points[''+my_iso.points.keys()[9]][idx], 'b*', ms=15, label='1 $M_\odot$')

        axis.plot(my_iso2.points[''+my_iso2.points.keys()[8]]
                - my_iso2.points[''+my_iso2.points.keys()[9]],
            my_iso2.points[''+my_iso2.points.keys()[9]], 'r-', label='_nolegend_')
        axis.plot(my_iso2.points[''+my_iso2.points.keys()[8]][idx2]
                - my_iso2.points[''+my_iso2.points.keys()[9]][idx2],
           my_iso2.points[''+my_iso2.points.keys()[9]][idx2], 'b*', ms=15, label='_nolegend_')

        axis.plot(my_iso3.points[''+my_iso3.points.keys()[8]]
                - my_iso3.points[''+my_iso3.points.keys()[9]],
            my_iso3.points[''+my_iso3.points.keys()[9]], 'r-', label='_nolegend_')
        axis.plot(my_iso3.points[''+my_iso3.points.keys()[8]][idx3]
             - my_iso3.points[''+my_iso3.points.keys()[9]][idx3],
           my_iso3.points[''+my_iso3.points.keys()[9]][idx3], 'b*', ms=15, label='_nolegend_')

        axis.plot(my_iso4.points[''+my_iso4.points.keys()[8]]
            - my_iso4.points[''+my_iso4.points.keys()[9]],
            my_iso4.points[''+my_iso4.points.keys()[9]], 'r-', label='_nolegend_')
        axis.plot(my_iso4.points[''+my_iso4.points.keys()[8]][idx4]
            - my_iso4.points[''+my_iso4.points.keys()[9]][idx4],
           my_iso4.points[''+my_iso4.points.keys()[9]][idx4], 'b*', ms=15, label='_nolegend_')
 
        axis.plot(my_iso5.points[''+my_iso5.points.keys()[8]]
                - my_iso5.points[''+my_iso5.points.keys()[9]],
            my_iso5.points[''+my_iso5.points.keys()[9]], 'r-', label='_nolegend_')
        axis.plot(my_iso5.points[''+my_iso5.points.keys()[8]][idx5]
                - my_iso5.points[''+my_iso5.points.keys()[9]][idx5],
           my_iso5.points[''+my_iso5.points.keys()[9]][idx5], 'b*', ms=15, label='_nolegend_')

        k = kde.gaussian_kde([arr_diff, m2_match])
        xi, yi = np.mgrid[min(arr_diff):max(arr_diff):nbins*1j,
                              min(m2_match):max(m2_match):nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        axis.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
        axis.invert_yaxis()
        axis.set_xlabel(f"{self.filt1_name} - {self.filt2_name}")
        axis.set_ylabel(f"{self.filt2_name}")

        file_name = "spisea_" + self.filt1_name + "-" + self.filt2_name + "_" + self.filt2_name
        plt.savefig(f"{self.cmd_directory}" + f"{file_name}.png")

        return


class Centroid: 
    def __init__(self, catalogs, catalogs_name, plot_dir):

        self.catalogs = catalogs
        self.catalogs_name = catalogs_name
        self.plot_dir = plot_dir

        """
        Parameters:
        -----------

        catalogs: NumPy Array
        contains dataframes to each catalog you wish to plot

        catalogs_name: list
        contains the names of catalogs in `catalogs'

        plot_dir: String
        directory to place plot .pngs. 
        """ 

    def centroid_plot(self): 

        #plotting individual catalogs
        for i in range(len(self.catalogs)): 
            fig, axis = plt.subplots(1, 1, figsize = (10, 10))
            x = self.catalogs[i]['x']
            y = self.catalogs[i]['y']

            plt.plot(x, y, 'k+')
            plt.xlabel(f'{self.catalogs_name[i]} x')
            plt.ylabel(f'{self.catalogs_name[i]} y')
            plt.title(f'{self.catalogs_name[i]} centroid plot')
            plt.savefig(f'{self.plot_dir}' + f'{self.catalogs_name[i]}.png')

        # plotting all catalogs together
        colors = ['k+', 'r+', 'g+', 'b+', 'y+', 'm+']
        fig, axis = plt.subplots(1, 1, figsize = (10, 10))
        for i in range(len(self.catalogs)): 
            x = self.catalogs[i]['x']
            y = self.catalogs[i]['y']

            plt.plot(x, y, colors[i], label = f'{self.catalogs_name[i]}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.title('all centroids plotted')
        
        plt.savefig(f'{self.plot_dir}' + 'all_catalogs.png') 

        return


#------------------------------------------------#

f115w = Table.read('catalogs/dr2/catalog115w.csv')
f212n = Table.read('catalogs/dr2/catalog212n.csv')
f115wf = starlists.StarList.from_table(f115w)
f212nf = starlists.StarList.from_table(f212n)
print(type(f115w))
print(type(f212n))

msc = align.MosaicSelfRef([f115wf, f212nf], iters=1,
                            dr_tol=[0.5], dm_tol=[99],
                            outlier_tol=[None], mag_lim= None,
                            trans_class=transforms.PolyTransform,
                            trans_args=[{'order': 1}],
                            use_vel=False,
                            ref_index = 0,
                            mag_trans=True,
                            weights='both,std',
                            init_guess_mode='name', verbose=True
                         )
msc.fit()
trans_list = msc.trans_args
stars_table = msc.ref_table




