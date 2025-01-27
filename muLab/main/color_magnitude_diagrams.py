import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import pickle 
import flystar
import copy
import os 

from scipy.spatial import cKDTree as KDT
from numpy.polynomial.polynomial import polyfit
from scipy.stats import norm, tstd, gaussian_kde, kde
from astropy.table import * 
from astropy.convolution import convolve, Gaussian2DKernel
from flystar import match, transforms, plots, align, starlists
from flystar.starlists import StarList
from flystar.startables import StarTable

#not used here
import pdb
import pylab as py

from astropy.modeling import models, fitting
from spisea import synthetic, evolution, atmospheres, reddening, ifmr
from spisea.imf import imf, multiplicity



class Color_Magnitude_Diagram: 
    def __init__(self, catalog1, catalog2, 
                catalog1name, catalog2name, catalogyname, 
                image_path = None, dr_tol = None, dm_tol = None): 

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
        # def match(self, dr_tol, dm_tol) 

        Finds matching stars between catalog1 and catalog2 within a certain 
        dr_tol (radius tolerancee) and dm_tol (magnitude tolerance). 
        
        Assumes catalogy = catalog1, or catalogy = catalog2 
        (i.e. you are generating a CMD between two wavelengths)

        No transformations are done and it is assumed both catalogs exist on 
        the same coordinate basis and magnitude system

        # def generate(self, color_by_density = False): 

        Generates CMD of catalog1 - catalog2 (mag) vs. catalogy (mag). 
        Places image in image_path/

        if color_by_density == True:
            makes plot color points by their density 
            outputs a fully colored mask CMD as well


        # def unsharp_mask(mag1, mag2, magy, mag1err, mag2err, magyerr, 
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
            catalog1        : pandas dataframe
                            catalog of first starlist of smaller λ

            catalog2        : pandas dataframe
                            catalog of second starlist of larger λ

            catalog1name   : string
                            name of first catalog

            catalog2name   : string
                            name of second catalog

            catalogyname    : string
                            name of y catalog

            dr_tol          : float
                            radius tolerance if performing matching

            dm_tol          : float
                            magnitude tolerance if performing matching

            image_path      : string
                            directory to place image files

        """

        self.catalog1 = catalog1
        self.catalog2 = catalog2 
        self.catalog1name = catalog1name
        self.catalog2name = catalog2name
        self.catalogyname = catalogyname
        self.dr_tol = dr_tol
        self.dm_tol = dm_tol
        self.image_path = image_path

    def match(self, verbose = False): 

        """

        Parameters: 
        -----------
        dr_tol  : float
                radius tolerance for stars to match
        
        dm_tol  : float
                magnitude tolerance for stars to match

        verbose : bool
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
                                                  self.dr_tol, self.dm_tol, 
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

    def generate(self, color_by_density = False): 

        """
        catalog1 - catalog2 (mag) vs. catalogy (mag)

        if color_by_density == True:
            makes plot color points by their density 
            outputs a fully colored mask CMD as well

        """

        idxs1, idxs2, m1_matched, m2_matched, m1_error, m2_error = Color_Magnitude_Diagram.match(self)

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
                raise Exception("`catalogyname` must equal `catalog1name` or `catalog2name`")

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
                raise Exception("`catalogyname` must equal `catalog1name` or `catalog2name`")

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
                     recalc = True, verbose = False): 

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
            if verbose:
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
                mag_var = norm(loc=mag, scale = magerr)
                clr_var = norm(loc=clr, scale = clrerr)

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
                    if verbose: 
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

    def generate_unsharp_mask(self, fig_path, data_path, verbose): 

        # Generates the unsharp masked CMD using `unsharp_mask` function

        idxs1, idxs2, m1, m2, m1_error, m2_error = Color_Magnitude_Diagram.match(self)
        check = False

        if self.catalogyname == self.catalog1name: 
            hess = Color_Magnitude_Diagram.unsharp_mask(m1, m2, m1, 
                                                        m1_error, m2_error, m1_error, 
                                                        self.catalog1name, self.catalog2name, 
                                                        self.catalog1name, 
                                                        fig_path = fig_path, 
                                                        data_path = data_path)
            check = True

        if self.catalogyname == self.catalog2name: 
            hess = Color_Magnitude_Diagram.unsharp_mask(m1, m2, m2, 
                                                        m1_error, m2_error, m2_error, 
                                                        self.catalog1name, self.catalog2name, 
                                                        self.catalog2name, 
                                                        fig_path = fig_path, 
                                                        data_path = data_path)

        return