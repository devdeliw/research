import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import image
import numpy as np
from flystar import match
from collections import Counter
from scipy.spatial import cKDTree as KDT
from astropy.table import Column, Table
import itertools
import copy
import scipy.signal
from scipy.spatial import distance
import math
import pdb
import pickle as pickle
from astropy.convolution import convolve, Gaussian2DKernel
import astropy.coordinates as coord
import astropy.units as u
from astropy.io import ascii
import mpl_scatter_density
from scipy.stats import gaussian_kde


class Generate_CMD: 
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

        fig, axis = plt.subplots(1,1, figsize = (20,10)) 

        if self.y_axis_m1 == False:
            xy = np.vstack([arr_diff, m2_match])
            z = gaussian_kde(xy)(xy)

            plt.xlabel(self.catalog1_name + " - " + self.catalog2_name)
            plt.ylabel(self.catalog2_name) 

            plt.scatter(arr_diff, m2_match, c = z, s = 1)
            plt.gca().invert_yaxis()
            plt.savefig("/Users/devaldeliwala/research/jwst_extinction/img/cmd/jwst_115212_212.png")

        else: 
            xy = np.vstack([arr_diff, m1_match])
            z = gaussian_kde(xy)(xy)

            plt.xlabel(self.catalog1_name + " - " + self.catalog2_name)
            plt.ylabel(self.catalog1_name)

            plt.scatter(arr_diff, m1_match, c = z, s = 1)
            plt.gca().invert_yaxis()
            plt.savefig("/Users/devaldeliwala/research/jwst_extinction/img/cmd/jwst_115212_115.png")

        return m1_match, m2_match


    def unsharp_mask(self, mag1, mag2, mag1err, mag2err,
                     mag1_filt, mag2_filt,
                     magerr_lim_max = 1.0,
                     mask_width = 0.2,
                     binsize_mag = 0.05,
                     binsize_clr = 0.025,
                     fig_dimensions = 'default',
                     hess_extent = None,
                     fig_path
                     = "/Users/devaldeliwala/research/jwst_extinction/img/unsharp_mask/",
                     data_path
                     = "/Users/devaldeliwala/research/jwst_extinction/img/unsharp_mask/",
                     vmini = None, vmaxi = None,
                     vminf = None, vmaxf = None,
                     recalc = False):
        
        """
        Apply the image-sharpening technique unsharp masking to
        the color-magnitude diagram (CMD) of the NSC to show the RC star 
        region cleanly. This procedure replicates the one used by De Marchi
        et al, 2016. (https://doi.org/10.1093/mnras/stv2528)

        Makes mag2 vs. (mag1 - mag2) CMD

        Parameters:
        --------------
        mag1, mag2      : arrays
            Magnitudes in the appropriate filters, in apparent mags

        mag1err, mag2err: arrays
            magnitude errors in the appropriate filters

        mag1_filt, mag2_filt: str
            Names of mag1 and mag2 filters

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
        print(type(mag1), type(mag1err))

        outname = 'Unsharp_hess_{0}_{1}.pickle'.format(mag1_filt, mag2_filt)
        
        mag1 = np.array(mag1)
        mag2 = np.array(mag2)
 
        if (not os.path.exists(data_path + outname)) or (recalc == True):
            # Clean out stars with large photometric errors and
            # stars not found in relevant filters
            good = np.where( (np.isfinite(mag1)) & (np.isfinite(mag2)) &
                            (mag1err <= magerr_lim_max) & (mag2err <= magerr_lim_max) &
                            (mag1err > 0) & (mag2err > 0))

            print('{1} of {0} stars considered'.format(len(mag1), len(good[0])))
            mag1 = np.array(mag1)[good[0]]
            mag2 = np.array(mag2)[good[0]]
            mag1err = np.array(mag1err)[good[0]]
            mag2err = np.array(mag2err)[good[0]]

            # Compute color: mag1 - mag2
            clr_arr = abs(mag1 - mag2)
            clrerr_arr = np.hypot(mag1err, mag2err)

            # Assign mag2 to be the 'color'
            mag_arr = mag2
            magerr_arr = mag2err

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

            #############################f######################################
            # Construct binned CMD of stars, where each star magnitude
            # is a 2D Guassian of width equal to photometric error
            ###################################################################

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
            assert np.isclose(np.sum(obs_pdf_sum), len(mag1), atol=1)
        
            ###################################################################
            # Plot the observed probability density functions (Hess diagram)
            ###################################################################
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
                fig_dimensions = (10,10)

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
            ax1.set_ylabel('{0} (mag)'.format(mag2_filt))
            cbar_hess = plt.colorbar(hess_plot, ax = ax1)
            cbar_hess.set_label('Number of stars per bin')
            ax1.axis('tight')
            fig1.savefig(fig_path + 'error_weighted_hess_diagram_{0}_{1}.png'.format(mag1_filt, mag2_filt))
        
            ###################################################################
            # Generate the unsharp mask
            ###################################################################
            # Create astropy kernel object of desired width based on the magnitude
            # binsize
            sigma_mask = mask_width / binsize_mag
            gauss = Gaussian2DKernel(sigma_mask)

            # Create mask by convolving gaussian kernel with pdf
            mask = convolve(obs_pdf_sum, gauss)

            ###################################################################
            # Plot the unsharp mask
            ###################################################################

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
                fig_dimensions = (10,10)

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
            ax2.set_ylabel('{0} (mag)'.format(mag2_filt))
            cbar_mask = plt.colorbar(mask_plot, ax = ax2)
            cbar_mask.set_label('Number of stars per bin')
            ax2.axis('tight')

            fig2.savefig(fig_path + 'unsharp_mask_{0}_{1}.png'.format(mag1_filt, mag2_filt))

            ###################################################################
            # Subtract mask from original binned CMD (Hess diagram) to get
            # the final, sharpened CMD
            ###################################################################
            print('Generating final pdf.')
            pdf_final = obs_pdf_sum - mask

            ###################################################################
            # Plot the final, sharpened binned CMD
            ###################################################################

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
                fig_dimensions = (10,10)

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
            ax3.set_ylabel('{0} (mag)'.format(mag2_filt))
            cbar_final = plt.colorbar(final_pdf_plot, ax = ax3)
            cbar_final.set_label('Number of stars per bin')
            ax3.axis('tight')

            fig3.savefig(fig_path + 'unsharp_final_cmd_{0}_{1}.png'.format(mag1_filt, mag2_filt))

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
        hist: 2D numpy array
           2D image-sharpened probability density
           function

        magbins: 1D numpy array
           bin boundaries for magnitudes

        clrbins: 1D numpy array
           bin boundaries for colors

        outName: string
           path + filename of the pickle file
           of the histogram

        """

        _out = open(outName, 'wb')

        pickle.dump(hist, _out)
        pickle.dump(magbins, _out)
        pickle.dump(clrbins, _out)

        _out.close()

        return


class RC_stars: 
    def __init__(self, m1_match, m2_match, slope1, b_min1, b_max1, slope2,
                 b_min2, b_max2):

        """
        Determines finalized Red Clump (RC) stars by analyzing stars within the red clump 'bar' 
        on the color magnitude diagram. 

        Parameters: 
        -----------
        
        m1_match    : list
            list of first catalog magnitudes from matched stars after using 
            Generate_CMD.match()

        m2_match    : list
            list of second catalog magnitudes from matched stars after using
            Generate_CMD.match()

        slope1, slope2 : float
            the slope of linear RC bar cutoffs for (vs. m1, and vs. m2),
            respectively

        b_min1 & b_min2         : float
            the y_intercept for the lower RC bar cutoff for (vs. m1, vs. m2)
            respectively

        b_max1 & b_max2         : float
            the y_intercept for the upper RC bar cutoff for (vs. m1, vs. m2)
            respectively

        """

        self.m1_match = m1_match
        self.m2_match = m2_match
        self.slope1 = slope1
        self.slope2 = slope2
        self.b_min1 = b_min1
        self.b_max1 = b_max1
        self.b_min2 = b_min2
        self.b_max2 = b_max2

    def rc_candidates(self): 

        # First finding RC candidates from the (m1 - m2 v. m2) CMD

        x = np.subtract(self.m1_match, self.m2_match)
        y = self.m2_match

        coords, RCcoords, idxs2_coords = [], [], []
    
        # idxs2_coords : indexes of stars in range of x that are RC candidates
        # from the v. m2 CMD

        #generating coordinate pairs (x1, y1) on the CMD
        for i in range(len(x)): 
            coord_pair = [x[i], y[i]]
            coords.append(coord_pair)

        #finding stars inside selected RC bar

        for i in range(len(coords)): 
            if (coords[i][1] > self.slope2 * coords[i][0] + self.b_min2 and  
               coords[i][1] < self.slope2 * coords[i][0] + self.b_max2): 
                rc_pair = [coords[i][0], coords[i][1]]
                RCcoords.append(rc_pair)
                idxs2_coords.append(i)

        
        print("There are " + str(len(RCcoords)) + " RC candidates from the (m1-m2) vs. (m2) catalog")

        # Second finding RC candidates from the (m1 - m2 v. m1) CMD

        
        y2 = self.m1_match

        coords2, RCcoords2, idxs1_coords = [], [], []

        # idxs1_cords : indexes of stars in range of x that are RC candidates 
        # from the v. m1 CMD

        for i in range(len(x)): 
            coord_pair = [x[i], y2[i]]
            coords2.append(coord_pair)

        for i in range(len(coords2)): 
            if (coords2[i][1] > self.slope1 * coords2[i][0] + self.b_min1 and 
               coords2[i][1] < self.slope1 * coords2[i][0] + self.b_max1): 
                rc_pair = [coords2[i][0], coords2[i][1]]
                RCcoords2.append(rc_pair)
                idxs1_coords.append(i)

        print("There are " + str(len(RCcoords2)) + " RC candidates from the (m1- m2) vs. (m1) catalog")

        return RCcoords, RCcoords2, idxs1_coords, idxs2_coords

    def ratios_extinction(self, RCcoords, RCcoords2, catalog1_name,
                          catalog2_name): 
    
        RCx, RCy, RCx2, RCy2 = [], [], [], []
        
        for i in range(len(RCcoords)): 
            RCx.append(RCcoords[i][0])
            RCy.append(RCcoords[i][1])
        for i in range(len(RCcoords2)): 
            RCx2.append(RCcoords2[i][0])
            RCy2.append(RCcoords2[i][1])

        fig, axis = plt.subplots(2, 1, figsize = (15, 15))

        axis[0].set_title(catalog1_name + " - " + catalog2_name + 
                          " vs. " + catalog2_name + " | Entries: " +
                          str(len(RCcoords)) + 
                          " | Slope " + str(self.slope2))

        axis[0].plot(RCx, RCy, 'k+', alpha = 1, 
                     label = "RC Candidates from v. " + catalog2_name)
        
        axis[0].set_ylabel(catalog2_name)
        axis[0].set_xlabel(catalog1_name + " - " + catalog2_name)


        axis[0].axline(xy1 = (0, self.b_min2), slope = self.slope2, color = 'r')
        axis[0].axline(xy1 = (0, self.b_max2), slope = self.slope2, color = 'r')

        # line of best fit
        axis[0].plot(np.unique(RCx), np.poly1d(np.polyfit(RCx, RCy, 1))(np.unique(RCx)))

        axis[0].invert_yaxis()

        ########################################################################

        axis[1].set_title(catalog1_name + " - " + catalog2_name + 
                          " vs. " + catalog1_name + " | Entries: " + 
                          str(len(RCcoords2)) +
                          " | Slope: " + str(self.slope1))

        axis[1].plot(RCx2, RCy2, 'k+', alpha = 1, 
                    label = "RC Candidates from v. " + catalog1_name)

        axis[1].set_ylabel(catalog1_name)
        axis[1].set_xlabel(catalog1_name + " - " + catalog2_name)

        axis[1].axline(xy1 = (0, self.b_min1), slope = self.slope1, color = 'r')
        axis[1].axline(xy1 = (0, self.b_max1), slope = self.slope1, color = 'r')

        # line of best fit 
        axis[1].plot(np.unique(RCx2), np.poly1d(np.polyfit(RCx2, RCy2, 1))(np.unique(RCx2)))

        axis[1].invert_yaxis()

        fig.savefig('/Users/devaldeliwala/research/jwst_extinction/img/cmd/f115w_212n_RCbar.png')
