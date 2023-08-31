import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import image
import numpy as np
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
from scipy.stats import gaussian_kde, kde

def color_mag_diagram_rcbar(mag1, mag2, mag_y, mag1_name, mag2_name, mag_y_name,
                      point1, point2, point3, point4, nbins):

    """
    Generates color-magnitude diagram (CMD) between two catalogs

    mag1 - mag2 v. mag_y if y_axis_m1 = False
    mag1 - mag2 v. mag1 if y_axis_m1 = True

    Parameters: 
    -----------

    mag1        : Array
        magnitude values of first filter, generally the one with the smaller
        wavelength
    
    mag2        : Array
        magnitude values of the second filter

    mag_y       : Array
        magnitude values of the filter going on the y_axis 

    mag1_name   : String
        name of the first filter

    mag2_name   : String
        name of the second filter

    mag_y_name  : String
        name of y axis filter

    point1, point2, point3, point4 = [ , ] array
        points that define the lines for RC Bar cutoff 
        point1 & point 2 define one line, point3 & point4 define the other

    nbins       : Int
        # of bins for color mesh of CMD

    """

    arr_diff = np.subtract(mag1, mag2) 

    xy = np.vstack([arr_diff, mag_y])
    z = gaussian_kde(xy)(xy)

    # Calculating line equations for RC bar cutoff
    
    slope1 = (point2[1] - point1[1])/(point2[0] - point1[0])
    slope2 = (point4[1] - point3[1])/(point4[0] - point3[0])

    b1 = point2[1] - slope1 * point2[0]
    b2 = point4[1] - slope2 * point4[0]

    # Plotting CMDs w/ RC Cutoff

    fig, axis = plt.subplots(2, 1, figsize = (20, 20))

    axis[0].scatter(arr_diff, mag_y, c = z, s = 1)
    axis[0].invert_yaxis()
    axis[0].set_xlabel(mag1_name + " - " + mag2_name)
    axis[0].set_ylabel(mag_y_name)
    axis[0].set_title(mag1_name + " - " + mag2_name + " vs. " + mag_y_name)

    axis[0].axline((point1[0], point1[1]), (point2[0], point2[1]), color = 'r')
    axis[0].axline((point3[0], point3[1]), (point4[0], point4[1]), color = 'r')

    k = kde.gaussian_kde([arr_diff, mag_y])
    xi, yi = np.mgrid[min(arr_diff):max(arr_diff):nbins*1j,
                      min(mag_y):max(mag_y):nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    
    print(f"generating pcolormesh for {mag1_name} - {mag2_name} vs. {mag_y_name}")
    print(f"the slope of the RC bars in the cmd {mag1_name} - {mag2_name} vs. {mag_y_name}: {slope1}")
    print("---------------------------------------------------------------------------")

    axis[1].pcolormesh(xi, yi, zi.reshape(xi.shape), shading = 'auto')
    axis[1].set_xlabel(mag1_name + " - " + mag2_name)
    axis[1].set_ylabel(mag_y_name)
    axis[1].set_title(mag1_name + " - " + mag2_name + " vs. " + mag_y_name)
    axis[1].invert_yaxis()

    axis[1].axline((point1[0], point1[1]), (point2[0], point2[1]), color = 'r')
    axis[1].axline((point3[0], point3[1]), (point4[0], point4[1]), color = 'r')

    plt.savefig(f"/Users/devaldeliwala/research/jwst_extinction/media/cmd/cmd_RCcutoff_{mag1_name}_{mag2_name}_{mag_y_name}.png")

   
    magdiffRC, magyRC = [], []
    coords = []

    for coord_num in range(len(arr_diff)):
        coord_pair = [arr_diff[coord_num], mag_y[coord_num]]
        coords.append(coord_pair)

    if b1 > b2: 
        for i in range(len(coords)):
            if coords[i][1] < slope1*(coords[i][0]) + b1 and coords[i][1] > slope2*(coords[i][0]) + b2:
                rc_pair = [coords[i][0], coords[i][1]]
                magdiffRC.append(rc_pair[0])
                magyRC.append(rc_pair[1])
    else: 
        for i in range(len(coords)): 
            if coords[i][1] < slope2*(coords[i][0]) + b2 and coords[i][1] > slope1*(coords[i][0]) + b1:
                rc_pair = [coords[i][0], coords[i][1]]
                magdiffRC.append(rc_pair[0])
                magyRC.append(rc_pair[1])

    fig, axis = plt.subplots(1, 1, figsize = (20, 10))

    plt.plot(magdiffRC, magyRC, 'k+') 
    plt.axline((point1[0], point1[1]), (point2[0], point2[1]), color = 'r')
    plt.axline((point3[0], point3[1]), (point4[0], point4[1]), color = 'r')
    plt.gca().invert_yaxis()

    plt.plot(np.unique(magdiffRC), np.poly1d(np.polyfit(magdiffRC, magyRC, 1))(np.unique(magdiffRC)))
    
    b, m = np.polyfit(magdiffRC, magyRC, 1)
    entries = len(magdiffRC)

    plt.title(f"{mag1_name} - {mag2_name} vs. {mag_y_name} RC stars | Best Fit Slope: {b} | Entries: {entries} | RC Cutoff Slope: {slope1} ")
    plt.xlabel(f"{mag1_name} - {mag2_name}")
    plt.ylabel(f"{mag_y_name}")

    plt.savefig(f"/Users/devaldeliwala/research/jwst_extinction/media/cmd/RC_cutoff_{mag1_name}_{mag2_name}_{mag_y_name}")


def unsharp_mask(mag1, mag2, magy, mag1err, mag2err, magyerr,
                 mag1_filt, mag2_filt, magy_filt,
                 magerr_lim_max = 1.5,
                 mask_width = 0.3,
                 binsize_mag = 0.1,
                 binsize_clr = 0.1,
                 fig_dimensions = 'default',
                 hess_extent = None,
                 fig_path = "/Users/devaldeliwala/research/jwst_extinction/media/unsharp_mask/",
                 data_path = "/Users/devaldeliwala/research/jwst_extinction/media/unsharp_mask/",
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
        
        print('###########################################################')
        print(f"Starting Unsharp Masking Algorithm on {mag1_filt} - {mag2_filt} vs. {magy_filt}")
        print('###########################################################')
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
