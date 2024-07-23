from catalog_helper_functions import * 

from astropy.modeling import models, fitting

from scipy.stats import linregress, norm, tstd, ks_2samp
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter


from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from pathlib import Path

import math 
import numpy as np 
import pandas as pd

import os


class Optimize: 

    """ 
    This class takes in a set of data and determines the optimal number
    of bins to define the data such that, when performing a compound 
    astropy.modeling.Linear1D + astropy.modeling.Gaussian1D fit, the error
    on the mean of the Gaussian is minimized. 

    It then returns the compound fit parameters that led to the optimal fitting. 

    Note this class is automatically ran by Analysis(...).optimize_bin_fitting(self)

    __init__ Parameters: 
    --------------------

    data    : array-like 
            contains the data you wish to perform optimized curve fitting on.

    name    : str
            name of the data for display use.

    verbose : bool 
            if you wish to print out the optimal bitting parameters and other information
            while the code is running


    Methods: 
    --------

    def determine_std(self): 

        Automatically run by def optimize_bin(self):
        determines the optimal standard deviation for the final compound fitting 
        algorithm to be able to compare to in order to determine if the outputted fitting 
        looks good. 

    def optimize_bin(self): 

        Actually runs the optimized curve fitting algorithm on the given `data`. Iterates 
        through possible compound fits for data defined by 8 through 20 subbins. Returns the 
        optimal number of bins, and the rest of the data outputted from the optimized fit. 

    """

    def __init__(self, data, name, verbose): 

        self.data = data
        self.name = name
        self.verbose = verbose

    def determine_std(self): 

        stds, means = [], []
        mu, std = norm.fit(self.data)  

        # iterates the compound fitting for 8 bins through 20. 
        for i in range(8, 20): 
            amplitude_works = False
            trial_amplitude = 10
            count = 0

            while not amplitude_works: 

                bin_heights, bin_borders = np.histogram(self.data, bins = i)
                bin_widths = np.diff(bin_borders)
                bin_centers = bin_borders[:-1] + bin_widths / 2

                # define the compound fit
                gaussian = models.Gaussian1D(trial_amplitude, mu, std)
                linear = models.Linear1D(1, 1)
                compound = gaussian + linear

                fit = fitting.LevMarLSQFitter() 
                result = fit(compound, bin_centers, bin_heights)

                fitted_mean = result.mean_0.value
                fitted_std = result.stddev_0.value
                fitted_amplitude = result.amplitude_0.value

                # determine if the fit worked
                if (fitted_std > 0.1 and fitted_std < 1 and 
                    fitted_amplitude < len(self.data) and fitted_amplitude > 0): 
              
                    amplitude_works = True
                    stds.append(fitted_std)
                    means.append(fitted_mean)

                # couldn't make the conde consistenly work with a simple 
                # trial_amplitude += len(self.data) / some_constant
                # I will fix this issue later. 

                trial = True 
                if len(self.data) > 2000: 
                    trial_amplitude += 200
                    trial = False
                if len(self.data) < 2000 and trial:
                    trial_amplitude += 20

                count += 1

                if count > 15: 
                    amplitude_works = True

        # retrieve the mean std from all successful fittings
        if len(means) > 0 and len(stds) > 0: 
            mean_mean = np.mean(means)
            mean_std = np.mean(stds)

            return mean_std, mean_mean

        else: 

            return 0, 0 

    def optimize_bin(self): 

        mu, std = norm.fit(self.data) 

        # initialize arrays to store succesful fit parameters
        bins, errors, amplitudes, means, stds, slopes, inters = [], [], [], [], [], [], []

        mean_std, mean_mean = self.determine_std()

        # if a fit outputs a standard deviation above the mean + 0.1, it is likely unsuccesful
        # the threshold of 0.1 was determined myself through many iterations
        maximum_allowed_std = mean_std + 0.1
        allowed_mean_range = [mean_mean - 0.1, mean_mean + 0.1]

        for i in range(8, 20): 
            
            amplitude_works = False
            current_amplitude = 10
            count = 0

            while not amplitude_works: 

                bin_heights, bin_borders = np.histogram(self.data, bins = i)
                bin_widths = np.diff(bin_borders)
                bin_centers = bin_borders[:-1] + bin_widths / 2

                gaussian = models.Gaussian1D(current_amplitude, mu, std)
                linear = models.Linear1D(1, 1)
                compound = gaussian + linear

                fit = fitting.LevMarLSQFitter() 
                result = fit(compound, bin_centers, bin_heights)

                fitted_mean = result.mean_0.value
                fitted_std = result.stddev_0.value
                fitted_amplitude = result.amplitude_0.value
                fitted_slope = result.slope_1.value 
                fitted_inter = result.intercept_1.value

                if (fitted_std > 0.1 and fitted_std < maximum_allowed_std and 
                   fitted_amplitude < len(self.data) and fitted_amplitude > 0 and  
                   fitted_mean >= allowed_mean_range[0] and fitted_mean < allowed_mean_range[1]): 

                    filtered_data = self.data[(self.data >= fitted_mean - 3 * fitted_std) & (self.data <= fitted_mean + 3 * fitted_std)]
              
                    amplitude_works = True
                    error = fitted_std / math.sqrt(len(filtered_data))

                    errors.append(error)
                    bins.append(i)
                    amplitudes.append(fitted_amplitude)
                    means.append(fitted_mean)
                    stds.append(fitted_std)
                    slopes.append(fitted_slope)
                    inters.append(fitted_inter)

                # couldn't make the code consistenly work with a simple 
                # current_amplitude += len(self.data) / some_constant
                # I will fix this issue later. 

                trial = True 
                if len(self.data) > 2000: 
                    current_amplitude += 200
                    trial = False
                if len(self.data) < 2000 and trial:
                    current_amplitude += 20

                count += 1

                if count > 15: 
                    amplitude_works = True

        errors = np.array(errors)
        bins = np.array(bins)
        amplitudes = np.array(amplitudes)
        means = np.array(means)
        stds = np.array(stds)
        slopes = np.array(slopes)
        inters = np.array(inters)

        # the mean can't be greater than the max of the data, remove the parameters 
        # where this occurs
        while means[np.where(errors == errors.min())] > self.data.max(): 
            errors = np.delete(errors, np.where(errors == errors.min()))
            means = np.delete(means, np.where(errors == errors.min()))

        optimized_index = np.where((errors == errors.min()))
        optimized_index = optimized_index[0][0]

        optimized_error = errors[optimized_index].round(3)
        optimized_bin = bins[optimized_index].round(3)
        optimized_amplitude = amplitudes[optimized_index].round(3)
        optimized_mean = means[optimized_index].round(3)
        optimized_std = stds[optimized_index].round(3)
        optimized_slope = slopes[optimized_index].round(3)
        optimized_inter = inters[optimized_index].round(3)


        if self.verbose: 
            print("\n")
            print(f"{self.name}")
            print(f"Accepted Bin #s: {bins}")
            print(" {:>13} | {:>8} | {:>11} | {:>8} | {:>8} ".format("Optimal Bin #", 
                                                             "Error", 
                                                             "Amplitude", 
                                                             "Mean", 
                                                             "Stddev"))
            print(" {:>13} | {:>8} | {:>11} | {:>8} | {:>8}".format(optimized_bin, 
                                                             optimized_error, 
                                                             optimized_amplitude, 
                                                             optimized_mean, 
                                                             optimized_std))
            print("\n")           

        return optimized_error, optimized_bin, optimized_amplitude, optimized_mean, optimized_std, optimized_slope, optimized_inter


class Analysis: 

    """
    This class is for determining the slope of the RC bar from a CMD defined by 
    the input parameters catalog1 - catalog2 vs. catalog1 or catalog2. 

    The catalog on the y-axis is the catalog whose name (catalog1name or catalog2name) 
    is equivalent to catalogyname. 

    Apart from the catalogs to define the CMD, a few other parameters are required, 
    specifically to cutoff the region of the RC bar from the entire CMD. However, these 
    parameters do not have to be accurate, only rough guidelines. 

    The main parameter is `n`, the rough number of segments you wish to separate the RC 
    bar and perform optimized curve fitting on individually. The mean for each segment is 
    then plotted against the midpoint for each bin to determine the slope. 

    __init__() Parameters: 
    ----------------------

    catalog1, 
    catalog2        : array-like
                    
                    stores mag information for catalogs of two wavelengths. 
                    the CMD is always generated for catalog1 - catalog2, so by convention catalog1 
                    should be of lower wavelength than catalog2.

    catalog1name, 
    catalog2name, 
    catalogyname    : str
                    
                    for plotting and filename purposes. the y-axis catalog of the generated CMD 
                    will be whichever catalog name is equal to catalogyname.

    region          : str
                    
                    the name of the region for stars. mainly for filename purposes. 

    parallel_cutoff1, 
    parallel_cutoff2: [(float, float), (float, float)]
                    
                    defines two (x, y) points on the CMD that together, define a line that defines either 
                    the lower or upper cutoff of the RC bar. the first index should be the point of 
                    maller x. 

                    The lines defined from both parallel_cutoff1 and parallel_cutoff2 must be parallel.

    x_range         : [float, float]
                    
                    defines the x_range of the entire RC bar

    n               : int
                    
                    the rough number of segments you wish to peform optimized curve fitting on 
                    to determine the slope of the Rc

    image_path      : str
                    
                    the upper directory you wish to store all generated plots in 

    verbose         : bool 
                    
                    if you wish to see extra info while the code is running


    Methods: 
    --------

    def cutoffs(self, verbose = True): 
        defines the cutoffs from the given input parameters and outputs 
        a plot showcasing the cutoffs you provided. 

        It expands the provided parallel cutoffs by 3x in order to allow more stars. 

        These cutoffs only need to be very rough, but in general have to adhere to 
        the following standard: 

            - they should squeeze the RC bar very tightly. Since the software accepts 
              stars within 3x the range of the parallel cutoffs, you have to ensure 
              that even after this 3x increase, the new expanded cutoffs do not extend
              into the main sequence (that would skew the results). The best way to ensure 
              this is to make the parallel cutoffs hug the RC bar fairly tightly. 

            - they can't be too tight. if even after 3x increase the data does not resemble 
              a gaussian and is still primarily in the RC cluster then the fitting will fail. 

    def generate_bin(self, bin_x_range, show_plot = True, verbose = True): 
        generates a bin based on bin_x_range that cuts off data to use in Optimize() class

    def extract_stars(self, bin_x_range, verbose = True): 
        extracts the stars from the bin cutoff generated by generate_bin(...)

    def optimized_bin_fitting(self, bin_x_range): 
        implements the optimized curve fitting algorithm from the Optimize() class on the stars 
        outputted by extract_stars(...)

    def analysis(self, show_hists = False):
        iterates through generate_bin --> optimized_bin_fitting across the entire RC bar and 
        stores all the succesful fittings for each bin generated in an array.  

    def plot(self, show_hists = False)

        displays the result of the optimized curve fitting on a 3D plot. 

        if show_hists is true, it generates a 3D plot with all the optimized fits with their 
        respective histograms underneath the curve

        otherwise (default), it just shows the curve without the bins underneath

    def slope(self): 
        plots the color-magnitude-diagram and the RC best-fit line determined by the optimized
        curve fitting algorithm. It also plots all the errors for each segment generated. 

        displays the determined slope and its error as the title. 

    def residuals(self): 
        generates a plot of residuals based on the RC best-fit line. 
        also generates a heat map of the same plot. 

    def n_analysis(self, ns): 

        ns: [int, int, int, ...]

        Runs the above algorithm where n, which determines the number
        of segments you wihs to divide the RC bar into, is instead each index of 
        ns. 

        It finds the optimal number of rough segments such that the error of the slope is
        minimized. 

        It afterwards generates a plot comparing the results of the optimized curve fitting 
        algorithm for every n in ns. 

    def run_optimal_bin(self): 
        runs the algorithm based on the optimal n generated by n_analysis(self, ns) and outputs 
        the final slope with error determined. 


    Note that this class is automatically run by Run(...).run().

    """

    def __init__(self, catalog1, catalog2, 
                 catalog1name, catalog2name, catalogyname, 
                 region, parallel_cutoff1, parallel_cutoff2, 
                 x_range, n, image_path, verbose = False): 

        self.catalog1 = catalog1
        self.catalog2 = catalog2
        self.catalog1name = catalog1name
        self.catalog2name = catalog2name
        self.catalogyname = catalogyname
        self.region = region
        self.parallel_cutoff1 = parallel_cutoff1
        self.parallel_cutoff2 = parallel_cutoff2
        self.x_range = x_range
        self.n = n
        self.image_path = image_path 
        self.verbose = verbose

    def cutoffs(self, show_plot = True): 

        x = np.subtract(self.catalog1, self.catalog2)

        line1 = self.parallel_cutoff1
        line2 = self.parallel_cutoff2

        # determine the slopes of the parallel cutoffs from the input parameters
        slope1 = round((line1[1][1] - line1[0][1]) / (line1[1][0] - line1[0][0]), 3)
        slope2 = round((line2[1][1] - line2[0][1]) / (line2[1][0] - line2[0][0]), 3)

        intercept1 = line1[1][1] - slope1 * line1[1][0]
        intercept2 = line2[1][1] - slope2 * line2[1][0]

        height = abs(intercept2 - intercept1) # height between parallel cutoffs. 
        
        height = 3 * height # increasing height 3x to select more stars outside

        if intercept1 > intercept2: 
            intercept1 += height / 3
            intercept2 -= height / 3
        else: 
            intercept2 += height / 3
            intercept1 -= height / 3 

        if slope1 != slope2:
            raise Exception(f"\n\
                            Slopes of parallel cutoffs must be equal. \n\
                            Slope1 = {slope1} but Slope2 = {slope2} ")

        if show_plot: 

            fig, axis = plt.subplots(1, 1, figsize = (20, 10))
            plt.gca().invert_yaxis()

            check = False
            if self.catalogyname == self.catalog1name: 

                plt.scatter(x, self.catalog1, c = 'k', s = 0.05)
                plt.ylabel(f"{self.catalog1name}", fontsize = 15)

                check = True

            if self.catalogyname == self.catalog2name: 

                plt.scatter(x, self.catalog2, c = 'k', s = 0.05)
                plt.ylabel(f"{self.catalog2name}", fontsize = 15)

                check = True

            if not check: 
                raise Exception(f"\n\
                                catalogyname must either be equal to catalog1name or catalog2name \n\
                                catalog1name: {catalog1name}, catalog2name: {catalog2name},\n\
                                but catalogyname = {catalogyname}"
                )

            
            plt.axline(line1[0], line1[1], c = 'r', label = 'provided rc cutoff')
            plt.axline(line2[0], line2[1], c = 'r')

            plt.axline([0, intercept1], slope = slope1, c = 'aqua', label = 'actual rc cutoff')
            plt.axline([0, intercept2], slope = slope2, c = 'aqua',)

            plt.axvline(x = self.x_range[0], c = 'r', linestyle = ':', label = 'selected x range')
            plt.axvline(x = self.x_range[1], c = 'r', linestyle = ':')

            plt.xlabel(f"{self.catalog1name} - {self.catalog2name}", fontsize = 15)
            plt.legend()

            filename = f"{self.catalog1name}-{self.catalog2name}-vs{self.catalogyname}_cutoff"
            my_file = Path(f"{self.image_path}{filename}.png")

            if not my_file.is_file():
                plt.savefig(f"{self.image_path}{filename}.png")
                
            plt.close() 

        if intercept1 < intercept2: 
            return intercept1, intercept2, slope1, height
        else: 
            return intercept2, intercept1, slope1, height

    def generate_bin(self, bin_x_range, verbose = True): 

        intercept1, intercept2, slope, height = self.cutoffs(show_plot = True)
        dx = bin_x_range[1] - bin_x_range[0]

        current_x = bin_x_range[0]
        yi = slope * current_x + intercept1
        yf = slope * current_x + intercept2

        segment = Rectangle((current_x, yi), dx, height, 
                            facecolor = (0,0,0,0), lw = 2, 
                            ec = (1,0,0,1)
        )

        bins = np.array(([current_x, current_x + dx], [yi, yf]))

        return bins, segment

    def extract_stars(self, bin_x_range, verbose = True): 

        bins, segment = self.generate_bin(bin_x_range, verbose = verbose)

        x = np.subtract(self.catalog1, self.catalog2)
        idxs = []

        catalog1_mag, catalog2_mag = [], []

        if self.catalogyname == self.catalog1name: 
            y = self.catalog1
        if self.catalogyname == self.catalog2name:
            y = self.catalog2

        # extracting stars within bin
        good = np.where( (x >= bins[0][0]) & (x < bins[0][1]) & 
                         (y >= bins[1][0]) & (y < bins[1][1])
        )
        
        idxs.append(good)
        catalog1_mag.append(self.catalog1[good])
        catalog2_mag.append(self.catalog2[good])

        df1 = pd.DataFrame({f'{self.catalog1name}': catalog1_mag})
        df2 = pd.DataFrame({f'{self.catalog2name}': catalog2_mag})

        starlist = pd.concat([df1, df2], axis = 1)

        return starlist, idxs

    def optimize_bin_fitting(self, bin_x_range): 

        starlist, idxs = self.extract_stars(bin_x_range)
        y = []
        x = np.subtract(starlist[starlist.columns[0]][0], starlist[starlist.columns[1]][0])

        # running the optimized curve fitting algorithm to determine the optimal fit parameters 
        # for the provided data
        if self.catalogyname == self.catalog2name: 

            y = np.array(starlist[starlist.columns[1]][0])
            optimal_error, optimal_bin, optimal_amplitude, optimal_mean, optimal_std, optimal_slope, optimal_inter = Optimize(y, 
                                                                                                starlist.columns[1], 
                                                                                                verbose = True
                                                                                        ).optimize_bin()
        if self.catalogyname == self.catalog1name: 

            y = np.array(starlist[starlist.columns[0]][0])
            optimal_error, optimal_bin, optimal_amplitude, optimal_mean, optimal_std, optimal_slope, optimal_inter = Optimize(y, 
                                                                                                starlist.columns[0], 
                                                                                                verbose = True
                                                                                        ).optimize_bin()

        num_stars = len(y)

        return optimal_error, optimal_bin, optimal_amplitude, optimal_mean, optimal_std, optimal_slope, optimal_inter, num_stars, y, x 

    def analysis(self, show_hists=False): 

        trial_dx = (self.x_range[1] - self.x_range[0]) / self.n 
        current_x = self.x_range[0]

        succesful_bin_parameters = []
        bins = []
        rc_mags = []
        rc_x = []

        # performs the optimized curve fitting algorithm for every n bin
        # in the range of the RC bar
        while current_x <= self.x_range[1]: 

            status = False
            count = 0
            y = 0

            while not status: 

                try: 

                    bin_x_range = [current_x, current_x + trial_dx + 0.1 * count]

                    error, num_bin, amplitude, mean, std, slope, inter, y, y2, x2 = self.optimize_bin_fitting(bin_x_range)
                    succesful_bin_parameters.append([error, num_bin, amplitude, mean, std, slope, inter])
    
                    status = True

                    current_x = current_x + trial_dx + 0.1 * count
                    current_bin = [bin_x_range[0], bin_x_range[1]]

                    print(f"Bin width: {current_bin[1]-current_bin[0]}")

                    bins.append(current_bin)

                    y2 = np.array([val for val in y2 if isinstance(val, float)])
                    x2 = np.array([val for val in x2 if isinstance(val, float)])
                    rc_mags.append(y2)
                    rc_x.append(x2)

                except ValueError as e: 

                    if str(e) == 'zero-size array to reduction operation minimum which has no identity':
                        bin_x_range = [current_x, current_x + trial_dx + count * 0.1]
                        count += 1

                if count > 7: 
                    raise Exception("\n\
                                     optimize_bin() never found a suitable fitting. \n\
                                     Try altering `x_range` or squeezing the `parallel_cutoff`s tighter to the RC bar \n\
                                     to ensure stars from main sequence aren't included. Or maybe there are just too few stars? "
                          )

        self.red_clump_mags = rc_mags
        self.red_clump_x = rc_x

        return succesful_bin_parameters, bins, y

    def plot(self, show_hists = False): 

        succesful_bin_parameters, bins, y = self.analysis()

        ax = plt.figure(figsize = (10, 10)).add_subplot(projection = '3d')
        ax.view_init(elev = 35, azim = -45, roll = 0)

        starlist_full = []

        for i in bins: 
            starlist, idxs = self.extract_stars(i, verbose = False)
            starlist_full.append([starlist[starlist.columns[0]][0], starlist[starlist.columns[1]][0]])

        colors = plt.cm.jet(np.linspace(0, 1, len(starlist_full)))

        ax.set_xlabel(f"{self.catalog1name} - {self.catalog2name}")
        ax.set_ylabel(f"{self.catalogyname}")
        ax.set_zlabel("Frequency")
        ax.zaxis.labelpad=-0.01

        for i in range(len(starlist_full)): 

            bin_heights, bin_borders, bin_widths, bin_centers = [], [], [], []

            optimal_error   = succesful_bin_parameters[i][0]
            optimal_bin     = succesful_bin_parameters[i][1]
            optimal_amp     = succesful_bin_parameters[i][2]
            optimal_mean    = succesful_bin_parameters[i][3]
            optimal_std     = succesful_bin_parameters[i][4]

            # plotting the RC cluster on the x-y plane
            if self.catalogyname == self.catalog2name: 

                ax.scatter(np.subtract(starlist_full[i][0], starlist_full[i][1]), 
                           starlist_full[i][1], color = colors[i], 
                           zs = 0, zdir = 'z', s = 0.3, label = 'Red Clump Cluster')

                bin_heights, bin_borders = np.histogram(starlist_full[i][1], bins=optimal_bin)
                bin_widths = np.diff(bin_borders)
                bin_centers = bin_borders[:-1] + bin_widths / 2

            if self.catalogyname == self.catalog1name: 

                ax.scatter(np.subtract(starlist_full[i][0], starlist_full[i][1]), 
                           starlist_full[i][0], color = colors[i], 
                           zs = 0, zdir = 'z', s = 0.3, label = 'Red Clump Cluster')

                bin_heights, bin_borders = np.histogram(starlist_full[i][0], bins=optimal_bin)
                bin_widths = np.diff(bin_borders)
                bin_centers = bin_borders[:-1] + bin_widths / 2

            min_x = np.min(np.subtract(starlist_full[i][0], starlist_full[i][1]))

            gaussian = models.Gaussian1D(optimal_amp, optimal_mean, optimal_std)
            linear = models.Linear1D(100, 100)
            compound = gaussian + linear

            fit = fitting.LevMarLSQFitter()
            result = fit(compound, bin_centers, bin_heights)

            fit_interval = np.linspace(bin_borders[0], bin_borders[-1], 10000)

            # displays the succesful fit curve on the z-axis
            ax.plot(fit_interval, result(fit_interval), 
                    label = 'fit', color = colors[i], zs = min_x, zdir = 'x')

            if show_hists: 
                ax.bar(bin_centers, bin_heights, width = bin_widths, label = 'histogram', 
                       zs = min_x, zdir = 'x', ec = (0,0,0, 0.5), facecolor = (0,0,0,0))

        if show_hists: 
            ax.figure.savefig(f"{self.image_path}{self.catalog1name}-{self.catalog2name}-vs{self.catalogyname}_result-hist.png")
        else: 
            ax.figure.savefig(f"{self.image_path}{self.catalog1name}-{self.catalog2name}-vs{self.catalogyname}_result.png")

        return 

    def slope(self, show_plot = True):
        
        succesful_bin_parameters, bins, y = self.analysis()

        means = []
        errors = []
        midpoints = []

        for i in range(len(succesful_bin_parameters)): 
            means.append(succesful_bin_parameters[i][3])
            errors.append(succesful_bin_parameters[i][0])
            midpoints.append(bins[i][0] + (bins[i][1] - bins[i][0]) / 2)

        def linear_func(x, b, a): 
            y = b + a * x
            return y

        # implement a weighted linear fit to determine the slope 
        result, cov = curve_fit(linear_func, midpoints, means, sigma = errors, absolute_sigma = True)

        inter = result[0]
        slope = result[1]
        d_inter = np.sqrt(cov[0][0])

        # determines the error of the slope
        d_slope = np.sqrt(cov[1][1]) 

        if show_plot: 
            fig, axis = plt.subplots(1, 1, figsize = (20, 10))

            if self.catalogyname == self.catalog1name: 

                plt.scatter(np.subtract(self.catalog1, self.catalog2), 
                            self.catalog1, c = 'k', s = 0.1, alpha = 0.3)
                plt.ylabel(f"{self.catalog1name}", fontsize = 15)

            if self.catalogyname == self.catalog2name: 

                plt.scatter(np.subtract(self.catalog1, self.catalog2), 
                            self.catalog2, c = 'k', s = 0.1, alpha = 0.3)
                plt.ylabel(f"{self.catalog2name}", fontsize = 15)

            plt.xlabel(f"{self.catalog1name} - {self.catalog2name}", fontsize = 15)
            plt.plot(midpoints, [inter + i * slope for i in midpoints], 'r-', label = 'linear fit')
            plt.scatter(midpoints, means, c = 'cyan', s = 20, marker = 'x', label = 'means')

            errors = [error * 50 for error in errors]
            plt.errorbar(midpoints, means, yerr=errors, color="cyan", capsize=2, capthick=1, lw = 1, ls = 'none')

            plt.legend()
            plt.gca().invert_yaxis()
            plt.title(f"Fitted Slope: {slope.round(3)} ± {d_slope.round(3)}. Error Bars scaled 50x. ", fontsize = 15)

            error_above, error_below = [], []

            for i in range(len(means)): 
                error_above.append(means[i] + errors[i])
                error_below.append(means[i] - errors[i])

            def poly2d(x, a, b, c): 
                return a + b*x + c*x**2 

            coefs_poly2d_above, pcov = curve_fit(poly2d, midpoints, error_above)
            coefs_poly2d_below, pcov = curve_fit(poly2d, midpoints, error_below)

            x_data = np.linspace(min(midpoints), max(midpoints), 50)
            y_data_above = poly2d(x_data, *coefs_poly2d_above)
            y_data_below = poly2d(x_data, *coefs_poly2d_below)

            # confidence interval 
            plt.fill_between(x_data, y_data_above, y_data_below, color='cyan', alpha=0.08)

            filename = f"{self.catalog1name}-{self.catalog2name}-vs{self.catalogyname}_slope"
            plt.savefig(f"{self.image_path}{filename}.png")
            plt.close()

        print("\n\n")
        print(f"[SLOPE] {self.catalog1name} - {self.catalog2name} vs. {self.catalogyname}")
        print("‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾")
        print(f"# For {self.n} Segments:")
        print(f"# Calculated Red Clump Slope: {slope.round(3)} ± {d_slope.round(3)}")
        print("___________________________________________")
        print("\n\n")

        return slope, inter, d_slope, d_inter

    def residuals(self): 

        succesful_bin_parameters, bins, y = self.analysis()
        slope, intercept, *errors = self.slope(show_plot = False)

        starlist_full = []

        for i in bins: 
            starlist, idxs = self.extract_stars(i, verbose = False)
            starlist_full.append([starlist[starlist.columns[0]][0], starlist[starlist.columns[1]][0]])

        colors = plt.cm.jet(np.linspace(0, 1, len(starlist_full)))

        fig, axis = plt.subplots(2, 1,  figsize = (15, 15))

        def heat(x, y, s, bins=1000):

            heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
            heatmap = gaussian_filter(heatmap, sigma=s)

            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

            return heatmap.T, extent

        x_full, y_full = [], []

        for i in range(len(starlist_full)): 

            x = np.subtract(starlist_full[i][0], starlist_full[i][1])

            predicted_y = []
            actual_y = []
            residuals = []

            if self.catalogyname == self.catalog1name: 
                actual_y = starlist_full[i][0]

            if self.catalogyname == self.catalog2name: 
                actual_y = starlist_full[i][1]

            x_full.append(x)

            for j in x: 
                predicted_y.append(slope * j + intercept)

            predicted_y = np.array(predicted_y)

            # residuals
            for k in range(len(predicted_y)): 
                residuals.append(actual_y[k] - predicted_y[k])

            y_full.append(residuals)

            # plot residuals
            axis[0].scatter(x, residuals, color = colors[i], s = 0.3, label = 'Red Clump Cluster' )
            axis[1].hist(residuals, color = colors[i], histtype='step')


        axis[0].set_xlabel(f"{self.catalog1name} - {self.catalog2name}", fontsize = 15)
        axis[0].set_ylabel(f"Residual", fontsize = 15)

        axis[1].set_xlabel(f"Residual", fontsize = 15)
        axis[1].set_ylabel(f"Frequency", fontsize = 15)

        filename = f"{self.catalog1name}-{self.catalog2name}-vs{self.catalogyname}_residual"
        axis[0].set_title(f"{self.catalog1name} - {self.catalog2name} vs. {self.catalogyname}", fontsize = 15)

        plt.savefig(f"{self.image_path}{filename}.png")

        fig, axis = plt.subplots(1, 1, figsize = (20, 10))

        x_arrays = [np.array(column) for column in x_full]
        x_full = np.concatenate(x_arrays)
        y_arrays = [np.array(column) for column in y_full]
        y_full = np.concatenate(y_arrays)

        # heat map
        img, extent = heat(x_full, y_full, 16)
        axis.imshow(img, extent=extent, origin='lower', cmap=cm.jet)

        plt.xlabel(f"{self.catalog1name} - {self.catalog2name}", fontsize = 15)
        plt.ylabel("Residual", fontsize = 15)
        plt.title(f"{self.catalog1name} - {self.catalog2name} vs. {self.catalogyname}", fontsize = 15)

        plt.savefig(f"{self.image_path}{filename}_heatmap.png")

        return 

    def n_analysis(self, ns): 

        fig, axis = plt.subplots(3, 1, figsize = (30, 30))

        colors = plt.cm.jet(np.linspace(0, 1, len(ns)))

        optimal_n = 0
        optimal_slope_error = 100
        bin_parameters = []

        print("\n")
        print(f"Starting Optimize Curve Fitting Algorithm (OCF)")
        print(f"Provided Possible # of Segments: {ns}")
        print("\n")

        # implements the entire optimized curve fitting algorithm to retrieve possible slopes
        # for every n in ns

        for i in range(len(ns)): 

            means = []
            errors = []
            midpoints = []

            self.n = ns[i] 

            print(f"[OCF]: {self.n} Segments ")
            print("‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾")
            print("\n")

            succesful_bin_parameters, bins, y = self.analysis()
            slope, intercept, d_slope, d_inter = self.slope(show_plot = False)

            if d_slope < optimal_slope_error: 
                optimal_slope_error = d_slope
                optimal_n = ns[i]

                bin_parameters = []
                bin_parameters.append(succesful_bin_parameters)


            for j in range(len(succesful_bin_parameters)): 
                means.append(succesful_bin_parameters[j][3])
                errors.append(succesful_bin_parameters[j][0])
                midpoints.append(bins[j][0] + (bins[j][1] - bins[j][0]) / 2)


            axis[0].scatter(ns[i], slope, color = colors[i], label = f'n = {ns[i]}')
            axis[0].errorbar(ns[i], slope, yerr=d_slope, color=colors[i], capsize=2, capthick=1, lw = 1, ls = 'none')

            axis[0].set_xlabel("$n$", fontsize = 22)
            axis[0].set_ylabel("RC Slope", fontsize = 20)
            axis[0].set_title(f"{self.catalog1name} - {self.catalog2name} vs. {self.catalogyname}", fontsize = 20)

            axis[1].plot(midpoints, means, color = colors[i], label = f'n = {ns[i]}')
            axis[2].plot(midpoints, errors, color = colors[i], label = f'n = {ns[i]}')

            axis[2].set_xlabel(f"{self.catalog1name} - {self.catalog2name}", fontsize = 20)

            axis[1].set_ylabel("Mean", fontsize = 20)
            axis[2].set_ylabel("Error", fontsize = 20)

            axis[2].legend()

        self.bin_parameters = bin_parameters

        filename = f"{self.catalog1name}-{self.catalog2name}-vs{self.catalogyname}_{ns}"

        axis[1].set_title(f"The Optimal Bin: {optimal_n}", fontsize = 20)
        plt.savefig(f"{self.image_path}{filename}.png")

        print("\n\n")
        print(f"[OCF Result]:  Provided {ns},")
        print(f"The Optimal Number of Segments: {optimal_n}")
        print("\n\n")

        return optimal_n, optimal_slope_error

    def run_optimal_bin(self, ns, show_hists = False, perform_plot = False, perform_residuals = False, perform_KS = False):

        optimal_n, optimal_slope_error = self.n_analysis(ns) 
        self.n = optimal_n 

        print(f"\n")
        print(f"[OCF]: Finally re-running OCF for the Optimal # of Segments: {self.n}")
        print(f"\n")

        if perform_plot: 
            self.plot(show_hists = show_hists)
        if perform_residuals:
            self.residuals()
        if perform_KS: 
            self.ks_test()

        slope, inter, d_slope, d_inter = self.slope()

        return slope, d_slope

    def ks_test(self, show_plot = True): 

        rc_mags     = self.red_clump_mags
        rc_x        = self.red_clump_x

        means, amplitudes, stds, slopes, inters = [], [], [], [], []

        synthetic_data = []
        compound_models = []

        for i in self.bin_parameters[0]:
            means.append(i[3])
            amplitudes.append(i[2])
            stds.append(i[4])
            slopes.append(i[5])
            inters.append(i[6])

        for i in range(len(means)): 
            gaussian = models.Gaussian1D(amplitudes[i], means[i], stds[i])
            linear = models.Linear1D(slopes[i], inters[i])
            compound = gaussian + linear

            compound_models.append(compound)

        for i in range(len(compound_models)): 
            y = rc_mags[i]

            # Normalize the PDF to create a CDF
            y_values = np.linspace(np.min(y), np.max(y), 10000)
            pdf_values = compound_models[i](y_values)
            pdf_values /= np.sum(pdf_values)
            cdf_values = np.cumsum(pdf_values)
            cdf_values /= cdf_values[-1]

            # Inverse transform sampling to generate synthetic y-values
            random_values = np.random.rand(len(y))
            synthetic_y = np.interp(random_values, cdf_values, y_values)
            synthetic_data.append(synthetic_y)

        synthetic_data_full = []
        for i in synthetic_data: 
            for j in i: 
                synthetic_data_full.append(j)
        synthetic_data_full = np.array(synthetic_data_full)

        rc_mags_full = []
        for i in rc_mags: 
            for j in i: 
                rc_mags_full.append(j)
        rc_mags_full = np.array(rc_mags_full)

        rc_x_full = []
        for i in rc_x: 
            for j in i: 
                rc_x_full.append(j)
        rc_x_full = np.array(rc_x_full)

        ks_statistic, p_value = ks_2samp(rc_mags_full, synthetic_data_full)

        if show_plot: 
            fig, axis = plt.subplots(2, 1, figsize = (20, 20))

            axis[0].scatter(rc_x_full, rc_mags_full, c = 'k', s = 0.05, label = 'Actual Cluster')
            axis[1].scatter(rc_x_full, synthetic_data_full, c = 'r', s = 0.05, label = 'Synthetic Cluster')

            for i in range(2): 
                axis[i].set_xlabel(f'{self.catalog1name} - {self.catalog2name}', fontsize = 15)
                axis[i].set_ylabel(f'{self.catalogyname}', fontsize = 15)
                axis[i].legend(fontsize = 15)

            axis[1].set_title(f'KS Statistic: {ks_statistic} / P Value: {p_value}', fontsize = 15)
            axis[0].set_title(f'{self.region} {self.catalog1name} - {self.catalog2name} vs. {self.catalogyname}')

            filename = 'KS_test'
            plt.savefig(f"{self.image_path}{filename}.png")

        else: 
            print(f'KS Statistic: {ks_statistic} / P Value: {p_value}')

        return






class Run_Riemann: 

    """
    This class runs the optimized curve fitting algorithm for the optimal number of 
    segments to define an RC bar and outputs the slope with error. 

    It also checks for subpopulations (which do not exist), but if they did it would 
    output a plot of the CMD with colored regions for each population. 

    The parameters are self explanatory, and are equivalent to the parameters for
    Analysis(...), except for the following: 

    catalog: Astropy Table
        matched star catalog for every wavelength.

    region1, region2, regiony: str
        the names of the regions from which catalog1, catalog2, and the y-catalog. 
        purely for plotting purposes. 

    catalog1zp, catalog2zp: float 
        the zeropoints for the catalogs 


    Methods: 
    --------

    def run(self, ns, perform_plot = True, perform_residuals = False): 
        finds the optimal number of segments to divide the RC bar from `ns` and 
        outputs the optimal slope

        is perform_residuals is set to True, it displays the residual plot and heat map 

    def subpopulations(self, n, ns, image_path = False, show_populations = True): 

        divides the RC bar into `n` equal sized segmenets and finds the slope in each segment 
        along with its error. If the difference in slopes between any two segments is less than 
        3*the quad summed errors of both slopes, it separates these two regions on a plot
        to show they are distinct populations, otherwise the outputted plot shows a uniform blue 
        region covering the RC bar. 

        if show_populations is set to False, it doesn't display a plot.

        if image_path is set to True, it chooses an alternative image_path to store the outputted
        plot.

    --------------
    Example Usage: 
    --------------

    fits ='~/.../jwst_init_NRCB.fits'
    catalog = Table.read(fits, format='fits')

                   -catalog1-          -catalog2-           -catalogy-
    script = [[['NRCB1', 'F115W'], ['NRCB1', 'F212N'], ['NRCB1', 'F115W']]]

    catalog1name = script[0][0][1]
    catalog2name= script[0][1][1]
    catalogyname = script[0][2][1]

    region1 = script[0][0][0]
    region2 = script[0][1][0]
    regiony = script[0][2][0]

    parallel_cutoff1 = [(6.3, 21.3), (9, 25.2)]
    parallel_cutoff2 = [(6.3, 22), (9, 25.9)]

    catalog1zp = 25.92
    catalog2zp = 22.12

    x_range = [4.9, 9]

    n = 10
    ns = [10, 11, 12, 13, 14, 15] 

    class_ = Run(
        catalog=catalog,
        catalog1name=catalog1name, 
        catalog2name=catalog2name, 
        catalogyname=catalogyname, 
        region1=region1, 
        region2=region2, 
        regiony=regiony, 
        parallel_cutoff1=parallel_cutoff1, 
        parallel_cutoff2=parallel_cutoff2, 
        x_range=x_range, 
        n=n, 
        image_path=image_path, 
        show_hists=True
        catalog1zp=catalog1zp, 
        catalog2zp=catalog2zp 
    )   

    class_.run(ns = ns)

    """

    def __init__(self, catalog, catalog1name, catalog2name, catalogyname, 
        region1, region2, regiony, parallel_cutoff1, parallel_cutoff2, 
        x_range, n, image_path, show_hists, catalog1zp, catalog2zp):
        
        self.catalog = catalog
        self.catalog1name = catalog1name
        self.catalog2name = catalog2name
        self.catalogyname = catalogyname
        self.region1 = region1
        self.region2 = region2
        self.regiony = regiony
        self.parallel_cutoff1 = parallel_cutoff1
        self.parallel_cutoff2 = parallel_cutoff2
        self.x_range = x_range
        self.n = n
        self.image_path = image_path
        self.show_hists = show_hists
        self.catalog1zp = catalog1zp
        self.catalog2zp = catalog2zp

    def run(self, ns, perform_plot = True, perform_residuals = True, perform_KS = True):
        
        catalog1, catalog2, *errors = get_matches(self.catalog, self.catalog1name, 
                                                  self.region1, self.catalog2name, self.region2
                                      )

        if self.catalog1zp: 
            catalog1 += self.catalog1zp
        if self.catalog2zp: 
            catalog2 += self.catalog2zp

        class_ = Analysis(catalog1, catalog2, 
                          self.catalog1name, self.catalog2name, self.catalogyname, self.region1,
                          self.parallel_cutoff1, self.parallel_cutoff2, self.x_range, 
                          n = self.n, image_path = self.image_path, verbose = True
        )

        slope, d_slope = 0, 0

        print(f"{self.region1} {self.catalog1name} -  {self.region2} {self.catalog2name} vs. {self.regiony} {self.catalogyname}")

        if self.show_hists: 
            slope, d_slope = class_.run_optimal_bin(ns, show_hists = True, 
                                                    perform_plot = perform_plot, 
                                                    perform_residuals = perform_residuals, 
                                                    perform_KS = perform_KS)
        else: 
            slope, d_slope = class_.run_optimal_bin(ns, 
                                                    perform_plot = perform_plot, 
                                                    perform_residuals = perform_residuals, 
                                                    perform_KS = perform_KS)

        return slope, d_slope

    def sub_populations(self, n, ns, image_path = False, show_populations = True):

        dx = (self.x_range[1] - self.x_range[0]) / n 
        start = self.x_range[0]

        x_ranges = []
        image_path_ = image_path

        slopes = []
        errors = []
        slope, d_slope = 0, 0

        for i in range(n): 

            print(f"\n\n")
            print(f"Running Population Segment {i+1} of {n}")
            print(f"\n\n")

            self.x_range = [start, start + dx]
            x_ranges.append(self.x_range)

            self.n = n

            image_path = image_path_
            image_path = image_path + f"{n}_segments/{i+1}_of_{n}/"

            if not os.path.isdir(image_path):
                os.makedirs(image_path)
            self.image_path = image_path

            if not image_path: 
                slope, d_slope = self.run(ns, perform_plot = True)

                slopes.append(slope)
                errors.append(d_slope)

            if image_path: 
                slope, d_slope = self.run(ns, perform_plot = True)

                slopes.append(slope)
                errors.append(d_slope)

            start +=  dx

        def quad_summed_errors(slopes, errors): 

            i = 0
            matches = []

            while i+1 < len(slopes): 
                summed_error = math.sqrt(errors[i]**2 + errors[i+1]**2)
                slope_difference = abs(slopes[i] - slopes[i+1])

                if slope_difference <= 3*summed_error: 
                    matches.append([i, i+1, True])
                else: 
                    matches.append([i, i+1, False])

                i += 1

            return matches

        matches = quad_summed_errors(slopes, errors)

        final_x_range_populations = []
        matched_indices = []
        temp = []

        for i in matches:
            if not temp:
                temp.extend(i[:2])
            else:
                temp.append(i[1])
    
            if not i[2]:
                matched_indices.append(temp)
                temp = []

        if temp:
            matched_indices.append(temp)

        for group in matched_indices:
            start_idx = group[0]
            end_idx = group[-1] # Add 1 to include the last index

            start_value = x_ranges[start_idx][0]
            end_value = x_ranges[end_idx][1]

            final_x_range_populations.append([start_value, end_value])

        if show_populations: 
            fig, axis = plt.subplots(1, 1, figsize = (20, 10))
            plt.gca().invert_yaxis()

            catalog1, catalog2, *errors = get_matches(self.catalog, self.catalog1name, 
                                                  self.region1, self.catalog2name, self.region2
                                          )

            if self.catalog1zp: 
                catalog1 += self.catalog1zp
            if self.catalog2zp: 
                catalog2 += self.catalog2zp

            x = np.subtract(catalog1, catalog2)

            if self.catalogyname == self.catalog1name: 
                plt.scatter(x, catalog1, c = 'k', s = 0.05)
            if self.catalogyname == self.catalog2name:
                plt.scatter(x, catalog2, c = 'k', s = 0.05)

            plt.xlabel(f"{self.catalog1name} - {self.catalog2name}", fontsize = 15)
            plt.ylabel(f"{self.catalogyname}", fontsize = 15)

            colors = plt.cm.jet(np.linspace(0, 1, len(final_x_range_populations)))

            for i in range(len(final_x_range_populations)): 
                segment = final_x_range_populations[i]
                plt.axvspan(segment[0], segment[1], alpha=0.1, color=colors[i])

            filename = f"{n}_segments_population_plot"
            plt.savefig(f"{image_path_}{filename}.png")

        return 
