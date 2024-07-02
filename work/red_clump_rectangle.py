from catalog_helper_functions import * 
from red_clump_riemann import Optimize

from astropy.modeling import models, fitting

from scipy.stats import linregress, norm, tstd
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

class Analysis: 

    def __init__(self, catalog1, catalog2, 
               catalog1name, catalog2name, catalogyname, 
               region, xlim, ylim, 
               n, image_path, verbose = False): 

        self.catalog1 = catalog1
        self.catalog2 = catalog2 
        self.catalog1name = catalog1name
        self.catalog2name = catalog2name
        self.catalogyname = catalogyname
        self.xlim = xlim
        self.ylim = ylim
        self.n = n 
        self.image_path = image_path
        self.verbose = verbose

    def cutoff(self, show_plot = True): 

        x, y, = self.xlim[0], self.ylim[0]
        width = self.xlim[1] - self.xlim[0]
        height = self.ylim[1] - self.ylim[0]

        if show_plot: 

            fig, axis = plt.subplots(1, 1, figsize = (20, 10))
            plt.gca().invert_yaxis() 

            check = False
            if self.catalogyname == self.catalog1name: 
                plt.scatter(np.subtract(self.catalog1, self.catalog2), self.catalog1, 
                            c = 'k', s = 0.05
                ) 

                plt.ylabel(f"{self.catalog1name}")
                check = True

            if self.catalogyname == self.catalog2name: 
                plt.scatter(np.subtract(self.catalog1, self.catalog2), self.catalog2, 
                            c = 'k', s = 0.05
                ) 

                plt.ylabel(f"{self.catalog2name}")
                check = True

            plt.xlabel(f"{self.catalog1name} - {self.catalog2name}")

            if not check: 
                raise Exception(f"\n\
                                catalogyname must either be equal to catalog1name or catalog2name \n\
                                catalog1name: {catalog1name}, catalog2name: {catalog2name},\n\
                                but catalogyname = {catalogyname}"
                )

            axis.add_patch(Rectangle((x, y), width, height, 
                           facecolor = (0, 0, 0, 0), lw = 2, 
                                          ec = (1, 0, 0, 1)
                           )
            )

            filename = f"{self.catalog1name}-{self.catalog2name}-vs{self.catalogyname}_cutoff"

            my_file = Path(f"{self.image_path}{filename}.png")

            if not my_file.is_file():
                plt.savefig(f"{self.image_path}{filename}.png")

            plt.close() 

        return x, y, width, height

    def generate_bin(self, bin_x_range, verbose = True):

        x, y, width, height = self.cutoff(show_plot = True)
        dx = bin_x_range[1] - bin_x_range[0]

        current_x = bin_x_range[0]

        segment = Rectangle((current_x, y), dx, height, 
                            facecolor = (0,0,0,0), lw = 2, 
                            ec = (1,0,0,1))

        bins = np.array(([current_x, current_x + dx], [y, y + height]))

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

        # running the optimized curve fitting algorithm to determine the optimal fit parameters 
        # for the provided data
        if self.catalogyname == self.catalog2name: 

            y = np.array(starlist[starlist.columns[1]][0])

            optimal_error, optimal_bin, optimal_amplitude, optimal_mean, optimal_std = Optimize(y, 
                                                                                                starlist.columns[1], 
                                                                                                verbose = True
                                                                                        ).optimize_bin()


        if self.catalogyname == self.catalog1name: 

            y = np.array(starlist[starlist.columns[0]][0])

            optimal_error, optimal_bin, optimal_amplitude, optimal_mean, optimal_std = Optimize(y, 
                                                                                                starlist.columns[0], 
                                                                                                verbose = True
                                                                                        ).optimize_bin()

        num_stars = len(y)

        return optimal_error, optimal_bin, optimal_amplitude, optimal_mean, optimal_std, num_stars

    def analysis(self, show_hists=False): 

        trial_dx = (self.xlim[1] - self.xlim[0]) / self.n 
        current_x = self.xlim[0]

        succesful_bin_parameters = []
        bins = []

        # performs the optimized curve fitting algorithm for every n bin
        # in the range of the RC bar
        while current_x <= self.xlim[1]: 

            status = False
            count = 0
            y = 0

            while not status: 

                try: 

                    bin_x_range = [current_x, current_x + trial_dx + 0.1 * count]

                    error, num_bin, amplitude, mean, std, y = self.optimize_bin_fitting(bin_x_range)

                    succesful_bin_parameters.append([error, num_bin, amplitude, mean, std])
    
                    status = True

                    current_x = current_x + trial_dx + 0.1 * count
                    current_bin = [bin_x_range[0], bin_x_range[1]]

                    print(f"Bin width: {current_bin[1]-current_bin[0]}")

                    bins.append(current_bin)

                except ValueError as e: 

                    if str(e) == 'zero-size array to reduction operation minimum which has no identity':
                        bin_x_range = [current_x, current_x + trial_dx + count * 0.1]
                        count += 1

                if count > 7: 
                    raise Exception("\n\
                                     optimize_bin() never found a suitable fitting.\n\
                                     Try altering `x_range`. Perhaps your bin is too small \n\
                                     or there are too few stars."
                    )

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
                plt.xlabel(f"{self.catalog1name} - {self.catalog2name}", fontsize = 15)
                plt.ylabel(f"{self.catalog1name}", fontsize = 15)

            if self.catalogyname == self.catalog2name: 

                plt.scatter(np.subtract(self.catalog1, self.catalog2), 
                            self.catalog2, c = 'k', s = 0.1, alpha = 0.3)
                plt.xlabel(f"{self.catalog1name} - {self.catalog2name}", fontsize = 15)
                plt.ylabel(f"{self.catalog2name}", fontsize = 15)

            plt.plot(midpoints, [inter + i * slope for i in midpoints], 'r-', label = 'linear fit')
            plt.scatter(midpoints, means, c = 'cyan', s = 20, marker = 'x', label = 'means')

            errors = [error * 50 for error in errors]
            plt.errorbar(midpoints, means, yerr=errors, color="cyan", capsize=2, capthick=1, lw = 1, ls = 'none')

            plt.legend()
            plt.gca().invert_yaxis()
            plt.title(f"Fitted Slope: {slope.round(3)} ± {d_slope.round(3)}.    Error Bars scaled 50x. ", fontsize = 15)

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

        axis[1].set_title(f"The Optimal Bin: {optimal_n}", fontsize = 20)
        filename = f"{self.catalog1name}-{self.catalog2name}-vs{self.catalogyname}_{ns}"

        plt.savefig(f"{self.image_path}{filename}.png")

        print("\n\n")
        print(f"[OCF Result]: Provided {ns},")
        print(f"The Optimal # of Segments: {optimal_n}")
        print("\n\n")

        return optimal_n, optimal_slope_error

    def run_optimal_bin(self, ns, show_hists = False, perform_plot = False, perform_residuals = False):

        optimal_n, optimal_slope_error = self.n_analysis(ns) 
        self.n = optimal_n 

        print(f"\n")
        print(f"[OCF]: Finally re-running OCF for the Optimal # of Segments: {self.n}")
        print(f"\n")

        if perform_plot: 
            self.plot(show_hists = show_hists)
        if perform_residuals:
            self.residuals()

        slope, inter, d_slope, d_inter = self.slope()

        return slope, d_slope


class Run_Rectangle: 

    def __init__(self, catalog, catalog1name, catalog2name, catalogyname, 
                 region1, region2, regiony, xlim, ylim, n, image_path, 
                 show_hists, catalog1zp, catalog2zp):

        self.catalog = catalog
        self.catalog1name = catalog1name
        self.catalog2name = catalog2name
        self.catalogyname = catalogyname
        self.region1 = region1
        self.region2 = region2
        self.regiony = regiony
        self.xlim = xlim
        self.ylim = ylim
        self.n = n
        self.image_path = image_path
        self.show_hists = show_hists
        self.catalog1zp = catalog1zp
        self.catalog2zp = catalog2zp

    def run(self, ns, perform_plot = True, perform_residuals = True): 

        catalog1, catalog2, *errors = get_matches(self.catalog, self.catalog1name, 
                                                  self.region1, self.catalog2name, self.region2
                                      )

        if self.catalog1zp: 
            catalog1 += self.catalog1zp 
        if self.catalog2zp: 
            catalog2 += self.catalog2zp

        class_ = Analysis(catalog1, catalog2, 
                          self.catalog1name, self.catalog2name, self.catalogyname, self.region1,
                          self.xlim, self.ylim, n = self.n, image_path = self.image_path, 
                          verbose = True
        )

        slope, d_slope = 0, 0 

        print("")
        print(f"{self.region1} {self.catalog1name} -  {self.region2} {self.catalog2name} vs. {self.regiony} {self.catalogyname}")
        print("")

        if self.show_hists: 
            slope, d_slope = class_.run_optimal_bin(ns, show_hists = True, 
                                                    perform_plot = perform_plot, perform_residuals = perform_residuals)
        else: 
            slope, d_slope = class_.run_optimal_bin(ns, 
                                                    perform_plot = perform_plot, perform_residuals = perform_residuals)

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

            print(f"=================================")
            print(f"Running Population Segment {i+1} of {n}")
            print(f"=================================")

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









