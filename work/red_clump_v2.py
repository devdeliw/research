from color_magnitude_diagrams import * 
from isochrones import *
from catalog_helper_functions import * 
from scipy.stats import linregress
from matplotlib.patches import Rectangle
import math 
import numpy as np 
import pandas as pd
import seaborn as sns

class Optimize: 

    def __init__(self, data, name, verbose): 

        self.data = data
        self.name = name
        self.verbose = verbose

    def optimize_bin(self): 

        mu, std = norm.fit(self.data) 
        std = tstd(self.data, limits = (mu - 3*std, mu + 3*std))

        bins, errors, amplitudes, means, stds = [], [], [], [], []

        for i in range(8, 20): 
            
            amplitude_works = False
            current_amplitude = 10
            count = 0

            while not amplitude_works: 

                bin_heights, bin_borders = np.histogram(self.data, bins = i)
                bin_widths = np.diff(bin_borders)
                bin_centers = bin_borders[:-1] + bin_widths / 2

                gaussian = models.Gaussian1D(current_amplitude, mu, std)
                linear = models.Linear1D(100, 100)
                compound = gaussian + linear

                fit = fitting.LevMarLSQFitter() 
                result = fit(compound, bin_centers, bin_heights)

                fitted_mean = result.mean_0.value
                fitted_std = result.stddev_0.value
                fitted_amplitude = result.amplitude_0.value

                filtered_data = self.data[(self.data >= fitted_mean - 3 * fitted_std) & (self.data <= fitted_mean + 3 * fitted_std)]

                if fitted_std > 0.1 and fitted_std < 1 and fitted_amplitude < 500 and fitted_amplitude > 0: 
                    
                    amplitude_works = True
                    error = fitted_mean / math.sqrt(len(filtered_data))

                    errors.append(error)
                    bins.append(i)
                    amplitudes.append(fitted_amplitude)
                    means.append(fitted_mean)
                    stds.append(fitted_std)

                current_amplitude += 20
                count += 1

                if count > 5: 
                    amplitude_works = True

        errors = np.array(errors)
        bins = np.array(bins)
        amplitudes = np.array(amplitudes)
        means = np.array(means)
        stds = np.array(stds)

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

        if self.verbose: 
            print("")
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
            print("")           

        return optimized_error, optimized_bin, optimized_amplitude, optimized_mean, optimized_std


class Analysis: 

    def __init__(self, catalog1, catalog2, 
                 catalog1name, catalog2name, catalogyname, 
                 parallel_cutoff1, parallel_cutoff2, 
                 x_range, n, image_path, verbose = False): 

        self.catalog1 = catalog1
        self.catalog2 = catalog2
        self.catalog1name = catalog1name
        self.catalog2name = catalog2name
        self.catalogyname = catalogyname
        self.parallel_cutoff1 = parallel_cutoff1
        self.parallel_cutoff2 = parallel_cutoff2
        self.x_range = x_range
        self.n = n
        self.image_path = image_path 
        self.verbose = verbose

    def cutoffs(self, verbose = True): 

        fig, axis = plt.subplots(1, 1, figsize = (20, 10))
        plt.gca().invert_yaxis()

        x = np.subtract(self.catalog1, self.catalog2)

        line1 = self.parallel_cutoff1
        line2 = self.parallel_cutoff2

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
            raise Exception(f"Slopes of parallel cutoffs must be equal. \
                            Slope1: {slope1} / Slope2: {slope2} ")

        if self.catalogyname == self.catalog1name: 

            plt.scatter(x, self.catalog1, c = 'k', s = 0.05)
            plt.ylabel(f"{self.catalog1name}")

        if self.catalogyname == self.catalog2name: 

            plt.scatter(x, self.catalog2, c = 'k', s = 0.05)
            plt.ylabel(f"{self.catalog2name}")
        
        plt.axline(line1[0], line1[1], c = 'r', label = 'provided rc cutoff')
        plt.axline(line2[0], line2[1], c = 'r')

        plt.axline([0, intercept1], slope = slope1, c = 'aqua', label = 'actual rc cutoff')
        plt.axline([0, intercept2], slope = slope2, c = 'aqua',)

        plt.axvline(x = self.x_range[0], c = 'r', linestyle = ':', label = 'selected x range')
        plt.axvline(x = self.x_range[1], c = 'r', linestyle = ':')

        plt.xlabel(f"{self.catalog1name} - {self.catalog2name}")
        plt.legend()

        filename = f"{self.catalog1name}-{self.catalog2name}-vs{self.catalogyname}_cutoff"
        plt.savefig(f"{self.image_path}{filename}.png")

        plt.close()

        if self.verbose: 
            if verbose: 
                print("")
                print("{:>15} | {:>15}".format("Cutoff Slope", "Cutoff Intercept"))
                print("{:>15} | {:>15}".format(slope1, round(intercept1, 3)))
                print("{:>15} | {:>15}".format(slope1, round(intercept2, 3)))
                print("")

        if intercept1 < intercept2: 
            return intercept1, intercept2, slope1, height
        else: 
            return intercept2, intercept1, slope1, height

    def generate_bin(self, bin_x_range, show_plot = True, verbose = True): 

        intercept1, intercept2, slope, height = self.cutoffs(verbose)
        dx = bin_x_range[1] - bin_x_range[0]

        current_x = bin_x_range[0]
        yi = slope * current_x + intercept1
        yf = slope * current_x + intercept2

        segment = Rectangle((current_x, yi), dx, height, 
                            facecolor = (0,0,0,0), lw = 2, 
                            ec = (1,0,0,1)
        )

        bins = np.array(([current_x, current_x + dx], [yi, yf]))

        if show_plot: 
            fig, axis = plt.subplots(1, 1, figsize = (20, 10))
            plt.gca().invert_yaxis()

            x = np.subtract(self.catalog1, self.catalog2)

            if self.catalogyname == self.catalog1name: 

                plt.scatter(x, self.catalog1, c = 'k', s = 0.05)
                plt.ylabel(f"{self.catalog1name}")

            if self.catalogyname == self.catalog2name: 

                plt.scatter(x, self.catalog2, c = 'k', s = 0.05)
                plt.ylabel(f"{self.catalog2name}")

            axis.add_patch(segment)

            plt.axline([0, intercept1], slope = slope, c = 'aqua', label = 'actual rc cutoff')
            plt.axline([0, intercept2], slope = slope, c = 'aqua',)
            plt.xlabel(f"{self.catalog1name} - {self.catalog2name}")
            plt.legend()

            filename = f"{self.catalog1name}-{self.catalog2name}-vs{self.catalogyname}_{self.n}_tile_bins"
            plt.savefig(f"{self.image_path}{filename}.png")

        return bins, segment

    def extract_stars(self, bin_x_range, verbose = True): 

        bins, segment = self.generate_bin(bin_x_range, verbose = verbose, show_plot = False)

        x = np.subtract(self.catalog1, self.catalog2)
        idxs = []

        catalog1_mag, catalog2_mag = [], []

        if self.catalogyname == self.catalog1name: 
            y = self.catalog1
        if self.catalogyname == self.catalog2name:
            y = self.catalog2

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

    def optimize_bin_fitting(self, bin_x_range, show_hists = False): 

        starlist, idxs = self.extract_stars(bin_x_range)

        if self.catalogyname == self.catalog2name: 

            optimal_error, optimal_bin, optimal_amplitude, optimal_mean, optimal_std = Optimize(np.array(starlist[starlist.columns[1]][0]), 
                                                                                                starlist.columns[1], 
                                                                                                verbose = True
                                                                                        ).optimize_bin()
        if self.catalogyname == self.catalog1name: 

            optimal_error, optimal_bin, optimal_amplitude, optimal_mean, optimal_std = Optimize(np.array(starlist[starlist.columns[0]][0]), 
                                                                                                starlist.columns[0], 
                                                                                                verbose = True
                                                                                        ).optimize_bin()

        return optimal_error, optimal_bin, optimal_amplitude, optimal_mean, optimal_std

    def analysis(self, show_hists=False): 

        trial_dx = (self.x_range[1] - self.x_range[0]) / self.n 
        current_x = self.x_range[0]

        succesful_bin_parameters = []
        bins = []

        while current_x <= self.x_range[1]: 

            status = False
            count = 0

            while not status: 

                try: 

                    bin_x_range = [current_x, current_x + trial_dx + 0.1 * count]

                    error, num_bin, amplitude, mean, std = self.optimize_bin_fitting(bin_x_range, show_hists)
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

                if count > 5: 
                    raise Exception("optimize_bin() never found a suitable fitting - try altering x_range")

        return succesful_bin_parameters, bins

    def plot(self, show_hists = False): 

        succesful_bin_parameters, bins = self.analysis()

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

    def slope(self, show_plot):
        
        succesful_bin_parameters, bins = self.analysis()

        means = []
        errors = []
        midpoints = []

        for i in range(len(succesful_bin_parameters)): 
            means.append(succesful_bin_parameters[i][3])
            errors.append(succesful_bin_parameters[i][0])
            midpoints.append(bins[i][0] + (bins[i][1] - bins[i][0]) / 2)

        gradient, intercept, r_value, p_value, std_err = linregress(midpoints, means)

        linear = models.Linear1D(slope = gradient, intercept = intercept)
        fit = fitting.LevMarLSQFitter()

        result = fit(linear, midpoints, means)
        filename = ''

        if show_plot: 
            fig, axis = plt.subplots(1, 1, figsize = (20, 10))

            if self.catalogyname == self.catalog1name: 

                plt.scatter(np.subtract(self.catalog1, self.catalog2), 
                            self.catalog1, c = 'k', s = 0.1, alpha = 0.8)
                plt.xlabel(f"{self.catalog1name} - {self.catalog2name}")
                plt.ylabel(f"{self.catalog1name}")

            if self.catalogyname == self.catalog2name: 

                plt.scatter(np.subtract(self.catalog1, self.catalog2), 
                            self.catalog2, c = 'k', s = 0.1, alpha = 0.8)
                plt.xlabel(f"{self.catalog1name} - {self.catalog2name}")
                plt.ylabel(f"{self.catalog2name}")

            plt.plot(midpoints, linear(midpoints), 'r-', label = 'linear fit')
            plt.scatter(midpoints, means, c = 'cyan', s = 20, marker = 'x', label = 'optimized means')
            plt.errorbar(midpoints, means, yerr=errors, color="cyan", capsize=2, capthick=1, lw = 1, ls = 'none')

            plt.legend()
            plt.gca().invert_yaxis()
            plt.title(f"Fitted Slope: {result.slope.value.round(3)}")

            filename = f"{self.catalog1name}-{self.catalog2name}-vs{self.catalogyname}_slope"
            plt.savefig(f"{self.image_path}{filename}.png")

        print("")
        print(f"Calculated Red Clump Slope: {result.slope.value}")
        print("")

        return result.slope.value, result.intercept.value

    def residuals(self): 

        succesful_bin_parameters, bins = self.analysis()
        slope, intercept = self.slope(show_plot = False)

        starlist_full = []

        for i in bins: 
            starlist, idxs = self.extract_stars(i, verbose = False)
            starlist_full.append([starlist[starlist.columns[0]][0], starlist[starlist.columns[1]][0]])

        colors = plt.cm.jet(np.linspace(0, 1, len(starlist_full)))

        fig, axis = plt.subplots(2, 1,  figsize = (15, 15))

        for i in range(len(starlist_full)): 

            x = np.subtract(starlist_full[i][0], starlist_full[i][1])

            predicted_y = []
            actual_y = []
            residuals = []

            if self.catalogyname == self.catalog1name: 
                actual_y = starlist_full[i][0]

            if self.catalogyname == self.catalog2name: 
                actual_y = starlist_full[i][1]

            for j in x: 
                predicted_y.append(slope * j + intercept)

            predicted_y = np.array(predicted_y)

            for k in range(len(predicted_y)): 
                residuals.append(actual_y[k] - predicted_y[k])

            axis[0].scatter(x, residuals, color = colors[i], s = 0.3, label = 'Red Clump Cluster' )
            axis[1].hist(residuals, color = colors[i], histtype='step')

        axis[0].set_xlabel(f"{self.catalog1name} - {self.catalog2name}")
        axis[0].set_ylabel(f"Residual")

        axis[1].set_xlabel(f"Residual")
        axis[1].set_ylabel(f"Frequency")

        filename = f"{self.catalog1name}-{self.catalog2name}-vs{self.catalogyname}_residual"
        plt.savefig(f"{self.image_path}{filename}.png")






























fits ='~/research/work/catalogs/dr2/jwst_init_NRCB.fits'
catalog = Table.read(fits, format='fits')
catalog1, catalog2, catalog1error, catalog2error = get_matches(catalog, 'F115W', 'NRCB1', 'F212N', 'NRCB1')

catalog1 += 25.92
catalog2 += 22.12

catalog1name = 'F115W'
catalog2name = 'F212N'
catalogyname = 'F212N'

parallel_cutoff1, parallel_cutoff2 = [(6.2, 15.5), (7.9, 16.3)], [(6.65, 15.2), (8.35, 16.0)]
x_range = [5, 9]
n = 10
image_path = f"/Users/devaldeliwala/research/work/plots&data/rc_analysis_v2_plots/NRCB1/{catalog1name}-{catalog2name}/vs{catalogyname}/"

if not os.path.isdir(image_path):
        os.makedirs(image_path)

verbose = True 

test = Analysis(catalog1, catalog2, catalog1name, catalog2name, catalogyname, 
                parallel_cutoff1, parallel_cutoff2, x_range, n, image_path, verbose = True)

#test.plot(show_hists = True)
#test.plot(show_hists = False)

test.plot()

























