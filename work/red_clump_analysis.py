from color_magnitude_diagrams import * 
from isochrones import * 
from catalog_helper_functions import * 
from scipy.stats import linregress
from matplotlib.patches import Rectangle
import math

class Red_Clump_Analysis_vRect: 
    def __init__(self, catalog1, catalog2, 
                 catalog1name, catalog2name, catalogyname, 
                 xlim, ylim, n, image_path, matched = True): 

        """ 
        Red Clump analysis involves trying to determine the slope of the 
        RC bar -- which should equal the slope of the extinction vector 
        due to Red Clump's stars amazing properties. 

        Most versions of Red Clump analysis involves trial and error with 
        jupyter notebooks. Normally, you have to extract the region of the 
        RC clump, then a linear perform fitting to determine the rough slope 
        of the RC bar. 

        This code attempts to limit the amount of by-eye dependence on the 
        Red Clump cutoff, the linear fitting, the entire analysis as a whole. 

        ----------
        Functions: 
        ----------

        # def divide_cutoff(self, show_plot): 

        Divides a rectangular cutoff of the CMD surrounding the Red Clump 
        Bar into `n` bins and stores the bounds of each segment and plots
        the segments if `show_plot == True`. 

        Default is True to see if program works correctly. 

        # def extract_stars(self, show_plot = True, verbose = False):

        extracts stars in each bin returned from `divide_cutoff` based on 
        a `catalog1` - `catalog2` vs. `catalogy` CMD. `catalogy` is chosen
        based on whether `catalogyname` == `catalog1name` or `catalog2name`. 

        if show_plot == True: 
            shows each bin on a CMD for visualization purposes

        returns a starlist containing all the mag information for stars in 
        both filters that are in each bin. 

        if verbose == True: 
            prints out the starlist. 

        Returns the indicies of stars from the original catalog that are 
        in each bin and the starlist containing all the mag info for stars
        in each bin. 

        # def optimize_bin(self, data, show_plot = True, verbose = True):

        Performs all the RC analysis. 

        - Generates a histogram for an individual bin of the RC Bar. 

        - Uses `scipy.stats.norm` to determine preliminary mean and stddev 
        - Implements a compound Linear1D + Gaussian1D fit using `astropy.modeling`
        - Iterates through possible amplitude until the program observes the fitting
          works well 
        - Iterates through all fits that work well until it finds the fit with the 
          smallest error on the mean. (stddev / sqrt(# stars in bin))

         if show_plot == True: 
            displays histogram with the optimal number of bins such that the fitting 
            minmizes the error on the mean

         if verbose == True: 
            displays the optimized_bin_value, the error on the mean, the amplitude, 
            the mean, the stddev for this particular bin in table format. 

         Returns the parameters for the optimized fit. 

         # def generate_hits(starlist, path, verbose = False):

         Performs `optimize_bin` method for each bin in the RC bar established from 
         `divide_cutoff` and returns a 3D Plot of all the optimized fittings. 

         Returns all the means from the compound fitting, and their corresponding 
         error. 

         # def determine_slope(self, show_cmd = True, verbose = True):

         Determines the determined RC slope along with errors from `generate_hists`. 

         if show_cmd == True: # default is True
            overplots the CMD for `catalog1 - catalog2` vs. `catalogy`. 
            catalogy is catalog1 if catalogyname == catalog1name, and likewise
            for catalog2. 

        ------------------
         Class Parameters: 
        ------------------

        catalog1        : pandas DataFrame or array-like (see `matched` parameter)
        catalog2        stores mag information for stars.

        catalog1name    : string 
        catalog2name    name for filter of catalog1 and catalog2.
        

        catalogyname    : string
                        indicates which filter/catalog is to be placed on the y
                        axis of the CMD. (whichever catalogname it equals).

        xlim, ylim      : tuple
                        defines the x range and y range for the rectangular cutoff
                        of the CMD.

        n               : int
                        defines the number of segments/bins you want to divide the 
                        cutoff into. 

        image_path      : string
                        directory to store image files

        matched         : bool
                        default is True. If True, assumes `catalog1` and `catalog2` 
                        are already matched and are of the same size corresponding
                        to the same stars. 

                        If False, implements Color_Magnitude_Diagram.match from 
                        color_magnitude_diagrams.py. Remember this matching does
                        not take into account coordinate transformations and merely
                        implements a KDtree nearest neighbor algo with a dr_tol and 
                        dm_tol. 

        """

        self.catalog1 = catalog1
        self.catalog2 = catalog2
        self.catalog1name = catalog1name
        self.catalog2name = catalog2name
        self.catalogyname = catalogyname
        self.xlim = xlim
        self.ylim = ylim
        self.n = n 
        self.image_path = image_path
        self.matched = matched

    def divide_cutoff(self, show_plot = True, dr_tol = None, dm_tol = None):

        x, y = self.xlim[0], self.ylim[0]
        width = self.xlim[1] - self.xlim[0]
        dx = width / self.n 

        segment_height = self.ylim[1] - self.ylim[0]
        xbins = [] # stores the [leftest x, farthest x]...[ , ] for each subregion/bin
        segments = [] # stores the matplotlib.patches.Rectangle objects for each bin
                      # for visualization purposes

        for i in range(self.n): 
            segments.append(Rectangle((x + i * dx, y), dx, segment_height, 
                                      facecolor = (0, 0, 0, 0), lw = 2, 
                                      ec = (1, 0, 0, 1)
                            )
            )

            xbins.append([x + i * dx, x + i * dx + dx])

        if show_plot: 
            if self.matched: 
                fig, axis = plt.subplots(1, 1, figsize = (20, 10))
                plt.gca().invert_yaxis()
                check = False

                if self.catalogyname == self.catalog1name:
                    plt.scatter(np.subtract(self.catalog1, self.catalog2), self.catalog1,
                                c = 'k', s = 0.5)

                    for segment in segments: 
                        axis.add_patch(segment)

                    filename = f"{self.catalog1name}_{self.catalog2name}_{self.catalog1name}_{self.n}_bins"

                    plt.xlabel(f"{self.catalog1name} - {self.catalog2name}")
                    plt.ylabel(f"{self.catalog1name}")

                    plt.savefig(f"{self.image_path}{filename}.png")
                    check = True

                if self.catalogyname == self.catalog2name: 
                    plt.scatter(np.subtract(self.catalog1, self.catalog2), self.catalog2, 
                                c = 'k', s = 0.5)

                    for segment in segments:
                        axis.add_patch(segment)

                    filename = f"{self.catalog1name}_{self.catalog2name}_{self.catalog2name}_{self.n}_bins"

                    plt.xlabel(f"{self.catalog1name} - {self.catalog2name}")
                    plt.ylabel(f"{self.catalog2name}")  

                    plt.savefig(f"{self.image_path}{filename}.png")
                    check = True

                if not check: 
                    raise Exception("catalogyname must equal catalog1name or catalog2name")

            if not self.matched: 
                idxs1, idxs2, catalog1, catalog2, m1_error, m2_error = Color_Magnitude_Diagram(
                                                           self.catalog1, self.catalog2, 
                                                           self.catalog1name, self.catalog2name, self.catalogyname, 
                                                           dr_tol = dr_tol, dm_tol = dm_tol).match()

                fig, axis = plt.subplots(1, 1, figsize = (20, 10))
                plt.gca().invert_yaxis() 
                check = False

                plt.xlabel(f"{self.catalog1name} - {self.catalog2name}")
                plt.ylabel(f"{self.catalog1name}")

                if self.catalogyname == self.catalog1name:
                    plt.scatter(np.subtract(self.catalog1, self.catalog2), self.catalog1,
                                c = 'k', s = 0.5)

                    for segment in segments: 
                        axis.add_patch(segment)

                filename = f"{self.catalog1name}_{self.catalog2name}_{self.catalog1name}"
                plt.savefig(f"{self.image_path}{filename}.png")
                check = True

                if self.catalogyname == self.catalog2name: 
                    plt.scatter(np.subtract(self.catalog1, self.catalog2), self.catalog2, 
                                c = 'k', s = 0.5)

                    for segment in segments:
                        axis.add_patch(segment)

                filename = f"{self.catalog1name}_{self.catalog2name}_{self.catalog2name}"
                plt.savefig(f"{self.image_path}{filename}.png")
                check = True

                if not check: 
                    raise Exception("catalogyname must equal catalog1name or catalog2name")

        return segments, xbins

    def extract_stars(self, show_plot = True, verbose = False): 

        segments, xbins = self.divide_cutoff(show_plot = False)

        x = np.subtract(self.catalog1, self.catalog2) 
        
        lower_y = self.ylim[0]
        upper_y = self.ylim[1]
        
        idxs = []
        catalog1_mag = []
        catalog2_mag = []

        if self.matched: 
            check = False

            if self.catalogyname == self.catalog1name: 
                check = True
                count = 0
                y = self.catalog1

                for i in range(len(xbins)): 
                    good = np.where((x >= xbins[i][0]) & (x < xbins[i][1]) & 
                                    (y >= lower_y) & (y <= upper_y))
                    idxs.append([good])
                    catalog1_mag.append(self.catalog1[good])
                    catalog2_mag.append(self.catalog2[good])

                if show_plot: 
                    colors = plt.cm.jet(np.linspace(0, 1, len(catalog1_mag)))
                    fig, axis = plt.subplots(1, 1, figsize = (20, 10))

                    for i in range(len(catalog1_mag)): 
                        plt.scatter(np.subtract(catalog1_mag[i], catalog2_mag[i]), 
                                    catalog1_mag[i], color = colors[i], s = 0.5)
                        count += len(catalog1_mag[i])

                    plt.xlabel(f' {self.catalog1name} - {self.catalog2name} (mag) | Count: {count}')
                    plt.ylabel(f' {self.catalog1name}')
                    plt.gca().invert_yaxis()

                    filename = f"{self.catalog1name}_{self.catalog2name}_{self.catalog1name}_{self.n}_colorbins.png"
                    plt.savefig(f"{self.image_path}{filename}.png")

                df1 = pd.DataFrame({f'{self.catalog1name}': catalog1_mag})
                df2 = pd.DataFrame({f'{self.catalog2name}': catalog2_mag})
                starlist = pd.concat([df1, df2], axis = 1)

                """for i in range(len(starlist)): 
                    starlist[i, self.catalog1name] = starlist[i, self.catalog1name][0]
                    starlist[i, self.catalog2name] = starlist[i, self.catalog2name][0]"""

                if verbose: 
                    print(starlist)

            if self.catalogyname == self.catalog2name: 
                check = True
                count = 0
                y = self.catalog2

                for i in range(len(xbins)): 
                    good = np.where((x >= xbins[i][0]) & (x < xbins[i][1]) & 
                                    (y >= lower_y) & (y <= upper_y))
                    idxs.append([good])
                    catalog1_mag.append(self.catalog1[good])
                    catalog2_mag.append(self.catalog2[good])

                if show_plot: 
                    colors = plt.cm.jet(np.linspace(0, 1, len(catalog1_mag)))
                    fig, axis = plt.subplots(1, 1, figsize = (20, 10))

                    for i in range(len(catalog1_mag)): 
                        plt.scatter(np.subtract(catalog1_mag[i], catalog2_mag[i]), 
                                    catalog2_mag[i], color = colors[i], s = 0.5)
                        count += len(catalog2_mag[i])

                    plt.xlabel(f' {self.catalog1name} - {self.catalog2name} (mag) | Count: {count}')
                    plt.ylabel(f' {self.catalog2name}')
                    plt.gca().invert_yaxis()
                    
                    filename = f"{self.catalog1name}_{self.catalog2name}_{self.catalog2name}_{self.n}_colorbins.png"
                    plt.savefig(f"{self.image_path}{filename}.png")

                df1 = pd.DataFrame({f'{self.catalog1name}': catalog1_mag})
                df2 = pd.DataFrame({f'{self.catalog2name}': catalog2_mag})
                starlist = pd.concat([df1, df2], axis = 1)

                if not check: 
                    raise Exception("catalogyname must equal catalog1name or catalog2name")

                """for i in range(len(starlist)): 
                    starlist[self.catalog1name][i] = starlist[self.catalog1name][i][0]
                    starlist[self.catalog2name][i] = starlist[self.catalog2name][i][0]"""

                if verbose: 
                    print(starlist)

        if not self.matched: 
            check = False

            idxs1, idxs2, catalog1, catalog2, m1_error, m2_error = Color_Magnitude_Diagram(
                                                           self.catalog1, self.catalog2, 
                                                           self.catalog1name, self.catalog2name, self.catalogyname, 
                                                           dr_tol = 0.5, dm_tol = 100).match()
            if self.catalogyname == self.catalog1name: 
                check = True
                count = 0
                y = catalog1

                for i in range(len(xbins)): 
                    good = np.where((x >= xbins[i][0]) & (x < xbins[i][1]) & 
                                    (y >= lower_y) & (y <= upper_y))
                    idxs.append([good])
                    catalog1_mag.append(catalog1[good])
                    catalog2_mag.append(catalog2[good])

                if show_plot: 
                    colors = plt.cm.jet(np.linspace(0, 1, len(catalog1_mag)))
                    fig, axis = plt.subplots(1, 1, figsize = (20, 10))

                    for i in range(len(catalog1_mag)): 
                        plt.scatter(np.subtract(catalog1_mag[i], catalog2_mag[i]), 
                                    catalog1_mag[i], color = colors[i], s = 0.5)
                        count += len(catalog1_mag[i])

                    plt.xlabel(f' {self.catalog1name} - {self.catalog2name} (mag) | Count: {count}')
                    plt.ylabel(f' {self.catalog1name}')
                    plt.gca().invert_yaxis()

                    filename = f"{self.catalog1name}_{self.catalog2name}_{self.catalog1name}_{self.n}_colorbins.png"
                    plt.savefig(f"{self.image_path}{filename}.png")

                df1 = pd.DataFrame({f'{self.catalog1name}': catalog1_mag})
                df2 = pd.DataFrame({f'{self.catalog2name}': catalog2_mag})
                starlist = pd.concat([df1, df2], axis = 1)

                """for i in range(len(starlist)): 
                    starlist[self.catalog1name][i] = starlist[self.catalog1name][i][0]
                    starlist[self.catalog2name][i] = starlist[self.catalog2name][i][0]"""

                if verbose: 
                    print(starlist)

            if self.catalogyname == self.catalog2name: 
                check = True
                count = 0
                y = catalog2

                for i in range(len(xbins)): 
                    good = np.where((x >= xbins[i][0]) & (x < xbins[i][1]) & 
                                    (y >= lower_y) & (y <= upper_y))
                    idxs.append([good])
                    catalog1_mag.append(catalog1[good])
                    catalog2_mag.append(catalog2[good])

                if show_plot: 
                    colors = plt.cm.jet(np.linspace(0, 1, len(catalog1_mag)))
                    fig, axis = plt.subplots(1, 1, figsize = (20, 10))

                    for i in range(len(catalog1_mag)): 
                        plt.scatter(np.subtract(catalog1_mag[i], catalog2_mag[i]), 
                                    catalog2_mag[i], color = colors[i], s = 0.5)
                        count += len(catalog2_mag[i][0])

                    plt.xlabel(f' {self.catalog1name} - {self.catalog2name} (mag) | Count: {count}')
                    plt.ylabel(f' {self.catalog2name}')
                    plt.gca().invert_yaxis()
                    
                    filename = f"{self.catalog1name}_{self.catalog2name}_{self.catalog2name}_{self.n}_colorbins.png"
                    plt.savefig(f"{self.image_path}{filename}.png")

                df1 = pd.DataFrame({f'{self.catalog1name}': catalog1_mag})
                df2 = pd.DataFrame({f'{self.catalog2name}': catalog2_mag})
                starlist = pd.concat([df1, df2], axis = 1)

                if not check: 
                    raise Exception("catalogyname must equal catalog1name or catalog2name")

                """for i in range(len(starlist)): 
                    starlist[self.catalog1name][i] = starlist.loc[i, self.catalog1name][0]
                    starlist[self.catalog2name][i] = starlist.loc[i, self.catalog2name][0]"""

                if verbose: 
                    print(starlist)

        return starlist, idxs

    def optimize_bin(self, data, data_name, show_plot = True, verbose = True): 

        # find the # of bins in a histogram of `data` that minimizes the error_on_the_mean

        mu, std = norm.fit(data) # preliminary mean and std using scipy.stats
        std = tstd(data, limits = (mu - 0.4, mu + 0.4))
        error_bin_amplitude_mean_std = [] 

        for i in range(8, 20): # range is 8 to 20 bins. Any more we get noise. Any less it's not enough.
            amplitude_works = False
            j = 10
            count = 0

            while not amplitude_works:
            
                bin_heights, bin_borders = np.histogram(data, bins = i)
                bin_widths = np.diff(bin_borders)
                bin_centers = bin_borders[:-1] + bin_widths / 2
        
                t_gaussian = models.Gaussian1D(j, mu, std)
                t_linear = models.Linear1D(100, 100)
                t_compound = t_gaussian + t_linear
        
                filt_t = fitting.LevMarLSQFitter()
                t = filt_t(t_compound, bin_centers, bin_heights)

                if t.stddev_0.value > 0.1 and t.amplitude_0.value < 500 and t.amplitude_0.value > 0: # testing for if the fit works
                    amplitude_works = True
                    error_on_mean = t.stddev_0.value / math.sqrt(len(data))
                    error_bin_amplitude_mean_std.append([error_on_mean, i, t.amplitude_0.value, t.mean_0.value, t.stddev_0.value])
                
                j += 20
                count += 1
                
                if count > 5: 
                    amplitude_works = True
    
        if show_plot: 
            fig, axis = plt.subplots(1, 1, figsize = (20, 10))
            x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
            
            plt.bar(bin_centers, bin_heights, width=bin_widths, label='histogram')
            plt.plot(x_interval_for_fit, t(x_interval_for_fit), label='fit', c='red', linestyle = 'dashed')
            plt.legend()

            plt.xlabel(f'{data_name}')
            plt.ylabel('Frequency')

            mean = t.mean_0.value.round(3)
            std = t.stddev_0.value.round(3)
            error = (t.stddev_0.value / math.sqrt(len(data))).round(3)

            plt.title(f" Compound-Fitted Histogram | Mean = {mean} | σ = {std} | Error_on_mean = {error}")

        bins = []
        errors = [] 
        amplitudes = []
        means = []
        stds = []
        
        for i in range(len(error_bin_amplitude_mean_std)): # extracting error values into its own array
            bins.append(error_bin_amplitude_mean_std[i][1])
            errors.append(error_bin_amplitude_mean_std[i][0])
            amplitudes.append(error_bin_amplitude_mean_std[i][2])
            means.append(error_bin_amplitude_mean_std[i][3])
            stds.append(error_bin_amplitude_mean_std[i][4])

        bins = np.array(bins)
        errors = np.array(errors)
        amplitudes = np.array(amplitudes)
        means = np.array(means)
        stds = np.array(stds)
        
        while means[np.where(errors == errors.min())] > 17: 
            errors = np.delete(errors, np.where(errors == errors.min()))
            means = np.delete(means, np.where(errors == errors.min()))

        min_index = np.where((errors == errors.min())) # determining the # of bins with the smallest error_on_mean
        min_index = min_index[0][0]

        optimized_error = errors[min_index].round(3)
        optimized_bin_value = bins[min_index].round(3)
        optimized_amplitude = amplitudes[min_index].round(3)
        optimized_mean = means[min_index].round(3)
        optimized_std = stds[min_index].round(3)

        if verbose: 
            print(f"{data_name}")
            print(f"Accepted Bin #s: {bins}")
            print(" {:>13} | {:>8} | {:>11} | {:>8} | {:>8} ".format("Optimal Bin #", 
                                                                     "Error", 
                                                                     "Amplitude", 
                                                                     "Mean", 
                                                                     "Stddev"))
            print(" {:>13} | {:>8} | {:>11} | {:>8} | {:>8}".format(optimized_bin_value, 
                                                                     optimized_error, 
                                                                     optimized_amplitude, 
                                                                     optimized_mean, 
                                                                     optimized_std))
            print("")

        return optimized_bin_value, optimized_amplitude, optimized_mean, optimized_error, optimized_std

    def generate_hists(self, verbose = False, show_hists = False):

        # if show_hists == True: 
        #   displays the individual hists on a that make up each fitting on the 3D plot

        starlist, idxs = self.extract_stars(show_plot = True, verbose = verbose)

        ax = plt.figure(figsize = (10, 10)).add_subplot(projection = '3d')
        ax.view_init(elev=35, azim=-45, roll=0)
        colors = plt.cm.jet(np.linspace(0,1,len(starlist)))

        ax.set_xlabel(f"{starlist.columns[0]} - {starlist.columns[1]}")
        ax.set_ylabel(f"{starlist.columns[1]}")
        ax.set_zlabel("Frequency")
        ax.zaxis.labelpad=-0.01 # <- change the value here

        optimized_means = []
        optimized_mean_errors = []

        for i in range(len(starlist)): 

            if self.catalogyname == self.catalog2name: 
            
                ax.scatter(np.subtract(starlist[starlist.columns[0]][i], starlist[starlist.columns[1]][i]), 
                           starlist[starlist.columns[1]][i], 
                           zs = 0, zdir = 'z', label = 'CMD', 
                           color = colors[i], s = 0.3
                          )
                min_x = np.min(np.subtract(starlist[starlist.columns[0]][i], starlist[starlist.columns[1]][i]))

                # fitting 
                optimized_bin_value, optimized_amplitude, optimized_mean, optimized_error, optimized_std = self.optimize_bin(
                                                                                                           starlist[starlist.columns[1]][i], 
                                                                                                           starlist.columns[1], 
                                                                                                           show_plot = False, 
                                                                                                           verbose = verbose
                )

                optimized_means.append(optimized_mean)
                optimized_mean_errors.append(optimized_error)

                bin_heights, bin_borders = np.histogram(starlist[starlist.columns[1]][i], bins=optimized_bin_value)
                bin_widths = np.diff(bin_borders)
                bin_centers = bin_borders[:-1] + bin_widths / 2
                
                t_gaussian = models.Gaussian1D(optimized_amplitude, optimized_mean, optimized_std)
                t_linear = models.Linear1D(100, 100)
                t_compound = t_gaussian + t_linear
                
                fit_t = fitting.LevMarLSQFitter()
                t = fit_t(t_compound, bin_centers, bin_heights)
                
                x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
                
                plt.plot(x_interval_for_fit, t(x_interval_for_fit), label='fit', c=colors[i], zs = min_x, zdir = 'x')

                if show_hists: 
                    plt.bar(bin_centers, bin_heights, width=bin_widths, label='histogram', 
                            zs = min_x, zdir = 'x', ec = 'k', facecolor = (0,0,0,0))
                    plt.savefig(f"{self.image_path}{starlist.columns[1]}_optimized_with_{len(starlist)}_bins_with_histograms.png")

                else: 
                    plt.savefig(f"{self.image_path}{starlist.columns[1]}_optimized_with_{len(starlist)}_bins.png")
                    
            if self.catalogyname == self.catalog1name: 
            
                ax.scatter(np.subtract(starlist[starlist.columns[0]][i], starlist[starlist.columns[1]][i]), 
                           starlist[starlist.columns[0]][i], 
                           zs = 0, zdir = 'z', label = 'CMD', 
                           color = colors[i], s = 0.3
                          )
                min_x = np.min(np.subtract(starlist[starlist.columns[0]][i], starlist[starlist.columns[1]][i]))

                # fitting 
                optimized_bin_value, optimized_amplitude, optimized_mean, optimized_error, optimized_std = self.optimize_bin(
                                                                                                           starlist[starlist.columns[0]][i], 
                                                                                                           starlist.columns[0], 
                                                                                                           show_plot = False, 
                                                                                                           verbose = verbose
                )

                optimized_means.append(optimized_mean)
                optimized_mean_errors.append(optimized_error)

                bin_heights, bin_borders = np.histogram(starlist[starlist.columns[0]][i], bins=optimized_bin_value)
                bin_widths = np.diff(bin_borders)
                bin_centers = bin_borders[:-1] + bin_widths / 2
                
                t_gaussian = models.Gaussian1D(optimized_amplitude, optimized_mean, optimized_std)
                t_linear = models.Linear1D(100, 100)
                t_compound = t_gaussian + t_linear
                
                fit_t = fitting.LevMarLSQFitter()
                t = fit_t(t_compound, bin_centers, bin_heights)
                
                x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)

                plt.plot(x_interval_for_fit, t(x_interval_for_fit), label='fit', c=colors[i], zs = min_x, zdir = 'x')
                
                if show_hists: 
                    plt.bar(bin_centers, bin_heights, width=bin_widths, label='histogram', 
                        zs = min_x, zdir = 'x', ec = 'k', facecolor = (0,0,0,0))
                    plt.savefig(f"{self.image_path}{starlist.columns[0]}_optimized_with_{len(starlist)}_bins_with_histograms.png")

                else: 
                    plt.savefig(f"{self.image_path}{starlist.columns[0]}_optimized_with_{len(starlist)}_bins.png")

        return optimized_means, optimized_mean_errors

    def determine_slope(self, show_cmd = True, verbose = True): 
        optimized_means, optimized_mean_errors = self.generate_hists(verbose = False)
        x = np.subtract(self.catalog1, self.catalog2)

        segments, xbins = self.divide_cutoff(show_plot = False)
        x_midpoints = [] 

        for i in range(len(xbins)): 
            x_midpoints.append((xbins[i][0] + xbins[i][1]) / 2)

        """if self.matched: 
            fig, axis = plt.subplots(1, 1, figsize = (20, 10))
            
            check = False

            if self.catalogyname == self.catalog1name:
                plt.scatter(x, self.catalog1, c = 'k', s = 0.05)
                plt.xlabel(f"{self.catalog1name} - {self.catalog2name}")
                plt.ylabel(f"{self.catalog1name}")

                filename = f"{self.catalog1}_{self.catalog2}_{self.catalog1}_rcfit"
                check = True

            if self.catalogyname == self.catalog2name: 
                plt.scatter(x, self.catalog2, c = 'k', s = 0.05)
                plt.xlabel(f"{self.catalog1name} - {self.catalog2name}")
                plt.ylabel(f"{self.catalog2name}")

                filename = f"{self.catalog1}_{self.catalog2}_{self.catalog2}_rcfit"
                check = True

            if not check:
                raise Exception("catalogyname must equal catalog1name or catalog2name")"""

        fig, axis = plt.subplots(1, 1, figsize = (20, 10))
        check = False

        if self.catalogyname == self.catalog1name: 
            plt.scatter(x, self.catalog1, c = 'k', s = 0.1, alpha = 0.8)
            plt.xlabel(f"{self.catalog1name} - {self.catalog2name}")
            plt.ylabel(f"{self.catalog1name}")

            filename = f"{self.catalog1name}_{self.catalog2name}_{self.catalog1name}_{self.n}bins_rcfit"
            check = True
        if self.catalogyname == self.catalog2name: 
            plt.scatter(x, self.catalog2, c = 'k', s = 0.1, alpha = 0.8)
            plt.xlabel(f"{self.catalog1name} - {self.catalog2name}")
            plt.ylabel(f"{self.catalog2name}")

            filename = f"{self.catalog1name}_{self.catalog2name}_{self.catalog2name}_{self.n}bins_rcfit"
            check = True

        if not check:
            raise Exception("catalogyname must equal catalog1name or catalog2name")

        gradient, intercept, r_value, p_value, std_err = linregress(x_midpoints, optimized_means)

        line_orig = models.Linear1D(slope = gradient, intercept = intercept)
        fit = fitting.LevMarLSQFitter()
        line_init = models.Linear1D()

        fitted_line = fit(line_init, x_midpoints, optimized_means)

        plt.plot(x_midpoints, line_orig(x_midpoints), 'r-', label = 'linear fit')
        plt.scatter(x_midpoints, optimized_means, c = 'cyan', s = 10, label = 'optimized means')
        
        plt.legend()
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.gca().invert_yaxis()
        plt.title(f"Fitted Slope: {fitted_line.slope.value.round(3)}")

        plt.savefig(f"{self.image_path}{filename}.png")

        if verbose: 
            print(f"Mean Errors For The {self.n} Segments:")
            print(optimized_mean_errors)
            print(f"")
            print(fitted_line)

        return fitted_line.slope.value, fitted_line.intercept.value

    def overplot_isochrones(self, logAge, AKs, AKs_step, dist, 
                            height, metallicity, filters, iso_dir,
                            show_cmd = True, verbose = True): 

        # filters = [filt1name, filt2name] - refer to SPISEA documentation, filter names are specific and unique. 

        slope, intercept = self.determine_slope(verbose = False)
        check = False

        catalog1 = np.array(self.catalog1)
        catalog2 = np.array(self.catalog2)

        x = np.subtract(catalog1, catalog2)

        red_law = reddening.RedLawFritz11(scale_lambda = 2.166) 
        evo_model = evolution.MISTv1()                  # evolution model
        atm_func = atmospheres.get_merged_atmosphere    # atmospheric model

        AKs2 = AKs + AKs_step
        AKs3 = AKs2 + AKs_step
        AKs4 = AKs3 + AKs_step
        AKs5 = AKs4 + AKs_step

        my_iso = isochrone = synthetic.IsochronePhot(logAge, AKs, dist, 
                                           metallicity, evo_model, atm_func, red_law = red_law, 
                                           filters = filters, iso_dir = iso_dir)
        idx = np.where( abs(my_iso.points['mass'] - 1.0) == min(abs(my_iso.points['mass'] - 1.0)) )[0]

        AKs = AKs2
        my_iso2 = isochrone = synthetic.IsochronePhot(logAge, AKs, dist, 
                                           metallicity, evo_model, atm_func, red_law = red_law, 
                                           filters = filters, iso_dir = iso_dir)
        idx2 = np.where( abs(my_iso2.points['mass'] - 1.0) == min(abs(my_iso2.points['mass'] - 1.0)) )[0]

        AKs = AKs3
        my_iso3 = isochrone = synthetic.IsochronePhot(logAge, AKs, dist, 
                                           metallicity, evo_model, atm_func, red_law = red_law, 
                                           filters = filters, iso_dir = iso_dir)
        idx3 = np.where( abs(my_iso3.points['mass'] - 1.0) == min(abs(my_iso3.points['mass'] - 1.0)) )[0]

        AKs = AKs4
        my_iso4 = isochrone = synthetic.IsochronePhot(logAge, AKs, dist, 
                                           metallicity, evo_model, atm_func, red_law = red_law, 
                                           filters = filters, iso_dir = iso_dir)
        idx4 = np.where( abs(my_iso4.points['mass'] - 1.0) == min(abs(my_iso4.points['mass'] - 1.0)) )[0]

        AKs = AKs5
        my_iso5 = isochrone = synthetic.IsochronePhot(logAge, AKs, dist, 
                                           metallicity, evo_model, atm_func, red_law = red_law, 
                                           filters = filters, iso_dir = iso_dir)
        idx5 = np.where( abs(my_iso5.points['mass'] - 1.0) == min(abs(my_iso5.points['mass'] - 1.0)) )[0]

        
        fig, axis = plt.subplots(1, 1, figsize = (20, 10))

        if self.catalogyname == self.catalog1name: 
            check = True
            y = np.array(self.catalog1)

            plt.scatter(x, y, c = 'k', s = 0.05)

            plt.plot(my_iso.points[''+my_iso.points.keys()[8]] - my_iso.points[''+my_iso.points.keys()[9]],
                     my_iso.points[''+my_iso.points.keys()[8]], 'r-', label='_nolegend_')
            plt.plot(my_iso2.points[''+my_iso2.points.keys()[8]] - my_iso2.points[''+my_iso2.points.keys()[9]],
                     my_iso2.points[''+my_iso2.points.keys()[8]], 'r-', label='_nolegend_')
            plt.plot(my_iso3.points[''+my_iso3.points.keys()[8]] - my_iso3.points[''+my_iso3.points.keys()[9]],
                     my_iso3.points[''+my_iso3.points.keys()[8]], 'r-', label='_nolegend_')
            plt.plot(my_iso4.points[''+my_iso4.points.keys()[8]] - my_iso4.points[''+my_iso4.points.keys()[9]],
                     my_iso4.points[''+my_iso4.points.keys()[8]], 'r-', label='_nolegend_')
            plt.plot(my_iso5.points[''+my_iso5.points.keys()[8]] - my_iso5.points[''+my_iso5.points.keys()[9]],
                     my_iso5.points[''+my_iso5.points.keys()[8]], 'r-', label='_nolegend_')
            
            plt.axline((my_iso5.points[''+my_iso5.points.keys()[8]][idx5][0]
                        - my_iso5.points[''+my_iso5.points.keys()[9]][idx5][0], 
                        my_iso5.points[''+my_iso5.points.keys()[8]][idx5][0] + height), 
                        (my_iso.points[''+my_iso.points.keys()[8]][idx][0]
                        - my_iso.points[''+my_iso.points.keys()[9]][idx][0],
                        my_iso.points[''+my_iso.points.keys()[8]][idx][0] + height), 
                        color = 'aqua', label = "isochrone extinction vector")

            plt.xlabel(f"{self.catalog1name} - {self.catalog2name}")
            plt.ylabel(f"{self.catalog1name}")
            filename = f"extinction_vec_{self.catalog1name}_{self.catalog2name}_{self.catalog1name}_{self.n}_tiled_bins" 

        if self.catalogyname == self.catalog2name: 
            check = True
            y = np.array(self.catalog2)

            plt.scatter(x, y, c = 'k', s = 0.05)

            plt.plot(my_iso.points[''+my_iso.points.keys()[8]] - my_iso.points[''+my_iso.points.keys()[9]],
                     my_iso.points[''+my_iso.points.keys()[9]], 'r-', label='_nolegend_')
            plt.plot(my_iso2.points[''+my_iso2.points.keys()[8]] - my_iso2.points[''+my_iso2.points.keys()[9]],
                     my_iso2.points[''+my_iso2.points.keys()[9]], 'r-', label='_nolegend_')
            plt.plot(my_iso3.points[''+my_iso3.points.keys()[8]] - my_iso3.points[''+my_iso3.points.keys()[9]],
                     my_iso3.points[''+my_iso3.points.keys()[9]], 'r-', label='_nolegend_')
            plt.plot(my_iso4.points[''+my_iso4.points.keys()[8]] - my_iso4.points[''+my_iso4.points.keys()[9]],
                     my_iso4.points[''+my_iso4.points.keys()[9]], 'r-', label='_nolegend_')
            plt.plot(my_iso5.points[''+my_iso5.points.keys()[8]] - my_iso5.points[''+my_iso5.points.keys()[9]],
                     my_iso5.points[''+my_iso5.points.keys()[9]], 'r-', label='_nolegend_')
            
            plt.axline((my_iso5.points[''+my_iso5.points.keys()[8]][idx5][0]
                        - my_iso5.points[''+my_iso5.points.keys()[9]][idx5][0], 
                        my_iso5.points[''+my_iso5.points.keys()[9]][idx5][0] + height), 
                        (my_iso.points[''+my_iso.points.keys()[8]][idx][0]
                        - my_iso.points[''+my_iso.points.keys()[9]][idx][0],
                        my_iso.points[''+my_iso.points.keys()[9]][idx][0] + height), 
                        color = 'aqua', label = "isochrone extinction vector")

            plt.xlabel(f"{self.catalog1name} - {self.catalog2name}")
            plt.ylabel(f"{self.catalog2name}")
            filename = f"extinction_vec_{self.catalog1name}_{self.catalog2name}_{self.catalog2name}_{self.n}_tiled_bins"    

        plt.axline((0, intercept), slope = slope, c = 'r', label = 'derived extinction vector') 
        plt.legend()
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.gca().invert_yaxis()
        plt.savefig(f"{self.image_path}{filename}.png")   

        if not check: 
            raise Exception("catalogyname must equal catalog1name or catalog2name")

        return 


class Red_Clump_Analysis_vRiemann:
    def __init__(self, catalog1, catalog2, 
                 catalog1name, catalog2name, catalogyname, 
                 parallel_cutoff1, parallel_cutoff2, x_range, n, 
                 image_path): 

        """
            Same as Red_Clump_Analysis_vRect except it implements 
            a `n` numbered tiled-rectangular cutoff that follows a rough estimate of the 
            slope of the RC-bar.

            Requires two parallel lines defined by two points each along the edges of the RC bar. 
            It expands the width of these lines by ~3 and afterwards establishes the rectangular bins. 

            For use if the RC bar has a rectangular cutoff that includes the main sequence track
            (i.e. you can not solely isolate the RC bar using a rectangle.)

            Similar to vRect, performs the analysis against a catalog1 - catalog2 vs `catalogy` CMD
            where `catalogy` is whichever catalog with the same `catalogname` as `catalogyname`

            parallel_cutoff1    :  list
            [(x1, y1), (x2, y2)] for first parallel cutoff

            parallel_cutoff2    : list
            [(x1, y1), (x2, y2)] for second parallel cutoff 

            x_range             : tuple
            (x_min, x_max) where the RC bar starts and ends on the x-axis
            defines where the tile bins starts and ends. 

            *Assumes stars are already matched and catalog1 and catalog2 are of the same size 

            If stars are not matched, implement Color_Magnitude_Diagram.match() from
            color_magnitude_diagrams.py to perform a nearest neighbor algorithms on the 
            catalogs (provided they contain `x` and `y` centroid information for each star). 
            This algorithm does not perform any coordinate transformations. 

        """

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

    def display_cutoffs(self, verbose = False): 

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
        """
        height = 4 * height # increasing height 4x to select more stars outside

        if intercept1 > intercept2: 
            intercept1 += height / 2.67
            intercept2 -= height / 2.67
        else: 
            intercept2 += height / 2.67
            intercept1 -= height / 2.67
        """
        height = 3 * height # increasing height 3x to select more stars outside

        if intercept1 > intercept2: 
            intercept1 += height / 3
            intercept2 -= height / 3
        else: 
            intercept2 += height / 3
            intercept1 -= height / 3 

        if slope1 != slope2:
            raise Exception(f"Slopes of *parallel* cutoffs must be equal. \
                            Slope1: {slope1} / Slope2: {slope2} ")

        if self.catalogyname == self.catalog1name: 
            plt.scatter(x, self.catalog1, c = 'k', s = 0.05)
            filename = f"{self.catalog1name}_{self.catalog2name}_{self.catalog1name}_parallel_cutoff"
            plt.ylabel(f"{self.catalog1name}")
        if self.catalogyname == self.catalog2name: 
            plt.scatter(x, self.catalog2, c = 'k', s = 0.05)
            filename = f"{self.catalog1name}_{self.catalog2name}_{self.catalog2name}_parallel_cutoff"
            plt.ylabel(f"{self.catalog2name}")
        
        plt.axline(line1[0], line1[1], c = 'r', label = 'provided rc cutoff')
        plt.axline(line2[0], line2[1], c = 'r')

        plt.axline([0, intercept1], slope = slope1, c = 'aqua', label = 'actual rc cutoff')
        plt.axline([0, intercept2], slope = slope2, c = 'aqua',)

        plt.axvline(x = self.x_range[0], c = 'r', linestyle = ':', label = 'selected x range')
        plt.axvline(x = self.x_range[1], c = 'r', linestyle = ':')

        plt.xlabel(f"{self.catalog1name} - {self.catalog2name}")
        plt.legend()
        plt.savefig(f"{self.image_path}{filename}.png")

        if verbose: 
            print("{:>15} | {:>15}".format("Cutoff Slope", "Cutoff Intercept"))
            print("{:>15} | {:>15}".format(slope1, round(intercept1, 3)))
            print("{:>15} | {:>15}".format(slope1, round(intercept2, 3)))

        if intercept1 < intercept2: 
            return intercept1, intercept2, slope1, height
        else: 
            return intercept2, intercept1, slope1, height

    def generate_tile_bins(self, show_plot = True, verbose = False):

        intercept1, intercept2, slope, height = self.display_cutoffs()
        dx = (self.x_range[1] - self.x_range[0]) / self.n 
        
        bins = []
        segments = []
        current_x = self.x_range[0]

        for i in range(self.n): 
            yi = slope * (current_x + i * dx) + intercept1
            yf = slope * (current_x + i * dx) + intercept2

            segments.append(Rectangle((current_x + i * dx, yi), dx, height, 
                                      facecolor = (0, 0, 0, 0), lw = 2, 
                                      ec = (1, 0, 0, 1)
                            )
            )

            bins.append([[current_x + i * dx, current_x + i * dx + dx], [yi, yf]])
        bins = np.array(bins)

        if show_plot: 
            fig, axis = plt.subplots(1, 1, figsize = (20, 10))
            plt.gca().invert_yaxis()

            x = np.subtract(self.catalog1, self.catalog2)

            if self.catalogyname == self.catalog1name: 
                plt.scatter(x, self.catalog1, c = 'k', s = 0.05)
                filename = f"{self.catalog1name}_{self.catalog2name}_{self.catalog1name}_{self.n}_tile_bins"
                plt.ylabel(f"{self.catalog1name}")
            if self.catalogyname == self.catalog2name: 
                plt.scatter(x, self.catalog2, c = 'k', s = 0.05)
                filename = f"{self.catalog1name}_{self.catalog2name}_{self.catalog2name}_{self.n}_tile_bins"
                plt.ylabel(f"{self.catalog2name}")

            for segment in segments: 
                axis.add_patch(segment)

            plt.axline([0, intercept1], slope = slope, c = 'aqua', label = 'actual rc cutoff')
            plt.axline([0, intercept2], slope = slope, c = 'aqua',)

            plt.axvline(x = self.x_range[0], c = 'r', linestyle = ':', label = 'selected x range')
            plt.axvline(x = self.x_range[1], c = 'r', linestyle = ':')

            plt.xlabel(f"{self.catalog1name} - {self.catalog2name}")
            plt.legend()
            plt.savefig(f"{self.image_path}{filename}.png")

        if verbose: 
            print(bins)

        return bins, segments

    def extract_tile_stars(self, show_plot = True, verbose = False): 

        bins, segments = self.generate_tile_bins(show_plot = False)
        x = np.subtract(self.catalog1, self.catalog2)
        idxs = []

        catalog1_mag = []
        catalog2_mag = []

        if self.catalogyname == self.catalog1name: 
            y = self.catalog1
        if self.catalogyname == self.catalog2name:
            y = self.catalog2

        for i in range(len(bins)): 
            good = np.where( (x >= bins[i][0][0]) & (x < bins[i][0][1]) & 
                             (y >= bins[i][1][0]) & (y < bins[i][1][1]) )
            idxs.append([good])
            catalog1_mag.append(self.catalog1[good])
            catalog2_mag.append(self.catalog2[good])

        if show_plot: 
            colors = plt.cm.jet(np.linspace(0, 1, len(catalog1_mag)))

            fig, axis = plt.subplots(1, 1, figsize = (20, 10))
            plt.gca().invert_yaxis()
            plt.xlabel(f"{self.catalog1name} - {self.catalog2name}")

            for i in range(len(catalog1_mag)):
                if self.catalogyname == self.catalog1name: 
                    plt.scatter(np.subtract(catalog1_mag[i], catalog2_mag[i]), 
                                catalog1_mag[i], color = colors[i], s = 0.5)
                    plt.ylabel(f"{self.catalog1name}")
                    filename = f"{self.catalog1name}_{self.catalog2name}_{self.catalog1name}_{self.n}_tiled_stars"
                if self.catalogyname == self.catalog2name: 
                    plt.scatter(np.subtract(catalog1_mag[i], catalog2_mag[i]), 
                                catalog2_mag[i], color = colors[i], s = 0.5)
                    plt.ylabel(f"{self.catalog2name}")
                    filename = f"{self.catalog1name}_{self.catalog2name}_{self.catalog2name}_{self.n}_tiled_stars"

            plt.savefig(f"{self.image_path}{filename}.png")

        df1 = pd.DataFrame({f'{self.catalog1name}': catalog1_mag})
        df2 = pd.DataFrame({f'{self.catalog2name}': catalog2_mag})
        starlist = pd.concat([df1, df2], axis = 1)

        #for i in range(len(starlist)): 
        #    starlist[i, self.catalog1name] = starlist[i, self.catalog1name][0]
        #    starlist[i, self.catalog2name] = starlist[i, self.catalog2name][0]

        if verbose: 
            print(starlist)

        return starlist, idxs 

    def find_max_mag(self): 
        starlist, idxs = self.extract_tile_stars(show_plot = False) 

        max_mag = 0

        if self.catalogyname == self.catalog1name: 
            for i in range(len(starlist)):
                for j in starlist[starlist.columns[0]][i]: 
                   
                    if j > max_mag: 
                        max_mag = j
        
        if self.catalogyname == self.catalog2name: 
            for i in range(len(starlist)):
                for j in starlist[starlist.columns[1]][i]: 
                    if j > max_mag: 
                        max_mag = j
        return max_mag

    def optimize_tile_bin(self, data, data_name, show_plot = True, verbose = True):

                # find the # of bins in a histogram of `data` that minimizes the error_on_the_mean

        mu, std = norm.fit(data) # preliminary mean and std using scipy.stats
        std = tstd(data, limits = (mu - 0.4, mu + 0.4))
        error_bin_amplitude_mean_std = [] 

        max_mag = self.find_max_mag()

        for i in range(8, 20): # range is 8 to 20 bins. Any more we get noise. Any less it's not enough.
            amplitude_works = False
            j = 10
            count = 0

            while not amplitude_works:
            
                bin_heights, bin_borders = np.histogram(data, bins = i)
                bin_widths = np.diff(bin_borders)
                bin_centers = bin_borders[:-1] + bin_widths / 2
        
                t_gaussian = models.Gaussian1D(j, mu, std)
                t_linear = models.Linear1D(100, 100)
                t_compound = t_gaussian + t_linear
        
                filt_t = fitting.LevMarLSQFitter()
                t = filt_t(t_compound, bin_centers, bin_heights)

                if t.stddev_0.value > 0.1 and t.amplitude_0.value < 500 and t.amplitude_0.value > 0: # testing for if the fit works
                    amplitude_works = True
                    error_on_mean = t.stddev_0.value / math.sqrt(len(data))
                    error_bin_amplitude_mean_std.append([error_on_mean, i, t.amplitude_0.value, t.mean_0.value, t.stddev_0.value])
                
                j += 20
                count += 1
                
                if count > 5: 
                    amplitude_works = True
    
        if show_plot: 
            fig, axis = plt.subplots(1, 1, figsize = (20, 10))
            x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)

            plt.bar(bin_centers, bin_heights, width=bin_widths, label='histogram')
            plt.plot(x_interval_for_fit, t(x_interval_for_fit), label='fit', c='red', linestyle = 'dashed')
            plt.legend()

            plt.xlabel(f'{data_name}')
            plt.ylabel('Frequency')

            mean = t.mean_0.value.round(3)
            std = t.stddev_0.value.round(3)
            error = (t.stddev_0.value / math.sqrt(len(data))).round(3)

            plt.title(f" Compound-Fitted Histogram | Mean = {mean} | σ = {std} | Error_on_mean = {error}")
            plt.show()


        bins = []
        errors = [] 
        amplitudes = []
        means = []
        stds = []
        
        for i in range(len(error_bin_amplitude_mean_std)): # extracting error values into its own array
            bins.append(error_bin_amplitude_mean_std[i][1])
            errors.append(error_bin_amplitude_mean_std[i][0])
            amplitudes.append(error_bin_amplitude_mean_std[i][2])
            means.append(error_bin_amplitude_mean_std[i][3])
            stds.append(error_bin_amplitude_mean_std[i][4])

        bins = np.array(bins)
        errors = np.array(errors)
        amplitudes = np.array(amplitudes)
        means = np.array(means)
        stds = np.array(stds)
        
        while means[np.where(errors == errors.min())] > max_mag:
            errors = np.delete(errors, np.where(errors == errors.min()))
            means = np.delete(means, np.where(errors == errors.min()))

        min_index = np.where((errors == errors.min())) # determining the # of bins with the smallest error_on_mean
        min_index = min_index[0][0]

        optimized_error = errors[min_index].round(3)
        optimized_bin_value = bins[min_index].round(3)
        optimized_amplitude = amplitudes[min_index].round(3)
        optimized_mean = means[min_index].round(3)
        optimized_std = stds[min_index].round(3)

        if verbose: 
            print(f"{data_name}")
            print(f"Accepted Bin #s: {bins}")
            print(" {:>13} | {:>8} | {:>11} | {:>8} | {:>8} ".format("Optimal Bin #", 
                                                                     "Error", 
                                                                     "Amplitude", 
                                                                     "Mean", 
                                                                     "Stddev"))
            print(" {:>13} | {:>8} | {:>11} | {:>8} | {:>8}".format(optimized_bin_value, 
                                                                     optimized_error, 
                                                                     optimized_amplitude, 
                                                                     optimized_mean, 
                                                                     optimized_std))
            print("")

        return optimized_bin_value, optimized_amplitude, optimized_mean, optimized_error, optimized_std

    def generate_tile_hists(self, verbose = False, show_hists = False):

        # if show_hists == True: 
        #   displays the individual hists on a that make up each fitting on the 3D plot

        ax = plt.figure(figsize = (10, 10)).add_subplot(projection = '3d')
        ax.view_init(elev=40, azim=-45, roll=0)

        starlist, idxs = self.extract_tile_stars(show_plot = True, verbose = verbose)
        colors = plt.cm.jet(np.linspace(0,1,len(starlist)))

        ax.set_xlabel(f"{starlist.columns[0]} - {starlist.columns[1]}")

        if self.catalogyname == self.catalog1name: 
            ax.set_ylabel(f"{starlist.columns[0]}")
        if self.catalogyname == self.catalog2name: 
            ax.set_ylabel(f"{starlist.columns[1]}")

        ax.set_zlabel("Frequency")
        ax.zaxis.labelpad=-0.01 # <- change the value here

        optimized_means = []
        optimized_mean_errors = []

        for i in range(len(starlist)): 

            if self.catalogyname == self.catalog2name: 
            
                ax.scatter(np.subtract(starlist[starlist.columns[0]][i], starlist[starlist.columns[1]][i]), 
                           starlist[starlist.columns[1]][i], 
                           zs = 0, zdir = 'z', label = 'CMD', 
                           color = colors[i], s = 0.3
                          )
                min_x = np.min(np.subtract(starlist[starlist.columns[0]][i], starlist[starlist.columns[1]][i]))

                # fitting 
                optimized_bin_value, optimized_amplitude, optimized_mean, optimized_error, optimized_std = self.optimize_tile_bin(
                                                                                                           starlist[starlist.columns[1]][i], 
                                                                                                           starlist.columns[1], 
                                                                                                           show_plot = False, 
                                                                                                           verbose = verbose
                )

                optimized_means.append(optimized_mean)
                optimized_mean_errors.append(optimized_error)

                bin_heights, bin_borders = np.histogram(starlist[starlist.columns[1]][i], bins=optimized_bin_value)
                bin_widths = np.diff(bin_borders)
                bin_centers = bin_borders[:-1] + bin_widths / 2
                
                t_gaussian = models.Gaussian1D(optimized_amplitude, optimized_mean, optimized_std)
                t_linear = models.Linear1D(100, 100)
                t_compound = t_gaussian + t_linear
                
                fit_t = fitting.LevMarLSQFitter()
                t = fit_t(t_compound, bin_centers, bin_heights)
                
                x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
                
                ax.plot(x_interval_for_fit, t(x_interval_for_fit), label='fit', c=colors[i], zs = min_x, zdir = 'x')

                if show_hists: 
                    ax.bar(bin_centers, bin_heights, width=bin_widths, label='histogram', 
                            zs = min_x, zdir = 'x', ec = (0, 0, 0, 0.5), facecolor = (0,0,0,0))
                    ax.figure.savefig(f"{self.image_path}{starlist.columns[1]}_optimized_with_{len(starlist)}_tiled_bins_with_histograms.png")

                else: 
                    ax.figure.savefig(f"{self.image_path}{starlist.columns[1]}_optimized_with_{len(starlist)}_tiled_bins.png")
                    
            if self.catalogyname == self.catalog1name: 
            
                ax.scatter(np.subtract(starlist[starlist.columns[0]][i], starlist[starlist.columns[1]][i]), 
                           starlist[starlist.columns[0]][i], 
                           zs = 0, zdir = 'z', label = 'CMD', 
                           color = colors[i], s = 0.3
                          )


                min_x = np.min(np.subtract(starlist[starlist.columns[0]][i], starlist[starlist.columns[1]][i]))

                # fitting 
                optimized_bin_value, optimized_amplitude, optimized_mean, optimized_error, optimized_std = self.optimize_tile_bin(
                                                                                                           starlist[starlist.columns[0]][i], 
                                                                                                           starlist.columns[0], 
                                                                                                           show_plot = False, 
                                                                                                           verbose = verbose
                )
              
                optimized_means.append(optimized_mean)
                optimized_mean_errors.append(optimized_error)

                bin_heights, bin_borders = np.histogram(starlist[starlist.columns[0]][i], bins=optimized_bin_value)
                bin_widths = np.diff(bin_borders)
                bin_centers = bin_borders[:-1] + bin_widths / 2
                
                t_gaussian = models.Gaussian1D(optimized_amplitude, optimized_mean, optimized_std)
                t_linear = models.Linear1D(100, 100)
                t_compound = t_gaussian + t_linear
                
                fit_t = fitting.LevMarLSQFitter()
                t = fit_t(t_compound, bin_centers, bin_heights)
                
                x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)

                ax.plot(x_interval_for_fit, t(x_interval_for_fit), label='fit', c=colors[i], zs = min_x, zdir = 'x')

                if show_hists: 
                    ax.bar(bin_centers, bin_heights, width=bin_widths, label='histogram', 
                            zs = min_x, zdir = 'x', ec = (0, 0, 0, 0.5), facecolor = (0,0,0,0))
                    ax.figure.savefig(f"{self.image_path}{starlist.columns[0]}_optimized_with_{len(starlist)}_tiled_bins_with_histograms.png")

                else: 
                    ax.figure.savefig(f"{self.image_path}{starlist.columns[0]}_optimized_with_{len(starlist)}_tiled_bins.png")

        return optimized_means, optimized_mean_errors

    def determine_tiled_slope(self, show_cmd = True, verbose = True): 

        optimized_means, optimized_mean_errors = self.generate_tile_hists(verbose = False)
        x = np.subtract(self.catalog1, self.catalog2)

        bins, segments = self.generate_tile_bins(show_plot = False)
        x_midpoints = [] 

        for i in range(len(bins)): 
            x_midpoints.append((bins[i][0][0] + bins[i][0][1]) / 2)

        fig, axis = plt.subplots(1, 1, figsize = (20, 10))
        check = False

        if self.catalogyname == self.catalog1name: 
            plt.scatter(x, self.catalog1, c = 'k', s = 0.1, alpha = 0.8)
            plt.xlabel(f"{self.catalog1name} - {self.catalog2name}")
            plt.ylabel(f"{self.catalog1name}")

            filename = f"{self.catalog1name}_{self.catalog2name}_{self.catalog1name}_{self.n}_tiled_bins_rcfit"
            check = True
        if self.catalogyname == self.catalog2name: 
            plt.scatter(x, self.catalog2, c = 'k', s = 0.1, alpha = 0.8)
            plt.xlabel(f"{self.catalog1name} - {self.catalog2name}")
            plt.ylabel(f"{self.catalog2name}")

            filename = f"{self.catalog1name}_{self.catalog2name}_{self.catalog2name}_{self.n}_tiled_bins_rcfit"
            check = True

        if not check:
            raise Exception("catalogyname must equal catalog1name or catalog2name")

        gradient, intercept, r_value, p_value, std_err = linregress(x_midpoints, optimized_means)

        line_orig = models.Linear1D(slope = gradient, intercept = intercept)
        fit = fitting.LevMarLSQFitter()
        line_init = models.Linear1D()

        fitted_line = fit(line_init, x_midpoints, optimized_means)

        plt.plot(x_midpoints, line_orig(x_midpoints), 'r-', label = 'linear fit')
        plt.scatter(x_midpoints, optimized_means, c = 'cyan', s = 20, marker = 'x', label = 'optimized means')
        
        plt.legend()
        plt.gca().invert_yaxis()
        plt.title(f"Fitted Slope: {fitted_line.slope.value.round(3)}")

        #plt.xlim(self.x_range[0], self.x_range[1])
        #plt.ylim(bins[self.n-1][1][1], bins[0][1][0])

        #plt.axline(self.parallel_cutoff1[0], self.parallel_cutoff1[1], c = 'r', label = 'provided rc cutoff', alpha = 0.5)
        #plt.axline(self.parallel_cutoff2[0], self.parallel_cutoff2[1], c = 'r', alpha = 0.5)

        plt.savefig(f"{self.image_path}{filename}.png")

        if verbose: 
            print(f"Mean Errors For The {self.n} Segments:")
            print(optimized_mean_errors)
            print(f"")
            print(fitted_line)

        return fitted_line.slope.value, fitted_line.intercept.value

    def overplot_isochrones(self, logAge, AKs, AKs_step, dist, 
                            height, metallicity, filters, iso_dir,
                            show_cmd = True, verbose = True): 

        # filters = [filt1name, filt2name] - refer to SPISEA documentation, filter names are specific and unique. 

        slope, intercept = self.determine_tiled_slope(verbose = False)
        check = False

        catalog1 = np.array(self.catalog1)
        catalog2 = np.array(self.catalog2)

        x = np.subtract(catalog1, catalog2)

        red_law = reddening.RedLawFritz11(scale_lambda = 2.166) 
        evo_model = evolution.MISTv1()                  # evolution model
        atm_func = atmospheres.get_merged_atmosphere    # atmospheric model

        AKs2 = AKs + AKs_step
        AKs3 = AKs2 + AKs_step
        AKs4 = AKs3 + AKs_step
        AKs5 = AKs4 + AKs_step

        my_iso = isochrone = synthetic.IsochronePhot(logAge, AKs, dist, 
                                           metallicity, evo_model, atm_func, red_law = red_law, 
                                           filters = filters, iso_dir = iso_dir)
        idx = np.where( abs(my_iso.points['mass'] - 1.0) == min(abs(my_iso.points['mass'] - 1.0)) )[0]

        AKs = AKs2
        my_iso2 = isochrone = synthetic.IsochronePhot(logAge, AKs, dist, 
                                           metallicity, evo_model, atm_func, red_law = red_law, 
                                           filters = filters, iso_dir = iso_dir)
        idx2 = np.where( abs(my_iso2.points['mass'] - 1.0) == min(abs(my_iso2.points['mass'] - 1.0)) )[0]

        AKs = AKs3
        my_iso3 = isochrone = synthetic.IsochronePhot(logAge, AKs, dist, 
                                           metallicity, evo_model, atm_func, red_law = red_law, 
                                           filters = filters, iso_dir = iso_dir)
        idx3 = np.where( abs(my_iso3.points['mass'] - 1.0) == min(abs(my_iso3.points['mass'] - 1.0)) )[0]

        AKs = AKs4
        my_iso4 = isochrone = synthetic.IsochronePhot(logAge, AKs, dist, 
                                           metallicity, evo_model, atm_func, red_law = red_law, 
                                           filters = filters, iso_dir = iso_dir)
        idx4 = np.where( abs(my_iso4.points['mass'] - 1.0) == min(abs(my_iso4.points['mass'] - 1.0)) )[0]

        AKs = AKs5
        my_iso5 = isochrone = synthetic.IsochronePhot(logAge, AKs, dist, 
                                           metallicity, evo_model, atm_func, red_law = red_law, 
                                           filters = filters, iso_dir = iso_dir)
        idx5 = np.where( abs(my_iso5.points['mass'] - 1.0) == min(abs(my_iso5.points['mass'] - 1.0)) )[0]

        
        fig, axis = plt.subplots(1, 1, figsize = (20, 10))
        plt.gca().invert_yaxis()

        if self.catalogyname == self.catalog1name: 
            check = True
            y = np.array(self.catalog1)

            plt.scatter(x, y, c = 'k', s = 0.05)

            plt.plot(my_iso.points[''+my_iso.points.keys()[8]] - my_iso.points[''+my_iso.points.keys()[9]],
                     my_iso.points[''+my_iso.points.keys()[8]], 'r-', label='_nolegend_')
            plt.plot(my_iso2.points[''+my_iso2.points.keys()[8]] - my_iso2.points[''+my_iso2.points.keys()[9]],
                     my_iso2.points[''+my_iso2.points.keys()[8]], 'r-', label='_nolegend_')
            plt.plot(my_iso3.points[''+my_iso3.points.keys()[8]] - my_iso3.points[''+my_iso3.points.keys()[9]],
                     my_iso3.points[''+my_iso3.points.keys()[8]], 'r-', label='_nolegend_')
            plt.plot(my_iso4.points[''+my_iso4.points.keys()[8]] - my_iso4.points[''+my_iso4.points.keys()[9]],
                     my_iso4.points[''+my_iso4.points.keys()[8]], 'r-', label='_nolegend_')
            plt.plot(my_iso5.points[''+my_iso5.points.keys()[8]] - my_iso5.points[''+my_iso5.points.keys()[9]],
                     my_iso5.points[''+my_iso5.points.keys()[8]], 'r-', label='_nolegend_')
            
            plt.axline((my_iso5.points[''+my_iso5.points.keys()[8]][idx5][0]
                        - my_iso5.points[''+my_iso5.points.keys()[9]][idx5][0], 
                        my_iso5.points[''+my_iso5.points.keys()[8]][idx5][0] + height), 
                        (my_iso.points[''+my_iso.points.keys()[8]][idx][0]
                        - my_iso.points[''+my_iso.points.keys()[9]][idx][0],
                        my_iso.points[''+my_iso.points.keys()[8]][idx][0] + height), 
                        color = 'aqua', label = "isochrone extinction vector")

            plt.xlabel(f"{self.catalog1name} - {self.catalog2name}")
            plt.ylabel(f"{self.catalog1name}")
            filename = f"extinction_vec_{self.catalog1name}_{self.catalog2name}_{self.catalog1name}_{self.n}_tiled_bins" 

        if self.catalogyname == self.catalog2name: 
            check = True
            y = np.array(self.catalog2)

            plt.scatter(x, y, c = 'k', s = 0.05)

            plt.plot(my_iso.points[''+my_iso.points.keys()[8]] - my_iso.points[''+my_iso.points.keys()[9]],
                     my_iso.points[''+my_iso.points.keys()[9]], 'r-', label='_nolegend_')
            plt.plot(my_iso2.points[''+my_iso2.points.keys()[8]] - my_iso2.points[''+my_iso2.points.keys()[9]],
                     my_iso2.points[''+my_iso2.points.keys()[9]], 'r-', label='_nolegend_')
            plt.plot(my_iso3.points[''+my_iso3.points.keys()[8]] - my_iso3.points[''+my_iso3.points.keys()[9]],
                     my_iso3.points[''+my_iso3.points.keys()[9]], 'r-', label='_nolegend_')
            plt.plot(my_iso4.points[''+my_iso4.points.keys()[8]] - my_iso4.points[''+my_iso4.points.keys()[9]],
                     my_iso4.points[''+my_iso4.points.keys()[9]], 'r-', label='_nolegend_')
            plt.plot(my_iso5.points[''+my_iso5.points.keys()[8]] - my_iso5.points[''+my_iso5.points.keys()[9]],
                     my_iso5.points[''+my_iso5.points.keys()[9]], 'r-', label='_nolegend_')
            
            plt.axline((my_iso5.points[''+my_iso5.points.keys()[8]][idx5][0]
                        - my_iso5.points[''+my_iso5.points.keys()[9]][idx5][0], 
                        my_iso5.points[''+my_iso5.points.keys()[9]][idx5][0] + height), 
                        (my_iso.points[''+my_iso.points.keys()[8]][idx][0]
                        - my_iso.points[''+my_iso.points.keys()[9]][idx][0],
                        my_iso.points[''+my_iso.points.keys()[9]][idx][0] + height), 
                        color = 'aqua', label = "isochrone extinction vector")

            plt.xlabel(f"{self.catalog1name} - {self.catalog2name}")
            plt.ylabel(f"{self.catalog2name}")
            filename = f"extinction_vec_{self.catalog1name}_{self.catalog2name}_{self.catalog2name}_{self.n}_tiled_bins"    

        plt.axline((0, intercept), slope = slope, c = 'r', label = 'derived extinction vector') 
        plt.legend()
        plt.savefig(f"{self.image_path}{filename}.png")   

        if not check: 
            raise Exception("catalogyname must equal catalog1name or catalog2name")

        return 



def riemann_result(catalog, catalog1filt, catalog2filt, catalogyfilt, 
                   region1, region2, regiony, filters, 
                   parallel_cutoff1, parallel_cutoff2, 
                   x_range, n, hists = False,
                   catalog1zp = None, catalog2zp = None):
    
    catalog1, catalog2, catalog1error, catalog2error = get_matches(catalog, catalog1filt, region1, catalog2filt, region2)

    if catalog1zp: 
        catalog1 += catalog1zp
    if catalog2zp: 
        catalog2 += catalog2zp

    riemann_class = Red_Clump_Analysis_vRiemann(catalog1, catalog2, 
                                catalog1name = catalog1filt, 
                                catalog2name = catalog2filt, 
                                catalogyname = catalogyfilt, 
                                parallel_cutoff1 = parallel_cutoff1, 
                                parallel_cutoff2 = parallel_cutoff2, 
                                x_range = x_range,
                                n = n,
                                image_path = f"/Users/devaldeliwala/research/work/plots&data/rc_analysis_plots/{region1}_{catalog1filt}-{region2}_{catalog2filt}_vs{catalogyfilt}/")

    riemann_class.display_cutoffs(verbose = True)
    riemann_class.generate_tile_bins()
    riemann_class.generate_tile_hists(verbose = True, show_hists = hists)
    riemann_class.determine_tiled_slope(show_cmd = True)
    riemann_class.overplot_isochrones(filters = filters, logAge = np.log(10**9), 
                       AKs = 2, AKs_step = 0.25, dist = 8000, metallicity = -0.3, height = -12.3,
                       iso_dir = "/Users/devaldeliwala/research/work/plots&data/isochrone_plots&data/plots/")


fits ='catalogs/dr2/jwst_init_NRCB.fits'
catalog = Table.read(fits, format='fits')

"""
riemann_result(catalog, 'F115W', 'F212N', 'F115W', 'NRCB2', 'NRCB2', 'NRCB2',
              ['jwst,F115W', 'jwst,F212N'], 
              [(6.3, 21.3), (9, 25.2)], [(6.3, 22), (9, 25.9)], 
              [6, 9], 10, hists = True, catalog1zp = 25.95, catalog2zp = 22.15)
riemann_result(catalog, 'F115W', 'F212N', 'F115W', 'NRCB3', 'NRCB3', 'NRCB3',
              ['jwst,F115W', 'jwst,F212N'], 
              [(6.3, 21.3), (9, 25.2)], [(6.3, 22), (9, 25.9)], 
              [6, 9], 10, hists = True, catalog1zp = 25.95, catalog2zp = 22.15)
riemann_result(catalog, 'F115W', 'F212N', 'F115W', 'NRCB4', 'NRCB4', 'NRCB4',
              ['jwst,F115W', 'jwst,F212N'], 
              [(6.3, 21.3), (9, 25.2)], [(6.3, 22), (9, 25.9)], 
              [6, 9], 10, hists = True, catalog1zp = 26.09, catalog2zp = 22.23)
riemann_result(catalog, 'F115W', 'F212N', 'F115W', 'NRCB2', 'NRCB2', 'NRCB2',
              ['jwst,F115W', 'jwst,F212N'], 
              [(6.3, 21.3), (9, 25.2)], [(6.3, 22), (9, 25.9)], 
              [6, 9], 10, hists = False, catalog1zp = 25.95, catalog2zp = 22.15)
riemann_result(catalog, 'F115W', 'F212N', 'F115W', 'NRCB3', 'NRCB3', 'NRCB3',
              ['jwst,F115W', 'jwst,F212N'], 
              [(6.3, 21.3), (9, 25.2)], [(6.3, 22), (9, 25.9)], 
              [6, 9], 10, hists = False, catalog1zp = 25.95, catalog2zp = 22.15)
riemann_result(catalog, 'F115W', 'F212N', 'F115W', 'NRCB4', 'NRCB4', 'NRCB4',
              ['jwst,F115W', 'jwst,F212N'], 
              [(6.3, 21.3), (9, 25.2)], [(6.3, 22), (9, 25.9)], 
              [6, 9], 10, hists = False, catalog1zp = 26.09, catalog2zp = 22.23)

riemann_result(catalog, 'F115W', 'F212N', 'F115W', 'NRCB2', 'NRCB2', 'NRCB2',
              ['jwst,F115W', 'jwst,F212N'], 
              [(6.3, 21.3), (9, 25.2)], [(6.3, 22), (9, 25.9)], 
              [6, 9], 15, hists = True, catalog1zp = 25.95, catalog2zp = 22.15)
riemann_result(catalog, 'F115W', 'F212N', 'F115W', 'NRCB3', 'NRCB3', 'NRCB3',
              ['jwst,F115W', 'jwst,F212N'], 
              [(6.3, 21.3), (9, 25.2)], [(6.3, 22), (9, 25.9)], 
              [6, 9], 15, hists = True, catalog1zp = 25.95, catalog2zp = 22.15)
riemann_result(catalog, 'F115W', 'F212N', 'F115W', 'NRCB4', 'NRCB4', 'NRCB4',
              ['jwst,F115W', 'jwst,F212N'], 
              [(6.3, 21.3), (9, 25.2)], [(6.3, 22), (9, 25.9)], 
              [6, 9], 15, hists = True, catalog1zp = 26.09, catalog2zp = 22.23)
riemann_result(catalog, 'F115W', 'F212N', 'F115W', 'NRCB2', 'NRCB2', 'NRCB2',
              ['jwst,F115W', 'jwst,F212N'], 
              [(6.3, 21.3), (9, 25.2)], [(6.3, 22), (9, 25.9)], 
              [6, 9], 15, hists = False, catalog1zp = 25.95, catalog2zp = 22.15)
riemann_result(catalog, 'F115W', 'F212N', 'F115W', 'NRCB3', 'NRCB3', 'NRCB3',
             ['jwst,F115W', 'jwst,F212N'], 
              [(6.3, 21.3), (9, 25.2)], [(6.3, 22), (9, 25.9)], 
              [6, 9], 15, hists = False, catalog1zp = 25.95, catalog2zp = 22.15)
riemann_result(catalog, 'F115W', 'F212N', 'F115W', 'NRCB4', 'NRCB4', 'NRCB4',
              ['jwst,F115W', 'jwst,F212N'], 
              [(6.3, 21.3), (9, 25.2)], [(6.3, 22), (9, 25.9)], 
              [6, 9], 15, hists = False, catalog1zp = 26.09, catalog2zp = 22.23)
"""
"""
riemann_result(catalog, 'F115W', 'F212N', 'F212N', 'NRCB2', 
              "NRCB2 F115W", "NRCB2 F212N", "NRCB2 F212N", ['jwst,F115W', 'jwst,F212N'], 
              [(6.2, 15.4), (7.9, 16.2)], [(6.65, 15.3), (8.35, 16.1)], 
              [6.4, 9.2], 10, hists = True, catalog1zp = 25.95, catalog2zp = 22.15)
riemann_result(catalog, 'F115W', 'F212N', 'F212N', 'NRCB3', 
              "NRCB3 F115W", "NRCB3 F212N", "NRCB3 F212N", ['jwst,F115W', 'jwst,F212N'], 
              [(6.2, 15.4), (7.9, 16.2)], [(6.65, 15.3), (8.35, 16.1)], 
              [6.4, 9.2], 10, hists = True, catalog1zp = 25.95, catalog2zp = 22.15)
riemann_result(catalog, 'F115W', 'F212N', 'F212N', 'NRCB4', 
              "NRCB4 F115W", "NRCB4 F212N", "NRCB4 F212N", ['jwst,F115W', 'jwst,F212N'], 
              [(6.2, 15.4), (7.9, 16.2)], [(6.65, 15.3), (8.35, 16.1)], 
              [6.4, 9.2], 10, hists = True, catalog1zp = 26.09, catalog2zp = 22.12)
riemann_result(catalog, 'F115W', 'F212N', 'F212N', 'NRCB2', 
              "NRCB2 F115W", "NRCB2 F212N", "NRCB2 F212N", ['jwst,F115W', 'jwst,F212N'], 
              [(6.2, 15.4), (7.9, 16.2)], [(6.65, 15.3), (8.35, 16.1)], 
              [6.4, 9.2], 10, hists = False, catalog1zp = 25.95, catalog2zp = 22.23)
riemann_result(catalog, 'F115W', 'F212N', 'F212N', 'NRCB3', 
              "NRCB3 F115W", "NRCB3 F212N", "NRCB3 F212N", ['jwst,F115W', 'jwst,F212N'], 
              [(6.2, 15.4), (7.9, 16.2)], [(6.65, 15.3), (8.35, 16.1)], 
              [6.4, 9.2], 10, hists = False, catalog1zp = 25.95, catalog2zp = 22.15)
riemann_result(catalog, 'F115W', 'F212N', 'F212N', 'NRCB4', 
              "NRCB4 F115W", "NRCB4 F212N", "NRCB4 F212N", ['jwst,F115W', 'jwst,F212N'], 
              [(6.2, 15.4), (7.9, 16.2)], [(6.65, 15.3), (8.35, 16.1)], 
              [6.4, 9.2], 10, hists = False, catalog1zp = 26.09, catalog2zp = 22.23)

riemann_result(catalog, 'F115W', 'F212N', 'F212N', 'NRCB2', 
              "NRCB2 F115W", "NRCB2 F212N", "NRCB2 F212N", ['jwst,F115W', 'jwst,F212N'], 
              [(6.2, 15.4), (7.9, 16.2)], [(6.65, 15.3), (8.35, 16.1)], 
              [6.4, 9.2], 15, hists = True, catalog1zp = 25.95, catalog2zp = 22.15)
riemann_result(catalog, 'F115W', 'F212N', 'F212N', 'NRCB3', 
              "NRCB3 F115W", "NRCB3 F212N", "NRCB3 F212N", ['jwst,F115W', 'jwst,F212N'], 
              [(6.2, 15.4), (7.9, 16.2)], [(6.65, 15.3), (8.35, 16.1)], 
              [6.4, 9.2], 15, hists = True, catalog1zp = 25.95, catalog2zp = 22.15)
riemann_result(catalog, 'F115W', 'F212N', 'F212N', 'NRCB4', 
              "NRCB4 F115W", "NRCB4 F212N", "NRCB4 F212N", ['jwst,F115W', 'jwst,F212N'], 
              [(6.2, 15.4), (7.9, 16.2)], [(6.65, 15.3), (8.35, 16.1)], 
              [6.4, 9.2], 15, hists = True, catalog1zp = 26.09, catalog2zp = 22.23)
riemann_result(catalog, 'F115W', 'F212N', 'F212N', 'NRCB2', 
              "NRCB2 F115W", "NRCB2 F212N", "NRCB2 F212N", ['jwst,F115W', 'jwst,F212N'], 
              [(6.2, 15.4), (7.9, 16.2)], [(6.65, 15.3), (8.35, 16.1)], 
              [6.4, 9.2], 15, hists = False, catalog1zp = 25.95, catalog2zp = 22.15)
riemann_result(catalog, 'F115W', 'F212N', 'F212N', 'NRCB3', 
              "NRCB3 F115W", "NRCB3 F212N", "NRCB3 F212N", ['jwst,F115W', 'jwst,F212N'], 
              [(6.2, 15.4), (7.9, 16.2)], [(6.65, 15.3), (8.35, 16.1)], 
              [6.4, 9.2], 15, hists = False, catalog1zp = 25.95, catalog2zp = 22.15)
riemann_result(catalog, 'F115W', 'F212N', 'F212N', 'NRCB4', 
              "NRCB4 F115W", "NRCB4 F212N", "NRCB4 F212N", ['jwst,F115W', 'jwst,F212N'], 
              [(6.2, 15.4), (7.9, 16.2)], [(6.65, 15.3), (8.35, 16.1)], 
              [6.4, 9.2], 15, hists = False, catalog1zp = 26.09, catalog2zp = 22.23)
"""

# Table of Relevant Zeropoints
"""
|  CLEAR+F115W |    NRCB1 |    25.92 |  
|  CLEAR+F115W |    NRCB2 |    25.95 |  
|  CLEAR+F115W |    NRCB3 |    25.95 |  
|  CLEAR+F115W |    NRCB4 |    26.09 | 

|  CLEAR+F212N |    NRCB1 |    22.12 |   
|  CLEAR+F212N |    NRCB2 |    22.15 |   
|  CLEAR+F212N |    NRCB3 |    22.15 |   
|  CLEAR+F212N |    NRCB4 |    22.23 |

| F323N+F322W2 | NRCBLONG |    21.14 |

| F405N+F444W  | NRCBLONG |     20.91 |
"""
