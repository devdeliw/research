from color_magnitude_diagrams import * 
from catalog_helper_functions import * 
from matplotlib.patches import Rectangle
import math

class Red_Clump_Analysis: 
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

        # extract_stars(self, show_plot = True, verbose = False):

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

         # generate_hits(starlist, path, verbose = False):

         Performs `optimize_bin` method for each bin in the RC bar established from 
         `divide_cutoff` and returns a 3D Plot of all the optimized fittings. 

         Returns all the means from the compound fitting, and their corresponding 
         error. 

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
                    plt.savefig(f"{self.image_path}{filename}.png")

                    plt.xlabel(f"{self.catalog1name} - {self.catalog2name}")
                    plt.ylabel(f"{self.catalog1name}")

                    check = True

                if self.catalogyname == self.catalog2name: 
                    plt.scatter(np.subtract(self.catalog1, self.catalog2), self.catalog2, 
                                c = 'k', s = 0.5)

                    for segment in segments:
                        axis.add_patch(segment)

                    filename = f"{self.catalog1name}_{self.catalog2name}_{self.catalog2name}_{self.n}_bins"
                    plt.savefig(f"{self.image_path}{filename}.png")

                    plt.xlabel(f"{self.catalog1name} - {self.catalog2name}")
                    plt.ylabel(f"{self.catalog2name}")  

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
















fits ='catalogs/dr2/jwst_init_NRCB.fits'
catalog = Table.read(fits, format='fits')

catalog1, catalog2, N1_f115w_vf212n_me, N1_f212n_vf115w_me = get_matches(catalog, 'F115W', 'NRCB1', 'F212N', 'NRCB1')
catalog1 += 25.92
catalog2 += 22.12

RC = Red_Clump_Analysis(catalog1,  catalog2, 
                   catalog1name = "NRCB1 F115W", 
                   catalog2name = "NRCB1 F212N", 
                   catalogyname = "NRCB1 F212N", 
                   xlim = (5.8, 9.3), 
                   ylim = (14.5, 16.9), 
                   n = 10, 
                   image_path = "/Users/devaldeliwala/research/work/plots&data/rc_analysis_plots/", 
                   matched = True)

RC.divide_cutoff()
RC.extract_stars(verbose = True)
RC.generate_hists(verbose = True, show_hists = True)