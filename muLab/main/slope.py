import os 
import numpy as np 
import pandas as pd 
import flystar 

from astropy.table import Table
from scipy.stats import gaussian_kde
from catalog_helper_functions import get_matches 

import scipy.ndimage as ndimage
import matplotlib.pyplot as plt 
import seaborn as sns


class Diagram: 
    """
    Standalone class for plotting color-magnitude diagrams. 

    Args: 
        mag1 (array-like): magnitudes for first wavelength catalog.
        mag2 (array-like): magnitudes for second wavelength catalog.
        magy (array-like): magnitudes for wavelength catalog on y axis.
        mag1name, mag2name, magyname (str): names for respective catalogs. 
        image_dir (str): directory to place plots.

    """

    def __init__(
            self, 
            mag1, mag2, magy, 
            mag1name, mag2name, magyname, 
            image_dir='./images/cmds/',
    ): 
        self.mag1 = mag1 
        self.mag2 = mag2 
        self.magy = magy 
        self.mag1name = mag1name 
        self.mag2name = mag2name 
        self.magyname = magyname 
        self.image_dir = image_dir 
        return 

    def render(
        self, 
        color_by_density=True,
        render_image=True, 
        plot_by_mags=True,
        save_image=True,
        color=None, 
        magy=None
    ): 
        # if plot_by_mags: renders a CMD using `self.mag1/2/y`
        # else renders using `color` (x) vs. `self.magy` (y) variables. 

        if plot_by_mags: 
            # Renders a `mag1`-`mag2` vs. `magy` diagram.
            x = np.subtract(self.mag1, self.mag2) 
            y = np.array(self.magy)
        else: 
            try:
                x = color 
                y = np.array(magy)
            except ValueError: 
                print(
                    '`color` and `magy` undefined. \
                    required for plot_by_mags=False.'
                )

        xy = np.vstack([x, y])
        density = gaussian_kde(xy)(xy) 

        data = pd.DataFrame(
            {
                'color': x, 
                'mag': y, 
                'density': density
            }
        )

        # Plotting
        if render_image:
            fig, axis = plt.subplots(1, 1, figsize=(10, 8))

            if color_by_density:
                hue = 'density'
            else: 
                hue = None

            scatter = sns.scatterplot(
                data=data, 
                x='color',
                y='mag',
                hue=hue, 
                palette='viridis', 
                marker='+',
                legend=False
            )

            plt.gca().invert_yaxis()
            plt.title(
                f'{self.mag1name} - {self.mag2name} vs. {self.magyname}', 
                fontsize=16)
            plt.xlabel(
                f'{self.mag1name} - {self.mag2name} (mag)', 
                fontsize=14)
            plt.ylabel(
                f'{self.magyname} (mag)', 
                fontsize=14)

            if color_by_density: 
                plt.colorbar(scatter.collections[0], label='Density')
            plt.tight_layout()

            if not os.path.exists(self.image_dir): 
                os.makedirs(self.image_dir)

            if plot_by_mags:
                filename = f'{self.mag1name}-{self.mag2name}_{self.magyname}'
            else: 
                filename = f'{self.mag1name}-{self.mag2name}_{self.magyname}_bbox'
            if save_image: 
                plt.savefig(f'{self.image_dir}{filename}.png', dpi=300)
            return plt

        self.data = data
        return

    def cutoff(
        self, 
        color_min, color_max, 
        mag_min, mag_max,
        render_image=False,
    ): 
        """
        Args:
            color_min, color_max (float): color-space (x) cutoffs.
            mag_min, mag_max (float): y-magnitude cutoffs.
            render (bool): whether to render CMD of cutoff-portion. 
        Returns: 
            x (np array): colors of stars within bbox.
            y (np array): y-mags of stars within bbox. 

        """
        
        # performs a cutoff of the CMD based on pre-defined bbox.
        self.render(render_image=False)

        x = self.data['color']
        y = self.data['mag'] 

        x_mask = (x >= color_min) & (x <= color_max) 
        y_mask = (y >= mag_min) & (y <= mag_max)

        mask = x_mask & y_mask
        x = x[mask]
        y = y[mask]

        if render_image: 
            self.render(plot_by_mags=False, color=x, magy=y) 
        return x, y
        
                

class RidgeTracing(Diagram): 
    """
        Performs ridge tracing algorithm to calculate 
        slope of Red Clump (RC) bar within `mag1`-`mag2` 
        vs. `magy` CMD. The RC bar is selected via the 
        `bbox` parameter. 

        Args: 
            mag1, mag2, magy, 
            mag1name, mag2name, magyname: Same as in `Diagram`. 
            bbox (pd DataFrame): defines bounding box of RC bar. 
                pd.DataFrame({
                    'color': [color_min, color_max], 
                    'mag': [mag_min, mag_max], 
                })
            image_dir (str): defines plot placement directory. 

    """

    def __init__(
        self, 
        mag1, mag2, magy, 
        mag1name, mag2name, magyname, 
        bbox,
        N_color_bins=100, N_mag_bins=100, 
        image_dir='./images/ridge/', 
    ): 
        self.mag1 = mag1 
        self.mag2 = mag2 
        self.magy = magy
        self.mag1name = mag1name 
        self.mag2name = mag2name
        self.magyname = magyname 
        self.N_color_bins = N_color_bins
        self.N_mag_bins = N_mag_bins
        self.bbox = bbox 
        self.image_dir = image_dir
        return 

    def density_kernel(
            self, 
            smooth=True
    ):
        color_min, color_max = self.bbox['color']
        mag_min, mag_max = self.bbox['mag']

        color, magnitude = self.cutoff(
                color_min, color_max, 
                mag_min, mag_max, 
        )

        color_bins = np.linspace(color_min, color_max, self.N_color_bins) 
        mag_bins = np.linspace(mag_min, mag_max, self.N_mag_bins)

        H, xedges, yedges = np.histogram2d(
                color, magnitude, bins=[color_bins, mag_bins]
        )

        if smooth: 
            H = ndimage.gaussian_filter(H, sigma=5.0)

        self.color_min, self.color_max = color_min, color_max 
        self.mag_min, self.mag_max = mag_min, mag_max
        self.xedges, self.yedges = xedges, yedges 
        self.color = color 
        self.magnitude = magnitude
        return H, xedges, yedges 

    def gradient(self): 
        H, _, _ = self.density_kernel(smooth=True)
        self.grad_x = ndimage.sobel(H, axis=1)
        self.grad_y = ndimage.sobel(H, axis=0)
        self.H_smooth = H
        return 

    # Helper Functions for Render Ridge
    def color_to_idx(self, c, offset):
        j = np.searchsorted(self.xedges, c) - 1
        return j-offset

    def mag_to_idx(self, m):
        i = np.searchsorted(self.yedges, m) - 1
        return i


    def render_ridge(
        self,
        ansatz,
        step_size=1, 
        max_iterations=100, 
        render_image=True, 
    ): 
        """
        Args: 
            ansatz (list of (x,y)): list of ansatz coordinates that are obviously 
                                    in the RC bar.
            step_size (float): momentum factor.
            max_iterations (int): max interations for ridge off a starting point. 

        """

        self.gradient() # calculate gradients for all histogram bins
        all_paths = [] 

        for (start_x, start_y) in ansatz: 
            x, y = self.color_to_idx(start_x, offset=9), self.mag_to_idx(start_y)
            path = [(x, y)] # initialize path to store ridge 

            for _ in range(max_iterations): 
                i = int(round(y))
                j = int(round(x))

                if (
                    i < 0 or i >= self.N_mag_bins or 
                    j < 0 or j >= self.N_color_bins
                ): break 

                gx = self.grad_x[i, j]
                gy = self.grad_y[i, j]

                # magnitude of gradient vector 
                grad_mag = np.sqrt(gx**2 + gy**2) 

                # end ridge if gradient vector becomes very small
                # (we've hit a peak)
                if grad_mag < 1e-7: 
                    break 

                x += step_size * (gx / grad_mag)
                y += step_size * (gy / grad_mag)

                path.append((x, y))

            all_paths.append(path)

        # Figure plotting 
        # ---------------

        plt = self.render(
            plot_by_mags=False, 
            color=self.color,
            magy=self.magnitude, 
            save_image=False
        )

        for path in all_paths: 
            if len(path) == 0: 
                continue 

            x_indices, y_indices = zip(*path) 
            x_indices = np.array(x_indices)
            y_indices = np.array(y_indices) 

            color_vals = self.color_min + x_indices * (self.color_max - self.color_min) / self.N_color_bins
            mag_vals   = self.mag_min + y_indices * (self.mag_max - self.mag_min) / self.N_mag_bins

            plt.plot(color_vals, mag_vals, 'r-', linewidth=2)

        if render_image:
            filename = f'{self.mag1name}-{self.mag2name}vs.{self.magyname}_RIDGE'
            if not os.path.exists(self.image_dir): 
                os.makedirs(self.image_dir)
            plt.savefig(f'{self.image_dir}{filename}.png', dpi=300) 
        return 
        




        




        








if __name__ == '__main__': 
    catalog_dir = './fits/'
    file_name = 'jwst_init_NRCB.fits'

    image_dir = './images/'

    catalog = Table.read(
        os.path.join(catalog_dir, file_name), 
        format='fits'
    )

    mf115w, mf212n, mef115w, mef212n = get_matches(
        catalog, 
        'F115W', 'NRCB1', 
        'F212N', 'NRCB1', 
    )

    rc_bbox = pd.DataFrame(
        {
            'color': [5.9, 8.4], 
            'mag': [20.7, 24.1], 
        }
    )

    ansatz = [(6.7, 21.8), (7.8, 23.2), (7.1, 22.2), (7.17, 22.53)]


    RidgeTracing(
            mf115w, mf212n, mf115w,
            'F115W', 'F212N', 'F115W', 
            rc_bbox, 
    ).render_ridge(ansatz=ansatz)





