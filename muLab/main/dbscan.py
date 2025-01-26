import os 
import numpy as np 
import matplotlib.pyplot as plt 

from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from astropy.table import Table 
from catalog_helper_functions import get_matches 
from slope import Diagram

class LocateRC(Diagram): 
    """
    Class that performs `Density-Based Spatial Clustering of Applications with Noise`
    (DBSCAN) on a starcluster's color-magnitude diagram to locate the Red Clump (RC)
    cluster without performing any cutoffs. 

    The color magnitude diagram is defined as 
    `mag1` - `mag2` vs. `magy`. 

    Args: 
        mag1 (array-like): magnitudes for first wavelength catalog.
        mag2 (array-like): magnitudes for second wavelength catalog.
        magy (array-like): magnitudes for wavelength catalog on y axis.
        mag1name, mag2name, magyname (str): names for respective catalogs. 
        region (str): region name for image directory
        image_dir (str): directory to place plots.

    """

    def __init__(
        self, 
        mag1, mag2, magy, 
        mag1name, mag2name, magyname, 
        region, 
        image_dir='./images/DBSCAN/', 
    ): 
        self.mag1 = mag1
        self.mag2 = mag2 
        self.magy = magy 
        self.mag1name = mag1name 
        self.mag2name = mag2name 
        self.magyname = magyname 
        self.image_dir = f'{image_dir}{region}/{mag1name}-{mag2name}/'
        return 
    
    def dbscan(self, eps, min_samples, render_image=True): 
        self.render(
            color_by_density=False,
            render_image=False, 
            save_image=False,
        )

        # extract color and magnitude coordinates from CMD
        color = self.data['color']
        mag = self.data['mag'] 

        data = np.vstack([color, mag]).T
        db = DBSCAN(eps=eps, min_samples=min_samples)

        # execute DBSCAN algorithm 
        labels = db.fit_predict(data)
        
        if render_image: 
            _, _ = plt.subplots(1, 1, figsize=(10, 8))
            scatter = plt.scatter(color, mag, c=labels, cmap='rainbow', s=5)
            plt.xlabel(f'{self.mag1name} - {self.mag2name}', fontsize=14)
            plt.ylabel(f'{self.magyname}', fontsize=14)
            plt.title(
                f'{self.mag1name} - {self.mag2name} vs. {self.magyname} DBSCAN',
                fontsize=14,
            )
            plt.colorbar(scatter, label='Cluster Label')
            plt.gca().invert_yaxis()

            filename = f'{self.mag1name}-{self.mag2name}_{self.magyname}_DBSCAN'
            if not os.path.exists(self.image_dir): 
                os.makedirs(self.image_dir)
            plt.savefig(f'{self.image_dir}{filename}.png', dpi=300)

        return color, mag, labels

    def isolate(self, eps=0.1, min_samples=50, render_rc=True): 
        # Isolates the RC bar rest of the CMD using one heuristic -- 
        # -- the rc bar is above the ms with inverted y-axis. 
        color, mag, labels = self.dbscan(eps, min_samples, render_image=True)
        unique_labels = list(set(labels)) 

        label_bboxes = [] # store the bboxes for each subcluster 
        for label in unique_labels: 
            rc_idxs = (labels == label)
            rc_color = color[rc_idxs] 
            rc_mag = mag[rc_idxs]

            bbox = [
                label, 
                (min(rc_color), max(rc_color)), 
                (min(rc_mag), max(rc_mag)),
            ]
            label_bboxes.append([bbox, len(rc_color)]) 

        # define the RC bar using basic heuristics 
        mag_ansatz = max(mag)
        col_ansatz = max(color) - min(color)

        rc_label = None
        for bbox, num in label_bboxes: 
            label = bbox[0] 
            min_color, max_color = bbox[1] 
            min_mag, _ = bbox[2]

            if (
                max_color-min_color < col_ansatz and 
                min_mag < mag_ansatz and 
                num >= 100 
            ): 
                mag_ansatz = min_mag
                col_ansatz = max_color-min_color
                rc_label = label 

        # Final Red Clump star coordinates 
        rc_idxs = (labels == rc_label)
        rc_color = color[rc_idxs] 
        rc_mag = mag[rc_idxs]

        # Red clump bar plotting 
        if render_rc: 
            _, _ = plt.subplots(1, 1, figsize=(10, 10))
            xy = np.vstack([rc_color, rc_mag]) 
            density = gaussian_kde(xy)(xy) 

            plt.scatter(rc_color, rc_mag, c=density, marker='D', s=15)
            plt.xlabel(f'{self.mag1name} - {self.mag2name}', fontsize=14)
            plt.ylabel(f'{self.magyname}', fontsize=14)
            plt.title(
                f'{self.mag1name} - {self.mag2name} vs. {self.magyname} Red Clump (RC)', 
                fontsize=14
            )
            plt.gca().invert_yaxis()

            filename = f'{self.mag1name}-{self.mag2name}_{self.magyname}_RC'
            if not os.path.exists(self.image_dir): 
                os.makedirs(self.image_dir) 
            plt.savefig(f'{self.image_dir}{filename}.png', dpi=300)

        return rc_color, rc_mag





            

        



if __name__ == '__main__': 
    catalog_dir = './fits/'
    file_name = 'jwst_init_NRCB.fits'

    catalog = Table.read(
        os.path.join(catalog_dir, file_name), 
        format='fits'
    )

    mf115w, mf212n, mef115w, mef212n = get_matches(
        catalog, 
        'F115W', 'NRCB1', 
        'F212N', 'NRCB1', 
    )

    rc_color, rc_mag = LocateRC(
        mf115w, mf212n, mf115w,
        'F115W', 'F212N', 'F115W', 
        region='NRCB1'
    ).isolate(eps=0.1, min_samples=50) 


















    


