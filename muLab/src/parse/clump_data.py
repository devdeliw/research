import numpy as np 
import pandas as pd 
import pickle 
import ast 
import matplotlib.pyplot as plt 

from util.catalog_helper_functions import get_matches
from astropy.table import Table 



class Define_RC(): 
    def __init__(
            self, 
            filt1, reg1, 
            filt2, reg2, 
            filty, regy, 
    ): 
        self.filt1 = filt1
        self.filt2 = filt2
        self.filty = filty
        self.reg1  = reg1 
        self.reg2  = reg2 
        self.regy  = regy

        FITS = "./files/jwst_init_NRCB.fits" 
        self.catalog = Table.read(FITS)


        self.xy1, self.xy2, self.m1, self.m2, _, _ = get_matches(
            self.catalog, filt1, reg1, filt2, reg2, 
        )

        self.my = self.m1 if filty == filt1 else self.m2
        return 

    def rc(self, cutoffs, write=False): 
        row = cutoffs[ 
            (cutoffs["region1"] == self.reg1) &
            (cutoffs["region2"] == self.reg2) &
            (cutoffs["regiony"] == self.regy) &
            (cutoffs["catalog1"] == self.filt1) &
            (cutoffs["catalog2"] == self.filt2) & 
            (cutoffs["catalogy"] == self.filty)
        ]

        line1 = ast.literal_eval(row["parallel_cutoff1"].iloc[0])
        line2 = ast.literal_eval(row["parallel_cutoff2"].iloc[0])

        slope = round((line1[1][1] - line1[0][1]) / (line1[1][0] - line1[0][0]), 3)
        intercept1 = line1[1][1] - slope * line1[1][0]
        intercept2 = line2[1][1] - slope * line2[1][0] 

        # Expand RC Cutoff by 3x to include more stars 
        height = abs(intercept1 - intercept2) 
        upper_intercept = max(intercept1, intercept2) + height 
        lower_intercept = min(intercept1, intercept2) - height

        x = np.subtract(self.m1, self.m2)
        upper_bound = slope*x + upper_intercept 
        lower_bound = slope*x + lower_intercept  

        # Extract RC star catalog indices 
        rc_mask = (self.my <= upper_bound) & (self.my >= lower_bound)
        rc_idxs = np.where(rc_mask == True)[0]
        full_idxs = np.where( 
            np.isin(
                self.catalog['m'], 
                self.my[rc_idxs], 
            )
        )[0] 

        rc_magnitudes = self.catalog[full_idxs]['m']
        rc_centroid_x = self.catalog[full_idxs]['x']
        rc_centroid_y = self.catalog[full_idxs]['y']

        # Different regions have different idxs in the main catalog 
        region_keys = {
            "NRCB1": 0, 
            "NRCB2": 1, 
            "NRCB3": 2, 
            "NRCB4": 3, 
        }

        # Store all filter RC data in dictionary 
        red_clump_data = {} 

        # Match identical stars found in m1-m2 v. my 
        # to higher wavelengths in the same region
        f115w_idx = region_keys.get(self.reg1, None)
        red_clump_data['idx']    = np.array(full_idxs)
        red_clump_data['x']      = np.array(rc_centroid_x[:, f115w_idx], dtype=float)
        red_clump_data['y']      = np.array(rc_centroid_y[:, f115w_idx], dtype=float)
        red_clump_data['mF115W'] = np.array(rc_magnitudes[:, f115w_idx], dtype=float) 
        red_clump_data['mF212N'] = np.array(rc_magnitudes[:, f115w_idx+4], dtype=float) 
        red_clump_data['mF323N'] = np.array(rc_magnitudes[:, 9], dtype=float) 
        red_clump_data['mF405N'] = np.array(rc_magnitudes[:, 8], dtype=float)

        red_clump_data = pd.DataFrame(red_clump_data) 

        # Export RC data to .pkl 
        if write:
            os.makedirs("./files/parse/", exist_ok=True)
            with open(f"./files/parse/red_clump_data_{self.reg1}.pkl", "wb") as f: 
                pickle.dump(red_clump_data, f) 

        return 

    def plot(
        self, 
        key1, 
        key2, 
        keyy,
        region, 
    ): 
        with open(f"./files/parse/red_clump_data_{region}.pkl", "rb") as f: 
            red_clump_data = pickle.load(f) 

        m1 = red_clump_data[key1] 
        m2 = red_clump_data[key2] 
        my = red_clump_data[keyy]  

        _, _ = plt.subplots(1, 1, figsize=(10, 8)) 

        x = np.subtract(m1, m2) 
        plt.scatter(x, my, marker='+')
        plt.gca().invert_yaxis() 

        plt.show() 

        return 


if __name__ == "__main__": 
    FITS = "/Users/devaldeliwala/research/mulab/src/files/jwst_init_NRCB.fits" 
    catalog = Table.read(FITS)

    CUTOFFS = "/Users/devaldeliwala/research/mulab/src/files/parse/red_clump_cuts.pkl" 
    with open(CUTOFFS, "rb") as f:
        cutoffs = pickle.load(f)

    inst = Define_RC(
        catalog, 
        'F115W', 'NRCB4', 
        'F212N', 'NRCB4', 
        'F115W', 'NRCB4', 
    ) 
    #inst.rc(cutoffs, write=True)
    inst.plot('mF115W', 'mF212N', 'mF115W', 'NRCB1')

