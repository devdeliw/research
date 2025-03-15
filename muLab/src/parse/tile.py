import numpy as np 
import pickle
import ast 
import matplotlib.pyplot as plt 

from parse.clump_data import Define_RC


class Tile(): 
    def __init__(
        self, 
        filt1, filt2, filty, 
        reg1, reg2, regy, **kwargs
    ): 
        self.filt1 = filt1 
        self.filt2 = filt2 
        self.filty = filty 
        self.reg1 = reg1
        self.reg2 = reg2
        self.regy = regy 
        return 

    def tile(self, n): 
        try: 
            with open(f"./files/parse/red_clump_cuts.pkl", "rb") as f: 
                cutoffs = pickle.load(f) 
        except: 
            raise Exception("./files/parse/red_clump_cuts.pkl missing.")

        row = cutoffs[ 
            (cutoffs["region1"] == self.reg1) &
            (cutoffs["region2"] == self.reg2) &
            (cutoffs["regiony"] == self.regy) &
            (cutoffs["catalog1"] == self.filt1) &
            (cutoffs["catalog2"] == self.filt2) & 
            (cutoffs["catalogy"] == self.filty)
        ]

        xmin, xmax = ast.literal_eval(row["x_range"].iloc[0])
        dx = abs(xmin - xmax)/n
        bins, xcur = {}, min(xmin, xmax)
        for idx in range(n):
            bins[idx] = [xcur, xcur + dx]
            xcur += dx 

        return bins 

    def extract_stars(self, n, write=False):
        bins = self.tile(n) 

        try:
            with open(f"./files/parse/red_clump_data_{self.reg1}.pkl", "rb") as f: 
                red_clump_data = pickle.load(f) 
        except: 
            inst = Define_RC(
                self.filt1, self.reg1, 
                self.filt2, self.reg2, 
                self.filty, self.regy,
            )

            CUTOFFS = './files/parse/red_clump_cuts.pkl' 
            with open(CUTOFFS, "rb") as f:
                cutoffs = pickle.load(f)

            inst.rc(cutoffs, write=True)

            with open(f"./files/parse/red_clump_data_{self.reg1}.pkl", "rb") as f: 
                red_clump_data = pickle.load(f) 
            
        m1 = np.array(red_clump_data[f"m{self.filt1}"]) 
        m2 = np.array(red_clump_data[f"m{self.filt2}"]) 

        x = np.subtract(m1, m2) 

        for idx, bin in bins.items():
            mask = ((x >= bin[0]) & (x < bin[1])) 
            bins[idx] = {'bin': bin, 'stars': np.vstack((m1[mask], m2[mask]))}

        if write: 
            os.makedirs("./files/parse/data/{self.reg1}/tiles/{n}/", exist_ok=True)
            with open(f"./files/parse/data/{self.reg1}/tiles/{n}/{self.filt1}-{self.filt2}_{self.filty}.pkl", "wb") as f: 
                pickle.dump(bins, f) 
        return bins 

    def plot_bins(self, n): 
        try: 
            with open(f"./files/parse/data/{self.reg1}/tiles/{n}/{self.filt1}-{self.filt2}_{self.filty}_{n}.pkl", "rb") as f: 
                bins = pickle.load(f) 
        except: 
            self.extract_stars(n)
            with open(f"./files/parse/data/{self.reg1}/tiles/{n}/{self.filt1}-{self.filt2}_{self.filty}_{n}.pkl", "rb") as f: 
                bins = pickle.load(f)

        _, _ = plt.subplots(1, 1, figsize=(10, 8)) 

        for _, starlist in bins.items(): 
            m1, m2 = starlist['stars']
            x = np.subtract(m1, m2) 
            y = m1 

            plt.scatter(x, y) 

        plt.gca().invert_yaxis() 
        plt.show() 
        return 

    def plot_tiles(self, n, write=False):
        from astropy.table import Table
        from scipy.stats import gaussian_kde
        from util.catalog_helper_functions import get_matches
        import os 

        catalog = "./files/jwst_init_NRCB.fits"
        catalog = Table.read(catalog)

        _, _, m1, m2, _, _ = get_matches(
            catalog, self.filt1, self.reg1, self.filt2, self.reg2, 
        )
        my = m1 if self.filt1 == self.filty else m2 

        m1 = np.array(m1) 
        m2 = np.array(m2) 
        my = np.array(my)

        mask = ~np.isnan(m1) & ~np.isnan(m2) 
        m1 = m1[mask] 
        m2 = m2[mask] 
        my = my[mask] 

        x, y = np.subtract(m1, m2), my 
        xy = np.vstack([x, y]) 
        z = gaussian_kde(xy)(xy)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        plt.scatter(x, y, c=z, cmap='viridis', s=0.2) 
        plt.gca().invert_yaxis() 
        plt.xlabel(f'{self.reg1} {self.filt1} - {self.reg2} {self.filt2}', fontsize=14) 
        plt.ylabel(f'{self.regy} {self.filty}', fontsize=14) 
        plt.title('RC Selection', fontsize=14)

        CUTOFFS = "/Users/devaldeliwala/research/mulab/src/files/parse/red_clump_cuts.pkl" 
        with open(CUTOFFS, "rb") as f:
            cutoffs = pickle.load(f)

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
        x_range = ast.literal_eval(row["x_range"].iloc[0])

        # 2 Bounding Parallel Lines 
        slope = round((line1[1][1] - line1[0][1]) / (line1[1][0] - line1[0][0]), 3)
        intercept1 = line1[1][1] - slope * line1[1][0]
        intercept2 = line2[1][1] - slope * line2[1][0] 

        # Expand RC Cutoff by 3x to include more stars 
        height = abs(intercept1 - intercept2) 
        upper_intercept = max(intercept1, intercept2) + height        
        lower_intercept = min(intercept1, intercept2) - height

        plt.axline(xy1=(line1[0][0], line1[0][1]), slope=slope, c='orangered', linestyle='--') 
        plt.axline(xy1=(line2[0][0], line2[0][1]), slope=slope, c='orangered', linestyle='--') 

        plt.axline(xy1=(0, upper_intercept), slope=slope, c='teal', linestyle='--') 
        plt.axline(xy1=(0, lower_intercept), slope=slope, c='teal', linestyle='--')

        # RC Tile Region 
        x = np.linspace(x_range[0], x_range[1], 100)
        y_upper = slope * x + upper_intercept 
        y_lower = slope * x + lower_intercept 

        ax.fill_between(x, y_upper, y_lower, color='crimson', alpha=0.2)

        borders = np.linspace(x_range[0], x_range[1], n+1) 
        y_lower = slope * borders + lower_intercept 
        y_upper = slope * borders + upper_intercept 
        ax.vlines(borders, ymin=y_lower, ymax=y_upper, color='crimson', alpha=0.3)

        if write: 
            plt.grid(alpha=0.2)
            plt.tight_layout()
            os.makedirs(f"./files/parse/images/{self.reg1}/{n}/", exist_ok=True)
            plt.savefig(f"./files/parse/images/{self.reg1}/{n}/{self.filt1}-{self.filt2}_{self.filty}_CUTOFF.png", dpi=300)
            plt.close() 
        else: 
            plt.close() 
        return  
            



if __name__ == "__main__": 
    inst = Tile(
        'F212N', 'F323N', 'F212N', 
        'NRCB2', 'NRCB5', 'NRCB2' 
    )
    bins = inst.extract_stars(8, write=False)
    inst.plot_tiles(8, write=True)



