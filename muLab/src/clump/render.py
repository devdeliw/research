import pandas as pd 
import pickle 
import os 

from clump.optimized_fit import OCFResult 


class Render(): 
    """ 
    Runs MCMC across all required CMD filter combinations 
    to derive A_lambda / A_{Ks~2.12 micron} extinction ratios

    """ 

    def __init__(self, n=28, verbose=True, write=False, output_dir='./files/clump/'): 
        os.makedirs(output_dir, exist_ok=True) 

        try: 
            with open("./files/parse/red_clump_cuts.pkl", "rb") as f: 
                script = pickle.load(f) 
        except: 
            raise Exception("./files/parse/red_clump_cuts.pkl missing.") 

        self.script     = script
        self.output_dir = output_dir 
        self.n          = n
        self.verbose    = verbose
        self.write      = write

    def _render_(self):

        results = {
            "reg1": [],
            "reg2": [],
            "regy": [],
            "filt1": [],
            "filt2": [],
            "filty": [],
            "slope": [],
            "slope_err": [],
            "intercept": [],
            "intercept_err": []
        }

        if self.verbose: 
            print(f"\n Initializing All NRCB Red Clump Clusters with {self.n} tiles...\n")

        # Iterate through every combination in script 
        # Store the output slopes 
        for row in self.script.itertuples(index=True, name="rc"):
            inst = OCFResult(
                row.catalog1, row.catalog2, row.catalogy, 
                row.region1, row.region2, row.regiony, 
                n=self.n, verbose=True 
            )

            m, b, me, be = inst.slope(rerun=True, write=self.write)

            results["reg1"].append(inst.reg1)
            results["reg2"].append(inst.reg2)
            results["regy"].append(inst.regy)
            results["filt1"].append(inst.filt1)
            results["filt2"].append(inst.filt2)
            results["filty"].append(inst.filty)
            results["slope"].append(m)
            results["slope_err"].append(me)
            results["intercept"].append(b)
            results["intercept_err"].append(be)

        result = pd.DataFrame(results)
        result.to_csv(f"{self.output_dir}result_{self.n}.csv")

        print(f"OCF Completed. Results placed in {self.output_dir}result_{self.n}.csv")
        return

    def _individual_(
        self, 
        filt1, filt2, filty, 
        reg1, reg2, regy, 
        n=28, verbose=True 
    ):
        if verbose: 
            print(f"Initializing Red Clump Cluster with {n} tiles...")

        inst = OCFResult(
            filt1, filt2, filty, 
            reg1, reg2, regy, 
            n=n, verbose=verbose
        )

        inst.slope(rerun=True, write=True) 

if __name__ == "__main__":
    import numpy as np 
    for iter in range(0, 1): 
        for n in np.arange(6, 31, 2): 
            Render(n=n, write=True, output_dir=f"./files/clump/{iter}/")._render_()

    #Render()._individual_("F115W", "F212N", "F115W" , "NRCB3", "NRCB3", "NRCB3")
    
