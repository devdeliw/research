import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from tqdm import tqdm
from lift import Lift 
from sim import Render
from astropy.table import Table 
from collections import defaultdict
from catalog_helper_functions import get_matches


class Run(Lift, Render):
    """
    Algorithm script. 

    Iterates through each region in `regions`. 

    First identifies the RC cluster from the CMD defined in `ansatz_params`. 
    This is accomplished via density-based clustering. 

    Afterwards lifts the RC clusters to all wavelengths in `param_table` and 
    performs a gradient ascent ridge-tracing algorithm to derive the slopes
    for each RC bar in every wavelength combination in `param_table`.

    Args: 
        catalog (astropy Table): full starlist with all wavelengths.
        regions (list of str): all the regions to calculate RC slopes.
        ansatz (list of filts): defines the well-defined RC bar CMD 
                                that the RC stars will be identified from. 
                                [`mag1`, `mag2`, `magy`] defines the 
                                `mag1`-`mag2` vs. `magy` CMD for the algorithm.

    """

    def __init__(
        self, 
        catalog, 
        regions, 
        ansatz, 
    ):
        self.catalog = catalog 
        self.regions = regions 
        self.ansatz = ansatz
        return 

    def define_rc(self): 
        print('\nIdentifying Red Clump Clusters...')
        print('---------------------------------')
        # initialize nested dictionary to store rc magnitudes
        data = defaultdict(lambda: defaultdict(dict))

        # iterate through regions in `regions`
        # with a pretty loading bar 
        for region in tqdm(self.regions, desc='NRCB'): 
            if self.ansatz[2] in ['F323N', 'F405N']: 
                y_region = 'NRCB5'
            else: 
                y_region = region

            m1, m2, _, _ = get_matches(
                self.catalog,
                self.ansatz[0], region, 
                self.ansatz[1], y_region, 
            )

            if self.ansatz[0] == self.ansatz[2]: 
                my = m1
            else: 
                my = m2

            region_data = Lift(
                m1, m2, my, 
                self.ansatz[0], self.ansatz[1], self.ansatz[2], 
                region=region, catalog=self.catalog
            ).lift()

            # Organizing 
            rc_f115w_mag = region_data['F115W']
            rc_f212n_mag = region_data['F212N']
            rc_f323n_mag = region_data['F323N'] 
            rc_f405n_mag = region_data['F405N'] 

            filt = self.ansatz
            data[region][f'{filt[0]}-{filt[1]}_{filt[2]}']['F115W'] = rc_f115w_mag
            data[region][f'{filt[0]}-{filt[1]}_{filt[2]}']['F212N'] = rc_f212n_mag
            data[region][f'{filt[0]}-{filt[1]}_{filt[2]}']['F323N'] = rc_f323n_mag
            data[region][f'{filt[0]}-{filt[1]}_{filt[2]}']['F405N'] = rc_f405n_mag

        return pd.DataFrame(data)

    def run(self, cmds, num_ansatz=30): 
        data = self.define_rc() 

        print('\nCalculating slopes...')
        print('---------------------')

        # initialize final result-storing nested dictionary 
        result = defaultdict(lambda: defaultdict(dict))

        for region in tqdm(self.regions, desc='Region', position=0, leave=True):
            for cmd in tqdm(cmds, desc=region, position=0, leave=True): 
                filt = self.ansatz
                label = f'{filt[0]}-{filt[1]}_{filt[2]}'

                mag1 = data[region][label][cmd[0]]
                mag2 = data[region][label][cmd[1]]
                magy = data[region][label][cmd[2]]
                mag1name = cmd[0]
                mag2name = cmd[1]
                magyname = cmd[2] 

                x = np.subtract(mag1, mag2) 
                y = np.array(magy) 


                # initalize rendering instance
                # will generate ansatz points along the RC bar 
                # and afterwards perform gradient ascent algo
                render_instance = Render(
                    mag1, mag2, magy, 
                    mag1name, mag2name, magyname, 
                    region=region
                ) 

                ansatz = render_instance.general_ansatz(
                    num_ansatz=num_ansatz, 
                    rc_color=x, 
                    rc_mag=y, 
                )

                slope, intercept = render_instance.gradient_ascent(
                    ansatz_points=ansatz
                )

                # Writing to result
                cmd_label = f'{mag1name}-{mag2name}'
                result[cmd_label][magyname] = (slope, intercept)

        return pd.DataFrame(result) 














if __name__ == '__main__': 
    ansatz = ['F115W', 'F212N', 'F115W']
    regions = [
            'NRCB1', 
            #'NRCB2', 
            #'NRCB3', 
            #'NRCB4', 
    ]

    cmds = [
            ['F115W', 'F212N', 'F115W'], 
            ['F115W', 'F212N', 'F212N'], 
            #['F212N', 'F323N', 'F212N'], 
            #['F212N', 'F323N', 'F323N'], 
            ['F212N', 'F405N', 'F212N'], 
            ['F212N', 'F405N', 'F405N'],
    ]

    catalog_dir = './fits/'
    file_name = 'jwst_init_NRCB.fits' 

    catalog = Table.read(
            os.path.join(catalog_dir, file_name), 
            format='fits'
    )

    df = Run(
            catalog, 
            regions, 
            ansatz, 
    ).run(cmds=cmds) 






                                

                    




