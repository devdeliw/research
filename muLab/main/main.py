import os 

from tqdm import tqdm
from sim import Render 
from astropy.table import Table 
from collections import defaultdict
from catalog_helper_functions import get_matches 


combinations = [
    #['F115W', 'F212N'],
    #['F212N', 'F323N'], 
    #['F212N', 'F405N'], 
    ['F115W', 'F405N'], 
]

regions = [
    'NRCB1', 
    #'NRCB2', 
    #'NRCB3', 
    #'NRCB4', 
]

catalog_dir = './fits/'
file_name = 'jwst_init_NRCB.fits'


# Triple for-loop looks ugly, but I swear it's not that bad. 
data = defaultdict(lambda: defaultdict(list))
def run():
    for region in tqdm(
        regions, 
        desc="NRCB"
    ):
        for combo in tqdm(
            combinations, 
            desc=f"Evaluating {region}", 
            leave=False
        ):
            catalog = Table.read(
                os.path.join(catalog_dir, file_name), 
                format='fits', 
            )

            if (
                combo[1] == 'F323N' or 
                combo[1] == 'F405N'
            ): 
                regiony = 'NRCB5'
            else: 
                regiony = region

            m1, m2, _, _ = get_matches(
                catalog,
                combo[0], region, 
                combo[1], regiony, 
            )

            slopes = []
            for i in tqdm(
                range(len(combo)), 
                desc=f"Running {combo[0]} and {combo[1]}", 
                leave=False
            ): 
                if i == 0: 
                    slope, intercept = Render(
                        m1, m2, m1, 
                        combo[0], combo[1], combo[i], 
                        region, 
                    ).gradient_ascent()
                else: 
                    slope, intercept = Render(
                        m1, m2, m2, 
                        combo[0], combo[1], combo[i], 
                        region,
                    ).gradient_ascent() 
                slopes.append((slope, intercept))
            data[f'{region}'][f'{combo[0]}-{combo[1]}'] = slopes

    return 
run()

