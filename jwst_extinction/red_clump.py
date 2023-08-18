from extinction import Generate_CMD, RC_stars
from spisea_vector import SPISEA_CMD
import pandas as pd
import numpy as np

"""
JWST F115w, F212n catalogs correspond to 1.15 micron and 2.12 micron
wavelengths respectively.

"""

# generating DataFrame objects using the 1.15 and 2.12 µm, 3.23µm, and 4.05µm wavelength catalogs

df = pd.read_csv("catalogs/catalog115w.csv", delimiter = ",")
df2 = pd.read_csv("catalogs/catalog212n.csv", delimiter = ",")
df3 = pd.read_csv("catalogs/catalog323n.csv", delimiter = ",")
df4 = pd.read_csv("catalogs/catalog444w.csv", delimiter = ",")


#------------------------------------------------------------#


cmd = Generate_CMD(df, df2, "jwst_115w", "jwst_212n", dr_tol = 15, dm_tol = 15,
                   y_axis_m1 = True)

cmd2 = Generate_CMD(df3, df4, "jwst_323n", "jwst_444w", dr_tol = 15, dm_tol = 15,
                   y_axis_m1 = True)


idxs1, idxs2, dr, dm, m1, m2, m1_err, m2_err = cmd.match()
m1_match, m2_match = cmd.cmd(idxs1, idxs2, m1, m2)

idxs3, idxs4, dr2, dm2, m3, m4, m3_err, m4_err = cmd2.match()
m3_match, m4_match = cmd2.cmd(idxs3, idxs4, m3, m4)


#-------------------------------------------------#

rc = RC_stars(m1_match, m2_match, 1.25, 14.875, 15.275, 0.35294, 14.25, 14.4582)
rc2 = RC_stars(m3_match, m4_match,1.1764 , 13.364705, 14.1529, 1.4285, 12.1314, 13.071)

RCcoords, RCcoords2, idxs1, idxs2 = rc.rc_candidates()
rc.ratios_extinction(RCcoords, RCcoords2, "jwst_115w", "jwst_212n")

RCcoords3, RCcoords4, idxs3, idxs4 = rc2.rc_candidates()
rc2.ratios_extinction(RCcoords3, RCcoords4, "jwst_323n", "jwst_405n")


#-------------------------------------------------------------------#

spisea = SPISEA_CMD(df, df2, "jwst_115w", "jwst_212n", dr_tol = 15, dm_tol = 15,
                    y_axis_m1 = True)

idxs1, idxs2, dr, dm, m1, m2, m1_err, m2_err = spisea.match()

m1_match, m2_match = spisea.cmd(idxs1, idxs2, m1, m2)


spisea.extinction_vector(np.log(10**9), 2, 0.3, 8000, -0.3, ['jwst,F115W', 'jwst,F212N'],
                    "/Users/devaldeliwala/research/jwst_extinction/media/isochrones")


spisea2 = SPISEA_CMD(df3, df4, "jwst_323n", "jwst_405n", dr_tol = 15, dm_tol = 15,
                   y_axis_m1 = True)

idxs3, idxs4, dr2, dm2, m3, m4, m3_err, m4_err = spisea2.match()

m3_match, m4_match = spisea2.cmd(idxs3, idxs4, m3, m4)

spisea2.extinction_vector(np.log(10**9), 2, 1.5, 8000, -0.3, ['jwst,F323N', 'jwst,F405N'],
                     "/Users/devaldeliwala/research/jwst_extinction/media/isochrones")

#------------------------------------------------------------------------------------#


