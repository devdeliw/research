from extinction import Generate_CMD, RC_stars
import pandas as pd

"""
JWST F115w, F212n catalogs correspond to 1.15 micron and 2.12 micron
wavelengths respectively.

"""

# generating DataFrame objects using the 1.15 and 2.12 µm wavelength catalogs

df = pd.read_csv("catalog115w.csv", delimiter = ",")
df2 = pd.read_csv("catalog212n.csv", delimiter = ",")

#---------------------------------------------------#

cmd = Generate_CMD(df, df2, "jwst_115w", "jwst_212n", dr_tol = 15, dm_tol = 15,
                   y_axis_m1 = False)

idxs1, idxs2, dr, dm, m1, m2, m1_err, m2_err = cmd.match()
m1_match, m2_match = cmd.cmd(idxs1, idxs2, m1, m2)


#--------------------------------------------------------#


rc = RC_stars(m1_match, m2_match, 1.25, 14.875, 15.275, 0.35294, 14.25, 14.4582)

RCcoords, RCcoords2, idxs1, idxs2 = rc.rc_candidates()
rc.ratios_extinction(RCcoords, RCcoords2, "jwst_115w", "jwst_212n")

