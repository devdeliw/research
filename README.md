## Research with the Moving Universe Lab (MULab) 
##### under Prof. Jessica Lu

[MULab](https://jluastro.atlassian.net/wiki/spaces/MULab/overview)

Primary work involves deriving the extinction law of the galactic center via
red clump (RC) stars using data from JWST. 

The work generally involves building color-magnitude diagrams, extracting theoretical
isochrones, isolating the RC cluster in the CMD, and using the slope of the RC
cluster to derive extinction ratios, which when extrapolated yield an
extinction law that researchers can use to better understand the region of
stars around Sagittarius A*. 

More specifically, I write statistical models to accurately measure the slope
of the Red Clump Cluster in the CMDs of the Galactic Center. This involves
unsharp masking, generating theoretical stellar isochrones,
building synthetic red clump clusters, compound fitting, bootstrapping, and
much more. 

[prev_work/](https://github.com/devdeliw/research/tree/main/prev_work) contains previous JWST messy work, [work/](https://github.com/devdeliw/research/tree/main/work) is current.

We are close to publication so I am compiling all my work into .py files in
the `work` directory. Within the `work` directory lies
a [plots&data/](https://github.com/devdeliw/research/tree/main/work/plots%26data) folder
which stores all image and data files corresponding to each python file. There
is also a `flystar` directory which contains code for performing matching
algorithms, with or without possible coordinate transformations between two star catalogs.

### Overview of Code in `work` 

[color_magnitude_diagrams.py](https://github.com/devdeliw/research/blob/main/work/color_magnitude_diagrams.py) performs all operations regarding generating a CMD. If the catalogs of different wavelengths are not matched, it implements a KDTree nearest neighbor algorithm with a certain `dr_tol` and checks for matches to be within a `dm_tol`. Provides the option to generate unsharp masks (deemphasizing less dense regions and emphasizing dense regions) with a certain `binsize` and `binwidth`. It also provides the option to plot a regular CMD or generate one that plots points with color as a function of their density. See `work/plots&data/color_magnitude_diagram_plots/` and `work/plots&data/unsharp_mask/`. 

[isochrones.py](https://github.com/devdeliw/research/blob/main/work/isochrones.py)
performs all operations regarding generating and plotting stellar isochrones,
with an emphasis on plotting isochrones of *increasing extinction* to hopefully
line them up along with the Red Clump Bar of a CMD. It allows you to specify
all parameters of a synthetic stellar isochrone based on
[SPISEA](https://spisea.readthedocs.io) along with the atmospheric, evolution,
and reddening laws to use. See `work/plots&data/isochrone_plots&data/`.

[red_clump_riemann.py](https://github.com/devdeliw/research/blob/main/work/red_clump_riemann.py)
performs the bulk of the analysis on the RC bar. Normally one cuts of the RC
portion of a CMD using two parallel lines by-eye, and afterwards determines the
slope of the stars within the cutoff accordingly. This procedure is heavily dependent on one's eye and resulting errors are hard to quantify this way.

This code, provided a color magnitude diagram, determines the slope, with error, of the RC cluster without any
dependency on RC cutoffs by-eye. Details about this algorithm, along with the
other `red_clump` files in the
[work/](https://github.com/devdeliw/research/tree/main/work) directory will be
presented in an upcoming research paper.


I utilize [SPISEA](https://spisea.readthedocs.io) developed by Hosek+18 et al. to derive 
theoretical synthetic isochrones for my research. 
