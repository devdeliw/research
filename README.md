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
unsharp masking, comparing to theoretical isochrones of increasing extinction,
binning the CMD and 3D compound-fitting and much more. 

`messy_jwst_exintction` and `messy_jwst` are both  previous JWST messy work, **`work` is current**.
We are close to publications so I am compiling all my work into .py files in
the `work` directory

I heavily utilize [SPISEA](https://spisea.readthedocs.io) developed by Hosek+18 et al. to derive 
theoretical synthetic isochrones for my research. 
