## Research with the Moving Universe Lab (MULab) 
##### under Prof. Jessica Lu

[MULab](https://jluastro.atlassian.net/wiki/spaces/MULab/overview)

Primary work involves deriving the extinction law of the galactic center via
red clump (RC) stars using data from JWST. 

I write statistical models to accurately measure the slope
of the Red Clump Cluster in the CMDs of the Galactic Center. This involves
unsharp masking, generating theoretical stellar isochrones,
building synthetic red clump clusters, compound fitting, and bootstrapping. As
of now, I have a [working algorithm](https://github.com/devdeliw/research/blob/main/work/red_clump_riemann.py) that, provided a CMD with a few parameters,
can output the red clump slope with error very accurately. See
[work/](https://github.com/devdeliw/research/tree/main/work) folder readme.   

[z_excess](https://github.com/devdeliw/research/tree/main/z_excess) contains previous JWST messy work, and writeups for group presentations.

We are close to publication so I am compiling all my work into .py files in
the `work` directory. Within the `work` directory lies
a [plots&data/](https://github.com/devdeliw/research/tree/main/work/plots%26data) folder
which stores all image and data files outputted from running code. There
is also a `flystar` directory which contains code for performing matching
algorithms, with or without possible coordinate transformations between two star catalogs.

To see a current writeup of my algorithm, visit
[my website](https://dev-undergrad.dev/ocf_writeup.pdf).  


I utilize [SPISEA](https://spisea.readthedocs.io) developed by Hosek+18 et al. to derive 
theoretical synthetic isochrones for my research. 
