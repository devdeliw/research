### Python File Information

---

[color_magnitude_diagrams.py](https://github.com/devdeliw/research/blob/main/work/color_magnitude_diagrams.py) deals with everything about CMDs. It performs kdtree nearest neighbor catalog matching w/o coordinate transforms and generates a CMD. It also has unsharp masking capabilities to accentuate the dense parts of the CMD and de-accentuate the less-dense regions. This is particularly useful for isolating the main-sequence or red-clump region of the CMD, the latter of which my research is heavily involved with. 

[isochrones.py](https://github.com/devdeliw/research/blob/main/work/isochrones.py)
deals with everything about stellar isochrones. It generates theoretical
isochrones using [SPISEA](https://spisea.readthedocs.io/en/latest/) and
overlays them on generated color magnitude diagrams generated from
[color_magnitude_diagrams.py](https://github.com/devdeliw/research/blob/main/work/color_magnitude_diagrams.py). 

[red_clump_riemann.py](https://github.com/devdeliw/research/blob/main/work/red_clump_riemann.py)
and the rest of the `red_clump` files are the pinnacle of my research thus far.
Together they allow a user to input a CMD, along with a few other parameters,
and outputs the slope of the RC bar with error without extremely any
significant eye-dependence. 

It works by tiling segments of the RC bar and performing my Optimized Curve
Fitting (OCF) algorithm which fits compound `astropy.models.Gaussian1D()
+ astropy.models.Linear1D()` models and performs weighted error analysis and/or
bootstrapping.
[red_clump_analysis_script.csv](https://github.com/devdeliw/research/blob/main/work/red_clump_analysis_script.csv)
stores all the parameters for the algorithm for every wavelength combination in
every region, along with their corresponding calculated RC slopes with error.

The RC slopes can then be used to accurately derive the extinction law of the
Galactic Center. 


