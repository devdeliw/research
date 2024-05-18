from color_magnitude_diagrams import * 


class Isochrones:   
	def __init__(self, logAge, AKs, dist, metallicity, filt_list, iso_dir, 
				red_law = None, verbose = False):

		"""

		Generates theoretical isochrone at a given age, extinction, and distance 
		from Earth. Also creates synthetic star cluster based on the same parameters. 

		Also generates a color-magnitude-diagram between `catalog1 and `catalog2` and
		overplots theoretical isochrones of increasing extinction. This is to see if 
		past reddening laws accurately model new data from the Galactic Center. 

		Specifically, if the theoretical isochrones of increasing extinction follow 
		the same slope on the CMD as the Red Clump Bar -- the reddening law works well. 

		----------
		Functions:
		----------
		# def generate_isochrone(self):

		Generates theoretical isochrone given parameters 
		`logAge`, `AKs`, `dist`, `metallicity`

		If `red_law` is not specified, I use the Fritz Reddening Law 
		spisea.reddening.RedLawFritz11(scale_lambda = 2.166) defined 
		from Fritz et al. 2011 for the Galactic Center.

		if verbose == True: 
			displays the properties of the synthetic cluster. 

		# display_isochrone(self, catalog1name, catalog2name, catalogyname)

		Makes a color-magnitude diagram from the isochrone. For visualization purposes.

		# def make_cluster(self, mass, show_cmd):

		Generates an Initial Mass Function (IMF) using the multiplicity properties 
		defined in Lu+13. 
		
		Makes & displays the resolved cluster using spisea synthetic.ResolvedCluster
		and the generated IMF.

		if verbose == True: 
			displays the properties of the synthetic cluster

		if show_cmd == True: 
			displays the isochrone & synthetic cluster on a color-magnitude-diagram

		# def extinction_vector(self, catalog1, catalog2, 
								catalog1name, catalog2name, catalogyname, 
								AKs_step, height, fig_path): 

		Generates an extinction_vector based on 5 isochrones of increasing extinction 
		AKs_current = AKs_previous + AKs_step

		Afterwards overlays the extinction vector on the color-magnitude diagram 
		between catalog1 (mag) and catalog2 (mag). 

		if catalogyname == catalog1name: 
			catalog1 - catalog2 (mag) vs. catalog1 (mag)
		if catalogyname == catalog2name: 
			catalog1 - catalog2 (mag) vs. catalog2 (mag)

		Note that catalog1 should be the magnitudes from the smaller wavelength.

		-----------------
        Class Parameters: 
        -----------------
            logAge      	: float (e.g. np.log10(5*10**6))
                            Age of synthetic cluster in log(years)

            AKs        		: float
                            extinction in mags

            dist   			: int
                            rough distance in parsec

            metallicity   	: float
                            metallicity in [M/H]

            filt    		: list 
                            list containing names of filters to use
                            (e.g. ['wfc3,ir,f127m', 'wfc3,ir,f139m', 'wfc3,ir,f153m'])
                            see documentation for filter name to use

            iso_dir         : string
                            directory to place isochrone.fits files

            red_law         : float
                            if specified, reddening law to be used
                            see SPISEA documentation for a list of red_laws 

            verbose      	: bool
                            if you want some info about the isochrones to be 
                            printed 

        """

		self.logAge = logAge
		self.AKs = AKs
		self.dist = dist
		self.metallicity = metallicity
		self.filt_list = filt_list
		self.iso_dir = iso_dir
		self.red_law = red_law
		self.verbose = verbose

	def generate_isochrone(self, different_filt_list = None): 

		evo_model = evolution.MISTv1()					# evolution model
		atm_func = atmospheres.get_merged_atmosphere	# atmospheric model

		if self.red_law != None: 
			red_law = self.red_law						# reddening law
		else: 
			red_law = reddening.RedLawFritz11(scale_lambda = 2.166)	

		if different_filt_list != None: 
			filters = different_filt_list
		else: 
			filters = self.filt_list

		isochrone = synthetic.IsochronePhot(self.logAge, self.AKs, self.dist, 
										   self.metallicity, evo_model, atm_func, red_law = red_law, 
										   filters = filters, iso_dir = self.iso_dir)

		if self.verbose: 
			"""

			The synthetic stars & their properties are stored in an astropy table
			called "points" within the IsochronePhot object

			"""

			print(isochrone.points)
			print("")
			print(f"The columns in the isochrone table are:") 
			print(f"{isochrone.points.keys()}")

		return isochrone

	def display_isochrone(self, catalog1name, catalog2name, catalogyname, fig_dir): 

		"""
		Displays Isochrone in a Color-Magnitude Diagram (CMD)

		if catalogyname == catalog1name: 
			Generates catalog1 - catalog2 vs. catalog1 CMD

		if catalogyname == catalog2name:
			Generates catalog1 - catalog2 vs. catalog2 CMD

		* catalog1name, catalog2name must be equal to some index of 
		  self.filt_list
		* catalogyname must either equal catalog1name or catalog2name

		Parameters: 
		-----------

		catalog1name	
		catalog2name	: string
						same string as the indices in self.filt_list
						you want to make a CMD between

		catalogyname	: string
						specifies which filter is going to be on the 
						y-axis of the CMD. must match either catalog1name 
						or catalog2name

		"""

		my_iso = Isochrones.generate_isochrone(self, different_filt_list = [catalog1name, catalog2name])
		
		print(my_iso)
		idx = np.where( abs(my_iso.points['mass'] - 1.0) == min(abs(my_iso.points['mass'] - 1.0)) )[0]

		check = False

		py.figure(1, figsize = (10, 10))
		py.clf()

		if catalogyname == catalog1name: 
			py.plot(my_iso.points[''+my_iso.points.keys()[8]] - my_iso.points[''+my_iso.points.keys()[9]], 
			my_iso.points[''+my_iso.points.keys()[8]], 'r-', label='_nolegend_')
			py.plot(my_iso.points[''+my_iso.points.keys()[8]][idx] - my_iso.points[''+my_iso.points.keys()[9]][idx], 
			my_iso.points[''+my_iso.points.keys()[8]][idx], 'b*', ms=15, label='1 $M_\odot$')
			py.xlabel(f'{catalog1name} - {catalog2name} (mag)')
			py.ylabel(f'{catalog1name} (mag)')
			py.gca().invert_yaxis()
			py.legend()
			py.savefig(f"{fig_dir}CMD_{catalog1name}_{catalog2name}_{catalog1name}")
			check = True

		if catalogyname == catalog2name: 
			py.plot(my_iso.points[''+my_iso.points.keys()[8]] - my_iso.points[''+my_iso.points.keys()[9]], 
			my_iso.points[''+my_iso.points.keys()[9]], 'r-', label='_nolegend_')
			py.plot(my_iso.points[''+my_iso.points.keys()[8]][idx] - my_iso.points[''+my_iso.points.keys()[9]][idx], 
			my_iso.points[''+my_iso.points.keys()[9]][idx], 'b*', ms=15, label='1 $M_\odot$')
			py.xlabel(f'{catalog1name} - {catalog2name} (mag)')
			py.ylabel(f'{catalog2name} (mag)')
			py.gca().invert_yaxis()
			py.legend()
			py.savefig(f"{fig_dir}CMD_{catalog1name}_{catalog2name}_{catalog2name}")
			check = True

		if not check: 
			raise Exception("catalogyname must equal catalog1name or catalog2name")

		return 

	def make_cluster(self, mass, catalog1name, catalog2name, catalogyname, fig_dir,
					show_cmd = True, verbose = False):

		"""

		Generates an Initial Mass Function (IMF) and a synthetic starcluster based on
		isochrone parameters. 

		The IMF is defined by Kroupa and uses the multiplicity properties defind in Lu+13

		if show_cmd == True: 
			displays the isochrone and synthetic cluster on a CMD 

			if catalogyname == catalog1name: 
				catalog1 - catalog2 vs. catalog1 synthetic CMD

			if catalogyname == catalog2name: 
				catalog1 - catalog2 vs. catalog2 synthetic CMD

		* `catalogyname` must either equal `catalog1name` or `catalog2name`
		* the names `catalog1name`, `catalog2name` and `catalogyname` must 
		  be valid names of filters used by SPISEA.
		
		if verbose == True: 
			print astropy table that stores info of synthetic stars
			print astropy `companions` table that contains properties of the 
			companions to the primary star within each system. See documentation 
			for description of the columns in both of these tables
		

		"""

		imf_multi = multiplicity.MultiplicityUnresolved()
		# star systems are unresolved, i.e., all components of a star system are combined
		# into a single 'star' in the synthetic cluster

		massLimits = np.array([0.2, 0.5, 1, 120])
		powers = np.array([-1.3, -2.3, -2.3])

		my_imf = imf.IMF_broken_powerlaw(massLimits, powers, imf_multi)
		my_iso = Isochrones.generate_isochrone(self, different_filt_list = [catalog1name, catalog2name])

		cluster = synthetic.ResolvedCluster(my_iso, my_imf, mass)

		if verbose: 
			print("Star Systems Table")
			print(cluster.star_systems)
			print("")
			print("The cluster table contains these columns")
			print(f"{cluster.star_systems.keys()}")
			print("")
			print("Companions Table")
			print(cluster.companions)

		if show_cmd: 
			clust = cluster.star_systems
			iso = my_iso.points
			check = False

			py.figure(2, figsize = (10, 10))
			py.clf()

			if catalogyname == catalog1name: 
				py.plot(clust[''+my_iso.points.keys()[8]] - clust[''+my_iso.points.keys()[9]], 
						clust[''+my_iso.points.keys()[8]], 
						'k.', ms = 5, alpha = 0.1, label = 'synthetic cluster')
				py.plot(iso[''+my_iso.points.keys()[8]] - iso[''+my_iso.points.keys()[9]], 
						iso[''+my_iso.points.keys()[8]], 
						'r-', label = 'isochrone')

				py.xlabel(f'{catalog1name} - {catalog2name} (mag)')
				py.ylabel(f'{catalog1name} (mag)')
				py.gca().invert_yaxis()
				py.legend()
				py.savefig(f"{fig_dir}synthetic_iso_cluster_{catalog1name}_{catalog2name}_{catalog1name}")
				check = True

			if catalogyname == catalog2name: 
				py.plot(clust[''+my_iso.points.keys()[8]] - clust[''+my_iso.points.keys()[9]], 
						clust[''+my_iso.points.keys()[9]], 
						'k.', ms = 5, alpha = 0.1, label = 'synthetic cluster')
				py.plot(iso[''+my_iso.points.keys()[8]] - iso[''+my_iso.points.keys()[9]], 
						iso[''+my_iso.points.keys()[9]], 
						'r-', label = 'isochrone')

				py.xlabel(f'{catalog1name} - {catalog2name} (mag)')
				py.ylabel(f'{catalog2name} (mag)')
				py.gca().invert_yaxis()
				py.legend()
				py.savefig(f"{fig_dir}synthetic_iso_cluster_{catalog1name}_{catalog2name}_{catalog2name}")
				check = True

			if not check: 
				raise Exception("catalogyname must equal catalog1name or catalog2name")

		return clust

	def extinction_vector(self, catalog1, catalog2, 
						  catalog1name, catalog2name, catalogyname, 
						  AKs_step, fig_path, height = 0, 
						  matched = True, dr_tol = None, dm_tol = None): 


		"""
		Plots a real CMD using data from `catalog1` and `catalog2` and overlay 
		5 isochrones of increasing extinction and plots the isochrone-based 
		extinction vector going through all isochrones. 

		If the extinction law used in generating the isochrones is a good fit, 
		the extinction vector should line up almost exactly with the slope of 
		the Red Clump cluster.

		Parameters: 
		-----------

		catalog1, catalog2 			: array-like or pandas DataFrame (see matched)
									contains mag information for both filters for the CMD

		catalog1name, catalog2name 	: string
									spisea names for filters of catalog1 and catalog2

		catalogyname 				: string
									indicates which filter is to be placed on the y-axis
									either equals catalog1name or catalog2name

		matched 					: bool
									
									if False, employs the matching algorithm in color_magnitude_diagrams.py,
									`Color_Magnitude_Diagrams.match(), to determine matching stars between 
									catalog1 and catalog2 are identical. Note if matched is False, 
									catalog1 and catalog2 must be *pandas DataFrame* that contain information 
									for the `x`, `y`, and magnitude `m` data for each star. 

									if True, goes straight to plotting the CMD and isochrones as both catalogs 
									are already the same size with information for the same stars. If this is 
									the case, catalog1 and catalog2 can just be array-like

		AKs_step					: float
									step increase for extinction value AKs for each of the 5 isochrones


		height 						: float
									defines how far to shift up the extinction vector to align with the RC bar. 
									requires a few runs to define correctly. Note since CMDs usually have their 
									axis inverted, you may choose to have height be negative

		"""

		fig, axis = plt.subplots(1, 1, figsize = (20, 10))
		plt.gca().invert_yaxis()

		if matched == True: 
			check = False

			catalog1 = np.array(catalog1)
			catalog2 = np.array(catalog2)

			x = np.subtract(catalog1, catalog2)

			AKs2 = self.AKs + AKs_step
			AKs3 = AKs2 + AKs_step
			AKs4 = AKs3 + AKs_step
			AKs5 = AKs4 + AKs_step

			different_filt_list = [catalog1name, catalog2name]

			my_iso = Isochrones.generate_isochrone(self, different_filt_list)
			idx = np.where( abs(my_iso.points['mass'] - 1.0) == min(abs(my_iso.points['mass'] - 1.0)) )[0]

			self.AKs = AKs2
			my_iso2 = Isochrones.generate_isochrone(self, different_filt_list)
			idx2 = np.where( abs(my_iso2.points['mass'] - 1.0) == min(abs(my_iso2.points['mass'] - 1.0)) )[0]

			self.AKs = AKs3
			my_iso3 = Isochrones.generate_isochrone(self, different_filt_list)
			idx3 = np.where( abs(my_iso3.points['mass'] - 1.0) == min(abs(my_iso3.points['mass'] - 1.0)) )[0]

			self.AKs = AKs4
			my_iso4 = Isochrones.generate_isochrone(self, different_filt_list)
			idx4 = np.where( abs(my_iso4.points['mass'] - 1.0) == min(abs(my_iso4.points['mass'] - 1.0)) )[0]

			self.AKs = AKs5
			my_iso5 = Isochrones.generate_isochrone(self, different_filt_list)
			idx5 = np.where( abs(my_iso5.points['mass'] - 1.0) == min(abs(my_iso5.points['mass'] - 1.0)) )[0]

			if catalogyname == catalog1name: 
				check = True
				y = np.array(catalog1)

				plt.scatter(x, y, c = 'k', s = 0.8)

				plt.plot(my_iso.points[''+my_iso.points.keys()[8]] - my_iso.points[''+my_iso.points.keys()[9]],
    					 my_iso.points[''+my_iso.points.keys()[8]], 'r-', label='_nolegend_')
				plt.plot(my_iso2.points[''+my_iso2.points.keys()[8]] - my_iso2.points[''+my_iso2.points.keys()[9]],
    					 my_iso2.points[''+my_iso2.points.keys()[8]], 'r-', label='_nolegend_')
				plt.plot(my_iso3.points[''+my_iso3.points.keys()[8]] - my_iso3.points[''+my_iso3.points.keys()[9]],
    					 my_iso3.points[''+my_iso3.points.keys()[8]], 'r-', label='_nolegend_')
				plt.plot(my_iso4.points[''+my_iso4.points.keys()[8]] - my_iso4.points[''+my_iso4.points.keys()[9]],
    					 my_iso4.points[''+my_iso4.points.keys()[8]], 'r-', label='_nolegend_')
				plt.plot(my_iso5.points[''+my_iso5.points.keys()[8]] - my_iso5.points[''+my_iso5.points.keys()[9]],
    					 my_iso5.points[''+my_iso5.points.keys()[8]], 'r-', label='_nolegend_')
				
				plt.axline((my_iso5.points[''+my_iso5.points.keys()[8]][idx5][0]
						    - my_iso5.points[''+my_iso5.points.keys()[9]][idx5][0], 
						    my_iso5.points[''+my_iso5.points.keys()[8]][idx5][0] + height), 
						    (my_iso.points[''+my_iso.points.keys()[8]][idx][0]
						    - my_iso.points[''+my_iso.points.keys()[9]][idx][0],
						    my_iso.points[''+my_iso.points.keys()[8]][idx][0] + height), 
						    color = 'aqua', label = "extinction vector")

				plt.xlabel(f"{catalog1name} - {catalog2name}")
				plt.ylabel(f"{catalog1name}")
				filename = f"extinction_vec_{catalog1name}_{catalog2name}_{catalog1name}"
				plt.legend()
				plt.savefig(f"{fig_path}{filename}.png")	

			if catalogyname == catalog2name: 
				check = True
				y = np.array(catalog2)

				plt.scatter(x, y, c = 'k', s = 0.8)

				plt.plot(my_iso.points[''+my_iso.points.keys()[8]] - my_iso.points[''+my_iso.points.keys()[9]],
    					 my_iso.points[''+my_iso.points.keys()[9]], 'r-', label='_nolegend_')
				plt.plot(my_iso2.points[''+my_iso2.points.keys()[8]] - my_iso2.points[''+my_iso2.points.keys()[9]],
    					 my_iso2.points[''+my_iso2.points.keys()[9]], 'r-', label='_nolegend_')
				plt.plot(my_iso3.points[''+my_iso3.points.keys()[8]] - my_iso3.points[''+my_iso3.points.keys()[9]],
    					 my_iso3.points[''+my_iso3.points.keys()[9]], 'r-', label='_nolegend_')
				plt.plot(my_iso4.points[''+my_iso4.points.keys()[8]] - my_iso4.points[''+my_iso4.points.keys()[9]],
    					 my_iso4.points[''+my_iso4.points.keys()[9]], 'r-', label='_nolegend_')
				plt.plot(my_iso5.points[''+my_iso5.points.keys()[8]] - my_iso5.points[''+my_iso5.points.keys()[9]],
    					 my_iso5.points[''+my_iso5.points.keys()[9]], 'r-', label='_nolegend_')
				
				plt.axline((my_iso5.points[''+my_iso5.points.keys()[8]][idx5][0]
						    - my_iso5.points[''+my_iso5.points.keys()[9]][idx5][0], 
						    my_iso5.points[''+my_iso5.points.keys()[9]][idx5][0] + height), 
						    (my_iso.points[''+my_iso.points.keys()[8]][idx][0]
						    - my_iso.points[''+my_iso.points.keys()[9]][idx][0],
						    my_iso.points[''+my_iso.points.keys()[9]][idx][0] + height), 
						    color = 'aqua', label = "extinction vector")

				plt.xlabel(f"{catalog1name} - {catalog2name}")
				plt.ylabel(f"{catalog2name}")
				filename = f"extinction_vec_{catalog1name}_{catalog2name}_{catalog2name}"
				plt.legend()	
				plt.savefig(f"{fig_path}{filename}.png")	

			if not check: 
				raise Exception("catalogyname must equal catalog1name or catalog2name")

		if matched == False: 
			check = False

			idxs1, idxs2, catalog1, catalog2, catalog1_error, catalog2_error = Color_Magnitude_Diagram(catalog1, catalog2, 
																									   catalog1name, catalog2name, catalogyname,
																									   dr_tol, dm_tol).match()
			catalog1 = np.array(catalog1)
			catalog2 = np.array(catalog2)

			x= np.subtract(catalog1, catalog2)

			AKs2 = self.AKs + AKs_step 
			AKs3 = AKs2 + AKs_step
			AKs4 = AKs3 + AKs_step
			AKs5 = AKs4 + AKs_step

			different_filt_list = [catalog1name, catalog2name]

			my_iso = Isochrones.generate_isochrone(self, different_filt_list)
			idx = np.where( abs(my_iso.points['mass'] - 1.0) == min(abs(my_iso.points['mass'] - 1.0)) )[0]

			self.AKs = AKs2
			my_iso2 = Isochrones.generate_isochrone(self, different_filt_list)
			idx2 = np.where( abs(my_iso2.points['mass'] - 1.0) == min(abs(my_iso2.points['mass'] - 1.0)) )[0]

			self.AKs = AKs3
			my_iso3 = Isochrones.generate_isochrone(self, different_filt_list)
			idx3 = np.where( abs(my_iso3.points['mass'] - 1.0) == min(abs(my_iso3.points['mass'] - 1.0)) )[0]

			self.AKs = AKs4
			my_iso4 = Isochrones.generate_isochrone(self, different_filt_list)
			idx4 = np.where( abs(my_iso4.points['mass'] - 1.0) == min(abs(my_iso4.points['mass'] - 1.0)) )[0]

			self.AKs = AKs5
			my_iso5 = Isochrones.generate_isochrone(self, different_filt_list)
			idx5 = np.where( abs(my_iso5.points['mass'] - 1.0) == min(abs(my_iso5.points['mass'] - 1.0)) )[0]

			if catalogyname == catalog1name: 
				check = True
				y = np.array(catalog1)

				plt.scatter(x, y, c = 'k', s = 0.8)

				plt.plot(my_iso.points[''+my_iso.points.keys()[8]] - my_iso.points[''+my_iso.points.keys()[9]],
    					 my_iso.points[''+my_iso.points.keys()[8]], 'r-', label='_nolegend_')
				plt.plot(my_iso2.points[''+my_iso2.points.keys()[8]] - my_iso2.points[''+my_iso2.points.keys()[9]],
    					 my_iso2.points[''+my_iso2.points.keys()[8]], 'r-', label='_nolegend_')
				plt.plot(my_iso3.points[''+my_iso3.points.keys()[8]] - my_iso3.points[''+my_iso3.points.keys()[9]],
    					 my_iso3.points[''+my_iso3.points.keys()[8]], 'r-', label='_nolegend_')
				plt.plot(my_iso4.points[''+my_iso4.points.keys()[8]] - my_iso4.points[''+my_iso4.points.keys()[9]],
    					 my_iso4.points[''+my_iso4.points.keys()[8]], 'r-', label='_nolegend_')
				plt.plot(my_iso5.points[''+my_iso5.points.keys()[8]] - my_iso5.points[''+my_iso5.points.keys()[9]],
    					 my_iso5.points[''+my_iso5.points.keys()[8]], 'r-', label='_nolegend_')
				
				plt.axline((my_iso5.points[''+my_iso5.points.keys()[8]][idx5][0]
						    - my_iso5.points[''+my_iso5.points.keys()[9]][idx5][0], 
						    my_iso5.points[''+my_iso5.points.keys()[8]][idx5][0] + height), 
						    (my_iso.points[''+my_iso.points.keys()[8]][idx][0]
						    - my_iso.points[''+my_iso.points.keys()[9]][idx][0],
						    my_iso.points[''+my_iso.points.keys()[8]][idx][0] + height), 
						    color = 'aqua', label = "extinction vector")

				plt.xlabel(f"{catalog1name} - {catalog2name}")
				plt.ylabel(f"{catalog1name}")
				filename = f"extinction_vec_{catalog1name}_{catalog2name}_{catalog1name}"
				plt.legend()
				plt.savefig(f"{fig_path}{filename}.png")	

			if catalogyname == catalog2name: 
				check = True
				y = np.array(catalog2)

				plt.scatter(x, y, c = 'k', s = 0.8)

				plt.plot(my_iso.points[''+my_iso.points.keys()[8]] - my_iso.points[''+my_iso.points.keys()[9]],
    					 my_iso.points[''+my_iso.points.keys()[9]], 'r-', label='_nolegend_')
				plt.plot(my_iso2.points[''+my_iso2.points.keys()[8]] - my_iso2.points[''+my_iso2.points.keys()[9]],
    					 my_iso2.points[''+my_iso2.points.keys()[9]], 'r-', label='_nolegend_')
				plt.plot(my_iso3.points[''+my_iso3.points.keys()[8]] - my_iso3.points[''+my_iso3.points.keys()[9]],
    					 my_iso3.points[''+my_iso3.points.keys()[9]], 'r-', label='_nolegend_')
				plt.plot(my_iso4.points[''+my_iso4.points.keys()[8]] - my_iso4.points[''+my_iso4.points.keys()[9]],
    					 my_iso4.points[''+my_iso4.points.keys()[9]], 'r-', label='_nolegend_')
				plt.plot(my_iso5.points[''+my_iso5.points.keys()[8]] - my_iso5.points[''+my_iso5.points.keys()[9]],
    					 my_iso5.points[''+my_iso5.points.keys()[9]], 'r-', label='_nolegend_')
				
				plt.axline((my_iso5.points[''+my_iso5.points.keys()[8]][idx5][0]
						    - my_iso5.points[''+my_iso5.points.keys()[9]][idx5][0], 
						    my_iso5.points[''+my_iso5.points.keys()[9]][idx5][0] + height), 
						    (my_iso.points[''+my_iso.points.keys()[8]][idx][0]
						    - my_iso.points[''+my_iso.points.keys()[9]][idx][0],
						    my_iso.points[''+my_iso.points.keys()[9]][idx][0] + height), 
						    color = 'aqua', label = "extinction vector")

				plt.xlabel(f"{catalog1name} - {catalog2name}")
				plt.ylabel(f"{catalog2name}")
				filename = f"extinction_vec_{catalog1nam}_{catalog2name}_{catalog2name}"
				plt.legend()
				plt.savefig(f"{fig_path}{filename}.png")	

			if not check: 
				raise Exception("catalogyname must equal catalog1name or catalog2name")

		return 

















		










Iso = Isochrones(logAge = np.log(10**9), AKs = 2, dist = 8000, 
				 metallicity = -0.3, filt_list = ['jwst,F115W', 'jwst,F212N'],
				 red_law = reddening.RedLawFritz11(scale_lambda=2.166), iso_dir = "/Users/devaldeliwala/research/work/plots&data/isochrone_plots&data/data/")
my_iso = Iso.display_isochrone(catalog1name = 'jwst,F115W', catalog2name = 'jwst,F212N', 
							   catalogyname = 'jwst,F115W', fig_dir = "/Users/devaldeliwala/research/work/plots&data/isochrone_plots&data/plots/")

catalog1 = Table.read("catalogs/dr2/NRCB1_catalog115w.csv").to_pandas()
catalog2 = Table.read("catalogs/dr2/NRCB1_catalog212n.csv").to_pandas()

idxs1, idxs2, catalog1, catalog2, catalog1_error, catalog2_error = Color_Magnitude_Diagram(catalog1, catalog2, 
																						   catalog1name = "NRCB1_catalog115w", catalog2name = "NRCB1_catalog212n", catalogyname = "NRCB1_catalog115w",
																					       dr_tol = 0.5, dm_tol = 100).match()

Iso.extinction_vector(catalog1 = catalog1, catalog2 = catalog2, 
					  catalog1name = 'jwst,F115W', catalog2name = 'jwst,F212N', catalogyname = 'jwst,F115W', 
					  AKs_step = 0.25, height = -12.4, fig_path = "/Users/devaldeliwala/research/work/plots&data/color_magnitude_diagram_plots/", 
					  matched = True) 


