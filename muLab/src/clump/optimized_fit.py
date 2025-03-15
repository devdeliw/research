import os 
import numpy as np
import emcee
import corner
import pickle 
import matplotlib.pyplot as plt

from parse.tile import Tile
from scipy.optimize import curve_fit

class MCMC_Optimizer():
    """ 
    Runs Markov Chain Monte Carlo (MCMC) to fit a compound Gaussian+Linear1D 
    model to a red clump distribution. 

    """
        
    def __init__(self, data, bins=50):
        """
        Parameters
        ----------
        data : array-like
            1D data samples 
        bins : int or sequence
            If int, number of bins for np.histogram. If sequence, the bin edges.
        """
        self.data = np.asarray(data)
        self.bins = bins

        self.bin_heights, self.bin_edges = np.histogram(self.data, bins=self.bins)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

        self.bin_errors = np.sqrt(self.bin_heights + 1)

        self.samples = None      
        self.best_fit_params = None  

    @staticmethod
    def _compound_model(theta, x):
        amplitude, mean, sigma, slope, intercept = theta
        gauss = amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)
        linear = slope * x + intercept
        return gauss + linear

    def _log_prior(self, theta):
        amplitude, mean, sigma, slope, intercept = theta

        data_mean = np.mean(self.data) 
        data_sigma= np.std(self.data) 

        allowed_mean_range = [data_mean-2., data_mean+2.]
        allowed_sigma_range= [data_sigma-0.5, data_sigma+0.5] 

        if (amplitude <= 0 or amplitude >= 1000 or 
            sigma <= allowed_sigma_range[0] or sigma >= allowed_sigma_range[1] or 
            mean <= allowed_mean_range[0] or mean >= allowed_mean_range[1]):
            return -np.inf

        if sigma > 1e5 or abs(slope) > 30 or abs(intercept) > 50:
            return -np.inf

        return 0.0  

    def _log_likelihood(self, theta):
        model_vals = self._compound_model(theta, self.bin_centers)
        resid = self.bin_heights - model_vals
        inv_var = 1.0 / (self.bin_errors ** 2)

        return -0.5 * np.sum(resid**2 * inv_var + np.log(2.0 * np.pi / inv_var))

    def _log_probability(self, theta):
        lp = self._log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self._log_likelihood(theta)

    def _initial_guess(self):
        # Ansatz guesses. 
        amp_guess = np.max(self.bin_heights)/4
        mean_guess = np.mean(self.data)
        sigma_guess = np.std(self.data)
        slope_guess = 0.0
        intercept_guess = np.min(self.bin_heights) if np.min(self.bin_heights) > 0 else 1.0

        return [amp_guess, mean_guess, sigma_guess, slope_guess, intercept_guess]

    def run(self, nwalkers=32, nsteps=3000, burnin=500, thin=10):
        """
        Runs the MCMC to fit the histogram; compound Gaussian+Linear1D model.

        Parameters
        ----------
        nwalkers : int
            Number of MCMC walkers.
        nsteps : int
            Total steps in the chain.
        burnin : int
            Steps to discard as burn-in.
        thin : int
            Thinning factor (use every `thin`th sample to reduce autocorrelation).

        Returns
        -------
        best_fit_params : dict
            Dictionary containing the median (best-fit) parameter values.
        samples : ndarray
            2D array of shape (n_samples, 5) with the posterior samples
            (after burn-in and thinning).
        """

        init_guess = self._initial_guess()
        ndim = len(init_guess)

        # Initial walker pos
        pos = init_guess + 1e-4 * np.random.randn(nwalkers, ndim)

        # emcee sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self._log_probability)

        sampler.run_mcmc(pos, nsteps, progress=False)

        # Log probability for all walkers at every step
        log_probs = sampler.get_log_prob(discard=0, flat=False)  

        # Discard burnin 
        self.samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)

        # Median of each parameter = best-fit
        amp_m, mean_m, sigma_m, slope_m, intercept_m = np.median(self.samples, axis=0)
        self.best_fit_params = {
            "amplitude": amp_m,
            "mean": mean_m,
            "stddev": sigma_m,
            "slope": slope_m,
            "intercept": intercept_m
        }

        return self.best_fit_params, self.samples, log_probs 

    def plot_corner(self):
        # Creates Corner Plot
        if self.samples is None:
            raise RuntimeError("No samples found. Run the MCMC first.")

        fig = corner.corner(
            self.samples,
            labels=["amplitude", "mean", "stddev", "slope", "intercept"],
            show_titles=True
        )
        return fig

    def plot_fit(self):
        # Histogram Fit Plot 
        if self.best_fit_params is None:
            raise RuntimeError("No best-fit parameters. Run the MCMC first.")

        plt.figure(figsize=(8, 5))
        plt.errorbar(
            self.bin_centers, self.bin_heights, yerr=self.bin_errors,
            fmt=".k", capsize=2, label="Histogram data"
        )

        bf = self.best_fit_params
        x_fine = np.linspace(self.bin_edges[0], self.bin_edges[-1], 500)
        model_vals = self._compound_model(
            [bf["amplitude"], bf["mean"], bf["stddev"], bf["slope"], bf["intercept"]],
            x_fine
        )
        plt.plot(x_fine, model_vals, "r-")

        plt.xlabel("Bin Center")
        plt.ylabel("Bin Count")
        plt.title("Compound Fit")
        plt.legend()
        return plt 

class OCFResult(): 
    def __init__(
        self, 
        filt1, filt2, filty, 
        reg1, reg2, regy, 
        n, verbose=True
    ): 
        self.filt1 = filt1 
        self.filt2 = filt2 
        self.filty = filty 
        self.reg1  = reg1 
        self.reg2  = reg2
        self.regy  = regy 
        self.n = n 
        self.verbose = verbose

        try: 
            with open(f"./files/parse/data/{self.reg1}/tiles/{n}/{self.filt1}-{self.filt2}_{self.filty}.pkl", "rb") as f: 
                self.bins = pickle.load(f)
        except FileNotFoundError: 
            self.bins = self.initialize_bins()
        return 

    def initialize_bins(self, plot=True):
        tile_instance = Tile(**self.__dict__)
        bins = tile_instance.extract_stars(self.n) 
        #tile_instance.plot_tiles(self.n, write=True)
        return bins 

    def optimize(self, plot=True, image_dir = "./files/clump/images/", write=False): 
        os.makedirs(image_dir + f'{self.reg1}/', exist_ok=True)

        if self.verbose: 
            print("[MCMC] Setting up...")
            print(f"\n{self.reg1} {self.filt1} - {self.reg2} {self.filt2} vs. {self.regy} {self.filty}")
            print("-"*75)
            print(f"| {'BIN':<5} | {'AMP':<8} | {'MEAN':<8} | {'ERROR':<8} | {'STDDEV':<8} | {'SLOPE':<8} | {'INTER':<8} | ")
            print("-"*75)


        if plot: 
            colors = plt.cm.jet(np.linspace(0, 1, len(self.bins)))
            _, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={"projection": "3d"}) 
            ax.view_init(elev = 35, azim = -45, roll = 0)
            ax.set_xlabel(f"{self.reg1} {self.filt1} - {self.reg2} {self.filt2}", fontsize=14)
            ax.set_ylabel(f"{self.regy} {self.filty}", fontsize=14)
            ax.set_zlabel("Frequency", fontsize=14)
            ax.zaxis.labelpad=-0.01

        for key, starlist in self.bins.items():
            m1, m2 = starlist["stars"] 
            x = np.subtract(m1, m2) 
            y = m1 if self.filt1 == self.filty else m2

            inst = MCMC_Optimizer(y)
            bf, samples, log_probs = inst.run() 

            mean_samples = samples[:, 1] 
            yerr_mean = np.sqrt(np.var(mean_samples, ddof=1))
            bf['error'] = yerr_mean

            self.bins[key] = {
                "starlist"      : starlist["stars"], 
                "fit_params"    : bf, 
                "log_probs"     : log_probs,
                "fit_samples"   : samples, 
            }

            if self.verbose: 
                print(f"| {key:<5} | {bf['amplitude']:<8.4f} | {bf['mean']:<8.4f} | {bf['error']:<8.4f} | {bf['stddev']:<8.4f} | {bf['slope']:<8.4f} | {bf['intercept']:<8.4f} | ")

            if plot:
                x_fine = np.linspace(inst.bin_edges[0], inst.bin_edges[-1], 500)
                model_vals = inst._compound_model(
                    [bf["amplitude"], bf["mean"], bf["stddev"], bf["slope"], bf["intercept"]],
                    x_fine
                )
                plt.scatter(x, y, color=colors[key], s=0.3, zs=0, zdir='z')
                plt.plot(x_fine, model_vals, color=colors[key], zs=np.min(x), zdir='x')

        if self.verbose: 
            print("-"*75)
        if plot:
            plt.tight_layout()
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1) 
            os.makedirs(f"./files/clump/images/{self.reg1}/{self.n}/", exist_ok=True)
            plt.savefig(f"./files/clump/images/{self.reg1}/{self.n}/MCMC_3D_{self.filt1}-{self.filt2}_{self.filty}.png", dpi=300)
            plt.close()

        if write: 
            os.makedirs(f"./files/clump/data/{self.reg1}/{self.n}/", exist_ok=True)
            with open(f"./files/clump/data/{self.reg1}/{self.n}/MCMC_RES{self.filt1}-{self.filt2}_{self.filty}.pkl", "wb") as f: 
                pickle.dump(self.bins, f)

        return self.bins

    def slope(self, rerun = False, write=False):
        try: 
            if not rerun: 
                with open(f"./files/clump/{self.reg1}/{self.n}/MCMC_RES{self.filt1}-{self.filt2}_{self.filty}.pkl", "rb") as f: 
                    self.bins = pickle.load(f)
            else: 
                raise ValueError("This is forced! Ignore!") 
        except:
            self.optimize(write=write) 

        _means, errors, x_vals = [], [], []

        for idx in range(len(self.bins)):
            params = self.bins[idx]["fit_params"]

            mean = params["mean"] 
            _err = params["error"]

            _means.append(mean) 
            errors.append(_err) 

            m1, m2 = self.bins[idx]["starlist"]

            x = np.subtract(m1, m2) 
            x_vals.append(np.mean(x))

        def linear_func(x, slope, intercept): 
            return slope * x + intercept

        # Weighted linear fit to calculate RC Bar Slope 
        popt, pcov = curve_fit(
            linear_func, 
            x_vals, 
            _means, 
            sigma=errors, 
            absolute_sigma=True, 
        )

        slope, intercept = popt 
        slope_err, intercept_err = np.sqrt(np.diag(pcov))

        if self.verbose: 
            print(f"\n{self.reg1} {self.filt1} - {self.reg2} {self.filt2} vs. {self.regy} {self.filty}")
            print(f"m: {round(slope, 4)} ± {round(slope_err, 4)}")
            print(f"b: {round(intercept, 4)} ± {round(intercept_err, 4)}\n")
        

        return slope, intercept, slope_err, intercept_err


if __name__ == "__main__":

    inst = OCFResult(
        "F212N", "F323N", "F212N",
        "NRCB1", "NRCB5", "NRCB1", 
        5, 
    ) 

    inst.slope(rerun=True) 

        

