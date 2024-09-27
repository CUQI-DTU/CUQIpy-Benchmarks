from cuqi.distribution import DistributionGallery, Gaussian, JointDistribution
from cuqi.testproblem import Poisson1D
from cuqi.problem import BayesianProblem
import cuqi
import inspect
import numpy as np
import matplotlib.pyplot as plt
from cuqi.experimental.mcmc import MH as MH, CWMH as CWMH, ULA as ULA, MALA as MALA, NUTS as NUTS
import time
import scipy.stats as sps
from scipy.stats import gaussian_kde
import pandas as pd
import cProfile, pstats, io
from pstats import SortKey
from prettytable import PrettyTable
from IPython.display import Image, display
import math
import warnings
from ._criteria import Criteria
from ._plot import Plot

class MCMCComparison():
   
    def __init__(self, target , scale, Ns, Nb , dim = 2, x0 = None, seed =None, chains = 2, selected_criteria =None, selected_methods = None ):
        """
        Initialize the MCMCSampler with a target distribution and parameters.
        
        Parameters:
        target : cuqi.distribution.Distribution object
        scale  : float or array, scaling factors for the samplers
        Ns     : int or array, number of samples for each sampler
        Nb     : int or array, number of burn-ins for each sampler
        x0     : initial state, either an array or a CUQI distribution
        seed   : int, random seed
        chains : int, number of MCMC chains for Rhat calculation
        selected_methods : list of strings, selected sampling methods (e.g., "MH", "CWMH")
        selected_criteria: list of strings, selected criteria for comparison (e.g., ["ESS", "AR"])
        """
        self.target = target
        self.scale = scale
        self.Ns = Ns
        self.Nb = Nb
        self.dim = dim
        self.x0 = x0
        if hasattr(self.x0, '__module__') and self.x0.__module__.startswith("cuqi.distribution"):
            self.x0 = self.x0.sample().to_numpy()
            self.distribution = True
        elif hasattr(self.target,'prior'):
                self.x0 = self.target.prior
                self.distribution = True
        else: self.distribution = False

        self.seed = seed
        self.chains = chains
        self.selected_methods = selected_methods or ["MH", "CWMH", "ULA", "MALA", "NUTS"]
        self.selected_criteria = selected_criteria or ["ESS", "AR", "LogPDF", "Gradient", "Rhat"]

        self.sampling_results = {}
        if hasattr(self.target,'prior'):
            self.x0 = self.target.prior
    
    def precompute_samples(self, save = False):

        """
        Precompute samples for various MCMC methods and return the results.
        
        Parameters:
        target : cuqi.distribution.Distribution object
        scale  : float or array, scaling factors for the samplers
        Ns     : int or array, number of samples for each sampler
        Nb     : int or array, number of burn-ins for each sampler
        x0     : initial state, either an array or a CUQI distribution
        seed   : int, random seed
        selected_methods: list of strings, selected sampling methods (e.g., )
        

        Returns:
        samples : dict, containing samples for each MCMC method
        pr      : dict, containing profiling objects for each MCMC method
        scale   : array, adjusted scaling factors
        Ns      : array, adjusted number of samples
        Nb      : array, adjusted number of burn-ins
        """
        np.random.seed(self.seed + self.chains)
        
        
        if isinstance(self.scale, float):
            self.scale = np.full(len(self.selected_methods), self.scale)
        elif len(self.scale) !=len(self.selected_methods):
            raise TypeError("scale array lenght is not correct")
        if isinstance(self.Ns, int):
            self.Ns = np.full(len(self.selected_methods), int(self.Ns))
        elif len(self.Ns) != len(self.selected_methods):
            raise TypeError("Ns array lenght is not correct")
        if isinstance(self.Nb, int):
            self.Nb = np.full(len(self.selected_methods), int(self.Nb))
        elif len(self.Nb) != len(self.selected_methods):
            raise TypeError("Nb array lenght is not correct")
        

        samples = {}
        pr = {}
        
        # Dictionary to map method names to their corresponding parameters and functions
        method_mapping = {
            'MH': (MH, False),
            'MH_adapted': (MH, True),
            'CWMH': (CWMH, False),
            'CWMH_adapted': (CWMH, True),
            'ULA': (ULA, False),
            'ULA_adapted': (ULA, True),
            'MALA': (MALA, False),
            'MALA_adapted': (MALA, True),
            'NUTS': (NUTS, False),
            'NUTS_adapted': (NUTS, True)

        }

        # Loop over selected methods and compute samples
        for idx, method in enumerate(self.selected_methods):
            if method in method_mapping:
                mcmc_method, adapted = method_mapping[method]
                samples[method], pr[method] = self.MCMC_sampling(mcmc_method, adapted,  self.scale[idx], self.Ns[idx], self.Nb[idx])
                if save:
                    self.sampling_results[method] = {
                        "samples": samples[method],
                        "profiling": pr[method]
                    }
        
        return samples, pr
    
    def MCMC_sampling(self, method, adapted,scale_index, ns_index, nb_index):
        

        pr = cProfile.Profile()
        pr.enable()
        try:
            np.random.seed(self.seed)
            if method == NUTS: 
                sampler = method(target = self.target, initial_point = self.x0)

            else:
                sampler = method(target = self.target, scale = scale_index, initial_point = self.x0)
            sampler.warmup(nb_index)
            sampler.sample(ns_index)
            if adapted:
                
                x = sampler.get_samples.burnthin(nb_index)
            else: 
                x = sampler.get_samples()

        finally:
            pr.disable()

        return x, pr
    
    def get_sampling_result(self, method):
        """
        Retrieve the sampling results for a given method.
        """
        if method not in self.sampling_results:
            raise ValueError(f"Method '{method}' has not been run or stored.")
        return self.sampling_results[method]
    
    def create_comparison(self):

        """
        Create a table comparing various sampling methods with ESS values.
        
        Parameters:
        target : cuqi.distribution.Distribution object
        scale  : float or array, scaling factors for the samplers
        Ns     : int or array, number of samples for each sampler
        Nb     : int or array, number of burn-ins for each sampler
        x0     : initial state, either an array or a CUQI distribution
        seed   : int, random seed
        chains : int, number of MCMC chains for Rhat calculation
        selected_criteria : list of strings, selected criteria for comparison (e.g., ["ESS", "AR"])
        selected_methods:
        
        Returns:
        df   : pandas.DataFrame, comparison table
        plot : matplotlib figure, plot of the samples
        """

        # Run precomputation
        samples, pr = self.precompute_samples(True)
        df_dict ={}
        for method in self.selected_methods:
            df_dict[method] ={}

        for idx, method in enumerate(self.selected_methods):
            df_dict[method]["samples"] = int(self.Ns[idx])
            df_dict[method]["burnins"] = int(self.Nb[idx])
            if method == "NUTS" or method == "NUTS_adapted":
                df_dict[method]["scale"] = "-"
            else: df_dict[method]["scale"] = self.scale[idx]

        criteria = Criteria(samples, pr, self.dim)
        if "ESS" in self.selected_criteria:
            

            ess, mean = criteria.compute_ESS()
            # mean = compute_meanESS(samples)  
            if self.dim ==1: 
                for method in self.selected_methods:
                    df_dict[method]["ESS"] = round(ess[method].item(), 3)
            elif self.dim == 2:
                for method in self.selected_methods: 
                    df_dict[method]["ESS(v0)"] = round(ess[method][0], 3)
                    df_dict[method]["ESS(v1)"] = round(ess[method][1], 3)
            else: 
                for method in self.selected_methods: 
                    df_dict[method]["ESS(max)"] = round(ess[method]['max'], 3)
                    df_dict[method]["ESS(min)"] = round(ess[method]['min'], 3)
                    df_dict[method]["ESS(mean)"] = round(mean[method], 3)
        
        if "AR" in self.selected_criteria:
            ar = criteria.compute_AR()

            for method in self.selected_methods:
                df_dict[method]["AR"] = round(ar[method], 3)

        if "LogPDF" in self.selected_criteria:
            logpdf = criteria.count_function("logpdf")
            for method in self.selected_methods:
                df_dict[method]["LogPDF"] = int(logpdf[method]) #make them nice
                
        
        
        if "Gradient" in self.selected_criteria:
            gradient = criteria.count_function("_gradient")
            for method in self.selected_methods:
                df_dict[method]["Gradient"] = int(gradient[method]) #make them nice
                # df_dict['Gradient'] = [int(x) if pd.notnull(x) else '-' for x in df_dict['Gradient']]

        if "Rhat" in self.selected_criteria:
            if self.distribution:
                data = []
                for i in range(self.chains - 1):
                    chain_samples, _ = self.precompute_samples()
                    data.append(chain_samples)
                    
        
                rhat = criteria.compute_Rhat(data)
                if self.dim == 1:
                    for method in self.selected_methods: 
                        df_dict[method]["Rhat"] = rhat[method]
                elif self.dim == 2:
                    for method in self.selected_methods: 
                        df_dict[method]["Rhat(v0)"] = round(rhat[method][0], 3)
                        df_dict[method]["Rhat(v1)"] = round(rhat[method][1], 3)
                else:
                    for method in self.selected_methods:
                        df_dict[method]["Rhat(max)"] = round(rhat[method]['max'], 3)
                        df_dict[method]["Rhat(min)"] = round(rhat[method]['min'], 3)
            
                
        if "ESS" in self.selected_criteria and "LogPDF" in self.selected_criteria:
            for method in self.selected_methods:
                df_dict[method]["LogPDF/ESS"] = round(logpdf[method]/mean[method], 3)
        
        if "ESS" in self.selected_criteria and "Gradient" in self.selected_criteria:
            for method in self.selected_methods:
                df_dict[method]["Gradient/ESS"] = round(gradient[method]/mean[method], 3)

        df = pd.DataFrame(df_dict)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            df.loc['samples'] = df.loc['samples'].apply(lambda x: f"{x:.0f}")
            df.loc['burnins'] = df.loc['burnins'].apply(lambda x: f"{x:.0f}")
            if "LogPDF" in self.selected_criteria:
                df.loc['LogPDF'] = df.loc['LogPDF'].apply(lambda x: f"{x:.0f}")
            if "Gradient" in self.selected_criteria:
                df.loc['Gradient'] = df.loc['Gradient'].apply(lambda x: f"{x:.0f}")

        return df
    
    def create_plt(self):
        if self.dim != 2: 
             raise ValueError(f"Plot of '{self.dim}' cannot be plotted.")
        else: 
            a = Plot(self.target, self.sampling_results, self.selected_methods, self.dim)
            return a.plot_sampling()
        
    

    
    

    