from cuqi.distribution import DistributionGallery, Gaussian, JointDistribution
from cuqi.testproblem import Poisson1D
from cuqi.problem import BayesianProblem
import cuqi
import inspect
import numpy as np
import matplotlib.pyplot as plt
from cuqi.sampler import MH, CWMH, ULA, MALA, NUTS
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

class Criteria():

    def __init__(self, samples, pr, dim = 2):
        self.samples = samples
        self.dim = dim
        self.pr =pr


    def compute_ESS(self):
        """Compute effective sample size (ESS) for the selected sampling methods."""
        ess_values = {}
        ess = {}
        mean = {}
        for method in self.samples.keys():
            ess_values[method] = self.samples[method].compute_ess()
            mean[method] = np.mean(ess_values[method])
            if self.dim > 2:
                ess[method] = {
                    'max': np.max(ess_values[method]),
                    'min': np.min(ess_values[method])
                }
            else:
                ess = ess_values 
        
        return ess, mean
    
    def compute_AR(self):
        """Compute acceptance rate (AR) for the selected sampling methods."""
        ar = {}

        for method in self.samples.keys():

            if method == 'NUTS' or 'CWMH' or 'NUTS_adapted' or 'CWMH_adapted':
                ar[method] = len(np.unique(self.samples[method].samples[0])) / len(self.samples[method].samples[0])
            else:
                ar[method] = self.samples[method].acc_rate
        
        return ar
    def count_function(self, string):
        """
        Count occurrences of a specific function call in profiling results.
        
        Parameters:
        pr     : dict, containing profiling objects for each MCMC method
        string : str, function name to count in the profiling results

        Returns:
        counter : dict, containing counts of the specified function calls for each method
        """
        counter = {}

        for method in self.pr.keys():
            s = io.StringIO()
            ps = pstats.Stats(self.pr[method], stream=s).sort_stats(SortKey.PCALLS)
            ps.print_stats()
            lines = s.getvalue().split('\n')
            
            if any(string in line for line in lines):
                idx = [string in line for line in lines].index(True)
                counter[method] = int(lines[idx].split()[0])
            else:
                counter[method] = 0  # If the function was not found, set the count to 0

        return counter
    def compute_Rhat(self, data):
        """
        Compute Rhat statistic for convergence diagnostics across multiple chains.
        
        Parameters:
        samples : dict, containing samples for each MCMC method
        data    : list of dicts, containing samples from different chains for each method

        Returns:
        rhat : dict, containing Rhat values for each sampling method
        """
        rhat = {}
        rhat_values = {}
        mean_rhat ={}

        for method in self.samples.keys():
            rhat_values[method] = self.samples[method].compute_rhat([item[method] for item in data])
            mean_rhat[method] = np.mean(rhat_values[method]) #havent tested it yet
            if self.dim >=3: 
                rhat[method] = {
                    'max': np.max(rhat_values[method]),
                    'min': np.min(rhat_values[method]),

                }
            else:
                rhat = rhat_values

        return rhat


