# %%
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





# %%
# generaal MCMC sampling function given target, method , adapted, scale, ns, nb, x and seed 
# return samples and the number of logpdf in the simulation
def MCMC_sampling(target, method, adapted, scale, Ns, Nb, x0, seed):
    if hasattr(x0, '__module__') and x0.__module__.startswith("cuqi.distribution"):
        x0 = x0.sample().to_numpy()

    pr = cProfile.Profile()
    pr.enable()
    np.random.seed(seed)
    if method == MH:
        sampler = method(target=target, scale=scale, x0=x0)
        if adapted:
            x = sampler.sample_adapt(Ns, Nb)
        else:
            x = sampler.sample(Ns, Nb)
    elif method == NUTS:
        sampler = method(target, x0)
        x = sampler.sample(Ns, Nb)
    else:
        sampler = method(target, scale, x0)
        x = sampler.sample(Ns, Nb)
    pr.disable()
    # s = io.StringIO()
    # sortby = SortKey.PCALLS
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # lines = s.getvalue().split('\n')
    # idx = ['distribution\_custom.py:227(_donut_logpdf_func)' in line for line in lines].index(True)

    return x, pr
    # return x, lines[idx].split()[0]

# %%
def precompute_samples(target, scale, Ns, Nb, x0, seed):
    np.random.seed(seed)
    if isinstance(scale, float):
        scale = np.full(5, scale)
    if isinstance(Ns, int):
        Ns = np.full(5, Ns)
    if isinstance(Nb, int):
        Nb = np.full(5, Nb)
    
    samples = {}
    pr = {}
    samples['MH_fixed'], pr['MH_fixed']= MCMC_sampling(target=target, method=MH, adapted=False, scale=scale[0], Ns=Ns[0], Nb=Nb[0], x0=x0, seed=seed)
    samples['MH_adapted'], pr['MH_adapted'] = MCMC_sampling(target=target, method=MH, adapted=True, scale=scale[1], Ns=Ns[1], Nb=Nb[1], x0=x0, seed=seed)
    samples['ULA'], pr['ULA'] = MCMC_sampling(target=target, method=ULA, adapted=False, scale=scale[2], Ns=Ns[2], Nb=Nb[2], x0=x0, seed=seed)
    samples['MALA'], pr['MALA'] = MCMC_sampling(target=target, method=MALA, adapted=False, scale=scale[3], Ns=Ns[3], Nb=Nb[3], x0=x0, seed=seed)
    samples['NUTS'], pr['NUTS'] = MCMC_sampling(target=target, method=NUTS, adapted=False, scale=scale[4], Ns=Ns[4], Nb=Nb[4], x0=x0, seed=seed)

    #logpdf = count_function(pr,"logpdf")
    

    return samples,pr,scale,Ns,Nb


# %%

def count_function(pr,string):
    counter = np.zeros(5)
    for i in range(5):
        s = io.StringIO()
        sortby = SortKey.PCALLS
        ps = pstats.Stats(list(pr.values())[i], stream=s).sort_stats(sortby)
        ps.print_stats()
        lines = s.getvalue().split('\n')
       
        # Check if the string is in the output and extract it if found
        # search_string = 'logpdf'  
        search_string = string 
        if any(search_string in line for line in lines):
            idx = [search_string in line for line in lines].index(True)

            counter[i] = lines[idx].split()[0]
        
    return counter

#%%
## compute ess for all sampling methods 
def compute_ESS(samples):
    ess = np.zeros((5, 2))  # Initialize the array for ESS values
    
    # Extract the ESS from the precomputed samples
    ess[0] = samples['MH_fixed'].compute_ess()
    ess[1] = samples['MH_adapted'].compute_ess()
    ess[2] = samples['ULA'].compute_ess()
    ess[3] = samples['MALA'].compute_ess()
    ess[4] = samples['NUTS'].compute_ess()
    return ess


# def compute_AR(samples):
def compute_AR(samples):
    ar= np.zeros((5, 2))  # Initialize the array for ESS values
    
    # Extract the AR from the precomputed samples
    ar[0] = samples['MH_fixed'].acc_rate
    ar[1] = samples['MH_adapted'].acc_rate
    ar[2] = samples['ULA'].acc_rate
    ar[3] = samples['MALA'].acc_rate
    ar[4] = (len(np.unique(samples['NUTS'].samples[0])))/ (len(samples['NUTS'].samples[0]))
    return ar

# def compute_Rhat(data):
def compute_Rhat(samples,data):
    rhat= np.zeros((5, 2))  # Initialize the array for ESS values
    
    print()
    # Extract the Rhat from the precomputed samples
    rhat[0] = samples['MH_fixed'].compute_rhat([item["MH_fixed"] for item in data])
    rhat[1] = samples['MH_adapted'].compute_rhat([item["MH_adapted"] for item in data])
    rhat[2] = samples['ULA'].compute_rhat([item["ULA"] for item in data])
    rhat[3] = samples['MALA'].compute_rhat([item["MALA"] for item in data])
    rhat[4] = samples['NUTS'].compute_rhat([item["NUTS"] for item in data])
    return rhat






