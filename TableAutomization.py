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

#rounds the array element at index by 3 decimals 
def safe_access(array, index):
    return round(array[index], 3) 

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






# %%

#main function- creates a table given a posterior distribution, with the ess values 
def create_table(target,scale,Ns,Nb,x0,seed,chains=1):
    #in case scale, nb or ns are scalars 
    
    
    # compute ess 
    samples, pr, scale, Ns, Nb = precompute_samples(target,scale,Ns,Nb,x0,seed)
    ess = compute_ESS(samples)
    ar = compute_AR(samples)
    logpdf = count_function(pr,"logpdf")
    gradient = count_function(pr,"_gradient")
    if hasattr(x0, '__module__') and x0.__module__.startswith("cuqi.distribution"):
        data = []
        for i in range(chains-1):
            data.append(precompute_samples(target,scale,Ns,Nb,x0,seed)[0])
        rhat = compute_Rhat(samples,data)
        print(rhat)



    df = pd.DataFrame({
        "Sampling Method": ["MH_fixed", "MH_adapted", "ULA", "MALA", "NUTS"],
        "No. of Samples": [Ns[0], Ns[1], Ns[2], Ns[3], Ns[4]],
        "No. of Burn-ins": [Nb[0], Nb[1], Nb[2], Nb[3], Nb[4]],
        "Scaling Factor": [scale[0], scale[1], scale[2], scale[3], scale[4]],
        "ESS (v0)":  [safe_access(ess[0], 0), safe_access(ess[1], 0), safe_access(ess[2], 0), safe_access(ess[3], 0), safe_access(ess[4], 0)],
        "ESS (v1)": [safe_access(ess[0], 1), safe_access(ess[1], 1), safe_access(ess[2], 1), safe_access(ess[3], 1), safe_access(ess[4], 1)],
        "AR":[safe_access(ar[0], 1), safe_access(ar[1], 1), safe_access(ar[2], 1), safe_access(ar[3], 1), safe_access(ar[4], 1)],
        "LogPDF": [logpdf[0], logpdf[1], logpdf[2], logpdf[3], logpdf[4]],
        "Gradient": [gradient[0], gradient[1], gradient[2], gradient[3], gradient[4]]
    })

    # Optional: Replace None values with "-"
    df = df.fillna("-")

    # Display the DataFrame without the index
    return df

  

#%%
#plotting function 
def plot2d(val, x1_min, x1_max, x2_min, x2_max, N2=201, **kwargs):
    # plot
    pixelwidth_x = (x1_max-x1_min)/(N2-1)
    pixelwidth_y = (x2_max-x2_min)/(N2-1)

    hp_x = 0.5*pixelwidth_x
    hp_y = 0.5*pixelwidth_y

    extent = (x1_min-hp_x, x1_max+hp_x, x2_min-hp_y, x2_max+hp_y)

    plt.imshow(val, origin='lower', extent=extent, **kwargs)
    plt.colorbar()


def plot_pdf_2D(distb, x1_min, x1_max, x2_min, x2_max, N2=201, **kwargs):
    N2 = 201
    ls1 = np.linspace(x1_min, x1_max, N2)
    ls2 = np.linspace(x2_min, x2_max, N2)
    grid1, grid2 = np.meshgrid(ls1, ls2)
    distb_pdf = np.zeros((N2,N2))
    for ii in range(N2):
        for jj in range(N2):
            distb_pdf[ii,jj] = np.exp(distb.logd(np.array([grid1[ii,jj], grid2[ii,jj]]))) 
    plot2d(distb_pdf, x1_min, x1_max, x2_min, x2_max, N2, **kwargs)

def plot_pdf_1D(distb, min, max, **kwargs):
    grid = np.linspace(min, max, 1000)
    y = [distb.pdf(grid_point) for grid_point in grid]
    plt.plot(grid, y, **kwargs)


#%%
# function that given a target and  scale, Ns, Nb, x0, seed shows the plot distribution 
def plot_sampling(target, scale, Ns, Nb, x0, seed):
    if isinstance(scale, float):
        scale = np.full(5, scale)
    if isinstance(Ns, int):
        Ns = np.full(5, Ns)
    if isinstance(Nb, int):
        Nb = np.full(5, Nb)

    # Perform MCMC sampling
    MH_fixed_samples = MCMC_sampling(target=target, method=MH, adapted=False, scale=scale[0], Ns=Ns[0], Nb=Nb[0], x0=x0, seed=seed)
    MH_adapted_samples = MCMC_sampling(target=target, method=MH, adapted=False, scale=scale[1], Ns=Ns[1], Nb=Nb[1], x0=x0, seed=seed)
    ULA_samples = MCMC_sampling(target=target, method=ULA, adapted=False, scale=scale[2], Ns=Ns[2], Nb=Nb[2], x0=x0, seed=seed)
    MALA_samples = MCMC_sampling(target=target, method=MALA, adapted=False, scale=scale[3], Ns=Ns[3], Nb=Nb[3], x0=x0, seed=seed)
    NUTS_samples = MCMC_sampling(target=target, method=NUTS, adapted=False, scale=scale[4], Ns=Ns[4], Nb=Nb[4], x0=x0, seed=seed)

    # Create a figure with a 2x3 grid of subplots (2 rows, 3 columns)
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))  # Adjust the figure size as needed

    # Plot each sample in the appropriate subplot
    plt.sca(axs[0, 0])  # Set the current axes to the first subplot
    plot_pdf_2D(target, -4, 4, -4, 4)
    MH_fixed_samples.plot_pair(ax=axs[0, 0])
    axs[0, 0].set_title('MH Fixed Samples')

    plt.sca(axs[0, 1])  # Set the current axes to the second subplot
    plot_pdf_2D(target, -4, 4, -4, 4)
    MH_adapted_samples.plot_pair(ax=axs[0, 1])
    axs[0, 1].set_title('MH Adapted Samples')

    plt.sca(axs[0, 2])  # Set the current axes to the third subplot
    plot_pdf_2D(target, -4, 4, -4, 4)
    ULA_samples.plot_pair(ax=axs[0, 2])
    axs[0, 2].set_title('ULA Samples')

    plt.sca(axs[1, 0])  # Set the current axes to the fourth subplot
    plot_pdf_2D(target, -4, 4, -4, 4)
    MALA_samples.plot_pair(ax=axs[1, 0])
    axs[1, 0].set_title('MALA Samples')

    plt.sca(axs[1, 1])  # Set the current axes to the fifth subplot
    plot_pdf_2D(target, -4, 4, -4, 4)
    NUTS_samples.plot_pair(ax=axs[1, 1])
    axs[1, 1].set_title('NUTS Samples')

    # Hide the empty subplot (bottom right) if there are fewer than 6 plots
    fig.delaxes(axs[1, 2])

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plots
    plt.show()


# %%
