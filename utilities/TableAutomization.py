# %% Imports and library configurations
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

# %% General MCMC sampling function
def MCMC_sampling(target, method, adapted, scale, Ns, Nb, x0=None, seed=None):

    """
    Perform MCMC sampling given a target distribution, method, and parameters.
    
    Parameters:
    target  : cuqi.distribution.Distribution object
    method  : cuqi.sampler.Sampler class (MH, CWMH, ULA, etc.)
    adapted : boolean, if the MH sampler is adapted or not
    scale   : float or array, scaling factor for the sampler
    Ns      : int, number of samples
    Nb      : int, number of burn-ins
    x0      : initial state, either an array or a CUQI distribution
    seed    : int, random seed

    Returns:
    x       : cuqi.samples.Samples, samples generated by the sampler
    pr      : cProfile.Profile, profiling object to analyze the performance
    """
    if hasattr(x0, '__module__') and x0.__module__.startswith("cuqi.distribution"):
        x0 = x0.sample().to_numpy()

    pr = cProfile.Profile()
    pr.enable()
    try:
        np.random.seed(seed)
        if method == NUTS: 
            sampler = method(target = target, x0 = x0)
        else:
            sampler = method(target = target, scale = scale, x0 = x0)
        if adapted:
            x = sampler.sample_adapt(Ns,Nb)
        else: 
            x = sampler.sample(Ns,Nb)

    finally:
        pr.disable()
    return x, pr

# %% Precompute samples function
def precompute_samples(target, scale, Ns, Nb, x0=None, seed=12, selected_methods = ["MH_fixed",  "CWMH" ,"ULA", "MALA", "NUTS"]):

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
    np.random.seed(seed)
       
    
    if isinstance(scale, float):
        scale = np.full(len(selected_methods), scale)
    if isinstance(Ns, int):
        Ns = np.full(len(selected_methods), Ns)
    if isinstance(Nb, int):
        Nb = np.full(len(selected_methods), Nb)
    
    samples = {}
    pr = {}
    
    # Dictionary to map method names to their corresponding parameters and functions
    method_mapping = {
        'MH_fixed': (MH, False),
        'MH_adapted': (MH, True),
        'CWMH': (CWMH, False),
        'ULA': (ULA, False),
        'MALA': (MALA, False),
        'NUTS': (NUTS, False)
    }

    # Loop over selected methods and compute samples
    for idx, method in enumerate(selected_methods):
        if method in method_mapping:
            mcmc_method, adapted = method_mapping[method]
            samples[method], pr[method] = MCMC_sampling(target, mcmc_method, adapted, scale[idx], Ns[idx], Nb[idx], x0, seed)
    
    return samples, pr, scale, Ns, Nb



# %% Function to count function calls in profiling results
def count_function(pr, string):
    """
    Count occurrences of a specific function call in profiling results.
    
    Parameters:
    pr     : dict, containing profiling objects for each MCMC method
    string : str, function name to count in the profiling results

    Returns:
    counter : dict, containing counts of the specified function calls for each method
    """
    counter = {}

    for method in pr.keys():
        s = io.StringIO()
        ps = pstats.Stats(pr[method], stream=s).sort_stats(SortKey.PCALLS)
        ps.print_stats()
        lines = s.getvalue().split('\n')
        
        if any(string in line for line in lines):
            idx = [string in line for line in lines].index(True)
            counter[method] = int(lines[idx].split()[0])
        else:
            counter[method] = 0  # If the function was not found, set the count to 0

    return counter


# %% Compute ESS for all sampling methods
def compute_ESS(samples):
    """Compute effective sample size (ESS) for the selected sampling methods."""
    ess = {}
    
    for method in samples.keys():
        ess[method] = samples[method].compute_ess()
    
    return ess

# %% Compute acceptance rate for all sampling methods
def compute_AR(samples):
    """Compute acceptance rate (AR) for the selected sampling methods."""
    ar = {}

    for method in samples.keys():
        if method == 'NUTS':
            ar[method] = len(np.unique(samples[method].samples[0])) / len(samples[method].samples[0])
        else:
            ar[method] = samples[method].acc_rate
    
    return ar


# %% Utility function to safely access array elements
def safe_access(value, index=None):
    """Round the element at the specified index or the scalar value to 3 decimals."""
    if isinstance(value, (list, np.ndarray)):
        return round(value[index], 3)
    else:
        return round(value, 3)




def compute_Rhat(samples, data):
    """
    Compute Rhat statistic for convergence diagnostics across multiple chains.
    
    Parameters:
    samples : dict, containing samples for each MCMC method
    data    : list of dicts, containing samples from different chains for each method

    Returns:
    rhat : dict, containing Rhat values for each sampling method
    """
    rhat = {}

    for method in samples.keys():
        rhat[method] = samples[method].compute_rhat([item[method] for item in data])

    return rhat


def create_comparison(target , scale, Ns, Nb , x0 = None, seed =None, chains = 2, selected_criteria= ["ESS", "AR", "LogPDF", "Gradient","Rhat"], selected_methods =["MH_fixed", "CWMH", "ULA", "MALA", "NUTS"]):
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
    samples, pr,  scale, Ns, Nb = precompute_samples(target, scale, Ns, Nb, x0, seed, selected_methods)

    df_dict = {
        "Method": selected_methods,
        "Samples": [Ns[i] for i in range(len(selected_methods))],
        "Burn-ins": [Nb[i] for i in  range(len(selected_methods))],
        #"Scale": [scale[i] for i in  range(len(selected_methods))]
        "Scale": [scale[i] if i != len(selected_methods) - 1 else math.nan for i in range(len(selected_methods))]


    }

    # Conditionally compute and add the selected metrics to the DataFrame dictionary
    if "ESS" in selected_criteria:
        ess = compute_ESS(samples)
        df_dict["ESS(v0)"] = [safe_access(ess[method], 0) for method in selected_methods]
        df_dict["ESS(v1)"] = [safe_access(ess[method], 1) for method in selected_methods]

    if "AR" in selected_criteria:
        ar = compute_AR(samples)
        df_dict["AR"] = [safe_access(ar[method]) for method in selected_methods]

    if "LogPDF" in selected_criteria:
        logpdf = count_function(pr, "logpdf")
        df_dict["LogPDF"] = [logpdf[method] for method in selected_methods]
        df_dict['LogPDF'] = [int(x) if pd.notnull(x) else '-' for x in df_dict['LogPDF']]
    
    
    if "Gradient" in selected_criteria:
        gradient = count_function(pr, "_gradient")
        df_dict["Gradient"] = [gradient[method] for method in selected_methods]
        df_dict['Gradient'] = [int(x) if pd.notnull(x) else '-' for x in df_dict['Gradient']]

    if "Rhat" in selected_criteria:
        if hasattr(target,'prior'):
            x0 = target.prior
        if hasattr(x0, '__module__') and x0.__module__.startswith("cuqi.distribution"):
            data = []
            for i in range(chains - 1):
                chain_samples, _, _, _, _ = precompute_samples(target, scale, Ns, Nb, x0, i, selected_methods)
                data.append(chain_samples)
            rhat = compute_Rhat(samples, data)
            df_dict["Rhat(v0)"] = [safe_access(rhat[method], 0) for method in selected_methods]
            df_dict["Rhat(v1)"] = [safe_access(rhat[method], 1) for method in selected_methods]

       

    
    # Generate sampling plot
    plot = plot_sampling(samples, target, selected_methods)

    # Initialize the DataFrame dictionary
    
    # Create the DataFrame
    df = pd.DataFrame(df_dict)

    # Optional: Replace None values with "-"
    df = df.fillna("-")

    # Display the DataFrame without the index
    return df, plot

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

# %% Plot the sampling results
def plot_sampling(samples, target, selected_methods):
    """Plot the sampling results for visual comparison."""
    # Determine the number of selected methods
    num_methods = len(selected_methods)
    
    # Create a figure with subplots based on the number of selected methods
    num_cols = 3  # We can fit up to 3 plots per row
    num_rows = (num_methods + num_cols - 1) // num_cols  # Calculate required rows
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(6*num_cols, 6*num_rows))  # Adjust the figure size as needed
    
    # If there's only one row, axs will not be a 2D array, so we need to handle that case
    if num_rows == 1:
        axs = np.expand_dims(axs, axis=0)
    
    # Flatten axs to iterate easily if the number of plots is less than the grid size
    axs = axs.flatten()

    # Loop through each selected method and plot the corresponding samples
    for i, method in enumerate(selected_methods):
        k = max(4, np.max(np.abs(samples[method].samples)))
        
        # Set the current axes to the correct subplot
        plt.sca(axs[i])
        
        # Plot the target distribution first
        plot_pdf_2D(target, -k, k, -k, k)
        
        # Plot the MCMC samples on top of the target distribution
        samples[method].plot_pair(ax=axs[i])
        
        axs[i].set_title(f'{method.replace("_", " ").title()} Samples')
    
    # Hide any unused subplots if fewer methods are selected
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.close(fig)

    return fig, axs

#%%
def print_table(df):
    
    

    # Create a PrettyTable object
    table = PrettyTable()

    # Add columns to the table
    table.field_names = df.columns.tolist()
    for row in df.itertuples(index=False):
        table.add_row(row)

    # Print the table
    print(table)

#%%
def show_plot(fig):
    fig.savefig("output_plot.png")
    display(Image(filename="output_plot.png"))