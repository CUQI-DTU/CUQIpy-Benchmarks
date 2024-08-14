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


# %%


def safe_access(array, index):
    return round(array[index], 3) if len(array) > index else None



# %%
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

# %%
def MCMC_sampling(target, method,adapted , scale, Ns ,Nb, x0, seed):
  np.random.seed(seed)
  if method == MH:
    sampler = method(target=target, scale=scale,x0=x0)
    if adapted:
       return sampler.sample_adapt(Ns, Nb)
    return sampler.sample(Ns, Nb)
  else: 
    if method == NUTS: 
      sampler = method(target, x0)
      return sampler.sample(Ns,Nb)
    sampler = method(target, scale, x0)
    return sampler.sample(Ns,Nb)

# %%
def safe_access(array, index):
    return round(array[index], 3) if len(array) > index else None

# %%
def create_table(target,scale,Ns,Nb,x0,seed):
    if isinstance(scale, float):
        scale = np.full(5, scale)
    if isinstance(Ns, int):
        Ns = np.full(5, Ns)
    if isinstance(Nb, int):
        Nb = np.full(5, Nb)
    
    ess_MH_fixed = MCMC_sampling(target = target, method=MH, adapted = False ,scale=scale[0],Ns=Ns[0],Nb=Nb[0],x0=x0,seed=seed).compute_ess()
    ess_MH_adapted = MCMC_sampling(target = target,method=MH, adapted = True,scale=scale[1],Ns=Ns[1],Nb=Nb[1],x0=x0,seed=seed).compute_ess()
    ess_ULA = MCMC_sampling(target = target,method= ULA, adapted = False ,scale=scale[2],Ns=Ns[2],Nb=Nb[2],x0=x0,seed=seed).compute_ess()
    ess_MALA = MCMC_sampling(target = target,method=MALA, adapted = False,scale=scale[3],Ns=Ns[3],Nb=Nb[3],x0=x0,seed=seed).compute_ess()
    ess_NUTS = MCMC_sampling(target = target,method=NUTS, adapted = False ,scale=scale[4],Ns=Ns[4],Nb=Nb[4],x0=x0,seed=seed).compute_ess()

    ess_df = pd.DataFrame({
        "Sampling Method": ["MH_fixed", "MH_adapted", "ULA", "MALA", "NUTS"],
        "No. of Samples": [Ns[0], Ns[1], Ns[2], Ns[3], Ns[4]],
        "No. of Burn-ins": [Nb[0], Nb[1], Nb[2], Nb[3], Nb[4]],
        "Scaling Factor": [scale[0], scale[1], scale[2], scale[3], scale[4]],
        "ESS (v0)":  [safe_access(ess_MH_fixed, 0), safe_access(ess_MH_adapted, 0), safe_access(ess_ULA, 0), safe_access(ess_MALA, 0), safe_access(ess_NUTS, 0)],
        "ESS (v1)": [safe_access(ess_MH_fixed, 1), safe_access(ess_MH_adapted, 1), safe_access(ess_ULA, 1), safe_access(ess_MALA, 1), safe_access(ess_NUTS, 1)]
    })

    # Optional: Replace None values with "-"
    ess_df = ess_df.fillna("-")

    # Display the DataFrame without the index
    return ess_df

  





