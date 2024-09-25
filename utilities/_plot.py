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

class Plot():
    
    def __init__(self, target = None, samples = None, selected_methods =None, dim =None ):
        self.samples = samples
        self.dim = dim
        self.target = target
        self.selected_methods = selected_methods
    
    def plot_sampling(self):
        """Plot the sampling results for visual comparison."""
        # Determine the number of selected methods
        num_methods = len(self.selected_methods)
        
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
        for i, method in enumerate(self.selected_methods):
            k = max(4, np.max(np.abs(self.samples[method]["samples"].samples)))
            # k = max(10, np.max(np.abs(samples[method].samples)))
            
            # Set the current axes to the correct subplot
            plt.sca(axs[i])
            
            # Plot the target distribution first
            self.plot_pdf_2D(-k, k, -k, k)
            
            # Plot the MCMC samples on top of the target distribution
            self.samples[method]["samples"].plot_pair(ax=axs[i])
            
            axs[i].set_title(f'{method.replace("_", " ").title()} Samples')
        
        # Hide any unused subplots if fewer methods are selected
        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.close(fig)

        self.show_plot(fig)

        return fig, axs
    
    def plot_pdf_2D(self, x1_min, x1_max, x2_min, x2_max, N2=201, **kwargs):
        N2 = 201
        ls1 = np.linspace(x1_min, x1_max, N2)
        ls2 = np.linspace(x2_min, x2_max, N2)
        grid1, grid2 = np.meshgrid(ls1, ls2)
        distb_pdf = np.zeros((N2,N2))
        for ii in range(N2):
            for jj in range(N2):
                distb_pdf[ii,jj] = np.exp(self.target.logd(np.array([grid1[ii,jj], grid2[ii,jj]]))) 
        self.plot2d(distb_pdf, x1_min, x1_max, x2_min, x2_max, N2, **kwargs)
    
    @staticmethod
    def plot2d(val, x1_min, x1_max, x2_min, x2_max, N2=201, **kwargs):
        # plot
        pixelwidth_x = (x1_max-x1_min)/(N2-1)
        pixelwidth_y = (x2_max-x2_min)/(N2-1)

        hp_x = 0.5*pixelwidth_x
        hp_y = 0.5*pixelwidth_y

        extent = (x1_min-hp_x, x1_max+hp_x, x2_min-hp_y, x2_max+hp_y)

        plt.imshow(val, origin='lower', extent=extent, **kwargs)
        plt.colorbar()

    def show_plot(self, fig):
        fig.savefig("output_plot.png")
        display(Image(filename="output_plot.png"))
        
