# %%
from SampleComputation import SampleComputation as SC

# %%

#rounds the array element at index by 3 decimals 
def safe_access(array, index):
    return round(array[index], 3) 


#main function- creates a table given a posterior distribution, with the ess values 
def create_comparison(target,scale,Ns,Nb,x0,seed,chains):
    #in case scale, nb or ns are scalars 
    
    
    # compute ess 
    samples, pr, scale, Ns, Nb = SC.precompute_samples(target,scale,Ns,Nb,x0,seed)
    ess = SC.compute_ESS(samples)
    ar = SC.compute_AR(samples)
    logpdf = SC.count_function(pr,"logpdf")
    gradient = SC.count_function(pr,"_gradient")
    plot = SC.plot_sampling(samples, target)




     # Initialize the DataFrame dictionary
    df_dict = {
        "Sampling Method": ["MH_fixed", "MH_adapted", "ULA", "MALA", "NUTS"],
        "No. of Samples": [Ns[0], Ns[1], Ns[2], Ns[3], Ns[4]],
        "No. of Burn-ins": [Nb[0], Nb[1], Nb[2], Nb[3], Nb[4]],
        "Scaling Factor": [scale[0], scale[1], scale[2], scale[3], scale[4]],
        "ESS (v0)": [safe_access(ess[0], 0), safe_access(ess[1], 0), safe_access(ess[2], 0), safe_access(ess[3], 0), safe_access(ess[4], 0)],
        "ESS (v1)": [safe_access(ess[0], 1), safe_access(ess[1], 1), safe_access(ess[2], 1), safe_access(ess[3], 1), safe_access(ess[4], 1)],
        "AR": [safe_access(ar[0], 1), safe_access(ar[1], 1), safe_access(ar[2], 1), safe_access(ar[3], 1), safe_access(ar[4], 1)],
    }

    # Check if x0 is a CUQI distribution object
    if hasattr(x0, '__module__') and x0.__module__.startswith("cuqi.distribution"):
        # Initialize data for Rhat calculation
        data = []
        for i in range(chains - 1):
            data.append(SC.precompute_samples(target, scale, Ns, Nb, x0, seed)[0])
        rhat = SC.compute_Rhat(samples, data)

        # Add Rhat values to the DataFrame dictionary
        df_dict["Rhat (v0)"] = [safe_access(rhat[0], 0), safe_access(rhat[1], 0), safe_access(rhat[2], 0), safe_access(rhat[3], 0), safe_access(rhat[4], 0)]
        df_dict["Rhat (v1)"] = [safe_access(rhat[0], 1), safe_access(rhat[1], 1), safe_access(rhat[2], 1), safe_access(rhat[3], 1), safe_access(rhat[4], 1)]

    # Continue adding other columns
    df_dict["LogPDF"] = [logpdf[0], logpdf[1], logpdf[2], logpdf[3], logpdf[4]]
    df_dict["Gradient"] = [gradient[0], gradient[1], gradient[2], gradient[3], gradient[4]]

    # Create the DataFrame
    df = pd.DataFrame(df_dict)

    # Optional: Replace None values with "-"
    df = df.fillna("-")

    # Display the DataFrame without the index
    return df, plot 

#%%
def create_table(target,scale,Ns,Nb,x0,seed,chains):
    #in case scale, nb or ns are scalars 
    
    
    # compute ess 
    samples, pr, scale, Ns, Nb = SC.precompute_samples(target,scale,Ns,Nb,x0,seed)
    ess = SC.compute_ESS(samples)
    ar = SC.compute_AR(samples)
    logpdf = SC.count_function(pr,"logpdf")
    gradient = SC.count_function(pr,"_gradient")

     # Initialize the DataFrame dictionary
    df_dict = {
        "Sampling Method": ["MH_fixed", "MH_adapted", "ULA", "MALA", "NUTS"],
        "No. of Samples": [Ns[0], Ns[1], Ns[2], Ns[3], Ns[4]],
        "No. of Burn-ins": [Nb[0], Nb[1], Nb[2], Nb[3], Nb[4]],
        "Scaling Factor": [scale[0], scale[1], scale[2], scale[3], scale[4]],
        "ESS (v0)": [safe_access(ess[0], 0), safe_access(ess[1], 0), safe_access(ess[2], 0), safe_access(ess[3], 0), safe_access(ess[4], 0)],
        "ESS (v1)": [safe_access(ess[0], 1), safe_access(ess[1], 1), safe_access(ess[2], 1), safe_access(ess[3], 1), safe_access(ess[4], 1)],
        "AR": [safe_access(ar[0], 1), safe_access(ar[1], 1), safe_access(ar[2], 1), safe_access(ar[3], 1), safe_access(ar[4], 1)],
    }

    # Check if x0 is a CUQI distribution object
    if hasattr(x0, '__module__') and x0.__module__.startswith("cuqi.distribution"):
        # Initialize data for Rhat calculation
        data = []
        for i in range(chains - 1):
            data.append(SC.precompute_samples(target, scale, Ns, Nb, x0, seed)[0])
        rhat = SC.compute_Rhat(samples, data)

        # Add Rhat values to the DataFrame dictionary
        df_dict["Rhat (v0)"] = [safe_access(rhat[0], 0), safe_access(rhat[1], 0), safe_access(rhat[2], 0), safe_access(rhat[3], 0), safe_access(rhat[4], 0)]
        df_dict["Rhat (v1)"] = [safe_access(rhat[0], 1), safe_access(rhat[1], 1), safe_access(rhat[2], 1), safe_access(rhat[3], 1), safe_access(rhat[4], 1)]

    # Continue adding other columns
    df_dict["LogPDF"] = [logpdf[0], logpdf[1], logpdf[2], logpdf[3], logpdf[4]]
    df_dict["Gradient"] = [gradient[0], gradient[1], gradient[2], gradient[3], gradient[4]]

    # Create the DataFrame
    df = pd.DataFrame(df_dict)

    # Optional: Replace None values with "-"
    df = df.fillna("-")

    # Display the DataFrame without the index
    return df 

    #%%
def print_table(df):
    df['LogPDF'] = df['LogPDF'].apply(lambda x: int(x) if pd.notnull(x) else '-')
    df['Gradient'] = df['Gradient'].apply(lambda x: int(x) if pd.notnull(x) else '-')

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
def plot_sampling(samples, target):

    # Perform MCMC sampling
    MH_fixed_samples = samples['MH_fixed']
    MH_adapted_samples = samples['MH_adapted']
    ULA_samples = samples['ULA']
    MALA_samples =samples['MALA']
    NUTS_samples = samples['NUTS']

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
    plt.close(fig)

    return fig,  axs


# %%
