import numpy as np
import cuqi 
from cuqi.distribution import Distribution, DistributionGallery, Gaussian, JointDistribution, UserDefinedDistribution
from cuqi.problem import BayesianProblem
from ._benchmarks import Benchmarks

class HeatStep(Benchmarks):
  def __init__(self,steps=3,nodes = 30, t_max = 0.02**kwargs):
    data = cuqi.array.CUQIarray(np.load("data/data_heat.npy"))
    
    model, _, _ = Heat1D(
        dim=nodes,
        endpoint=1,
        max_time=t_max,
        field_type="Step",
        field_params={"n_steps": steps},
    ).get_components()

    mean = 0
    std = 1.2
    x = Gaussian(mean, std**2, geometry=model.domain_geometry)

    sigma_noise = 0.02
    y = Gaussian(mean=model(x), cov=sigma_noise**2, geometry=model.range_geometry)

    


    super().__init__(model_type = "bayesian", dim = steps, prior_distribution = x, model = y,  data = data,**kwargs)
  
  