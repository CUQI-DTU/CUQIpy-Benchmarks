import numpy as np
from cuqi.distribution import Distribution, DistributionGallery, Gaussian, JointDistribution, UserDefinedDistribution
from cuqi.model import LinearModel
from ._benchmarks import Benchmarks

class SimplestBip(Benchmarks):
  def __init__(self,dim = 2, matrix = np.array([[1.0, 1.0]]),noise = 0.1,data = [1,5,1.5],**kwargs):
    x = Gaussian(np.zeros(dim), 2.5)
    A = LinearModel(matrix)
    model = Gaussian(A.forward(x),noise)
    super().__init__(model_type = "bayesian",dim = 2,prior_distribution = x,forward_operator =  A, additive_noise = noise,model = model,data = data, **kwargs)
  
  