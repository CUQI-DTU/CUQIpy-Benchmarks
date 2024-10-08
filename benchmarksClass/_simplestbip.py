import numpy as np
from cuqi.distribution import Distribution, DistributionGallery, Gaussian, JointDistribution, UserDefinedDistribution
from cuqi.model import LinearModel
from cuqi.problem import BayesianProblem
from ._benchmarks import Benchmarks

class SimplestBip(Benchmarks):
  def __init__(self,dim = 2, matrix = np.array([[1.0, 1.0]]),noise = 0.1,data = 3,**kwargs):
    x = Gaussian(np.zeros(dim), 2.5)
    A = LinearModel(matrix)
    y = Gaussian(A.forward(x),noise)
    super().__init__(model_type = "bayesian", dim = dim, prior_distribution = x,model = y, data = data,**kwargs)
  
  