import numpy as np
from cuqi.distribution import Distribution, DistributionGallery, Gaussian, JointDistribution, UserDefinedDistribution
from ._benchmarks import Benchmarks

class Sixmodal(Benchmarks):
  def __init__(self, **kwargs):
    super().__init__(model_type ="distribution", dim = 2, log_pdf =  self._sixmodal_log_func, grad_logpdf = self._sixmodal_grad_logpdf, **kwargs)
  
  def _sixmodal_log_func(self, x):
        if len(x.shape) == 1:
            x = x.reshape((1, 2))
        if -10 <= x[:,0] <= 10 and -10 <= x[:,1] <= 10:
            return (- x[:,0]**2 - ((np.sin(x[:, 1]))**(-5) - x[:, 0])**2)/2
        return float('-inf')

  def _sixmodal_grad_logpdf(self, x):
        if len(x.shape) == 1:
            x = x.reshape((1, 2))
        np.array(- 2*x[:, 0] + (np.sin(x[:, 1]))**(-5), 5 * (np.sin(x[:, 1])**(-5) - x[0]) * (np.sin(x[:, 1]))**(-6) * np.cos(x[:, 1]))
