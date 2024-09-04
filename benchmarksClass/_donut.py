import numpy as np
from ._benchmarks import Benchmarks

class Donut(Benchmarks):
  def __init__(self,radius = 2.6,sigma2 = 0.033,**kwargs):
    self.radius = radius
    self.sigma2 = sigma2

    super().__init__(model_type = "distribution", dim = 2,log_pdf = self._donut_logpdf_func,grad_logpdf = self._donut_grad_logpdf,**kwargs)
  
  def _donut_logpdf_func(self, x):
      if len(x.shape) == 1:
          x = x.reshape((1, 2))
      r = np.linalg.norm(x, axis=1)
      return - (r - self.radius)**2 / self.sigma2

  def _donut_grad_logpdf(self, x):
      if len(x.shape) == 1:
          x = x.reshape((1, 2))
      r = np.linalg.norm(x, axis=1)
      idx = np.argwhere(r==0)
      r[idx] = 1e-16
      grad = np.array([(x[:, 0]*((self.radius/r)-1)*2)/self.sigma2, \
                        (x[:, 1]*((self.radius/r)-1)*2)/self.sigma2])
      if x.shape[0] == 1:
          return grad.flatten()
      else:
          return grad
