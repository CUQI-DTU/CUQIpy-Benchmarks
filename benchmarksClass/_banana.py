import numpy as np
from cuqi.distribution import Gaussian
from ._benchmarks import Benchmarks

class Banana(Benchmarks):
  def __init__(self, m0 = np.array([0, 4]),S0 = np.array([[1, 0.5], [0.5, 1]]),**kwargs):
    self.G0 = Gaussian(m0, S0)
    self.a, self.b = 2, 0.2

    super().__init__(model_type = "distribution", dim = 2,log_pdf = self._banana_logpdf_func,grad_logpdf = self._banana_grad_logpdf,**kwargs)
  
  def _banana_logpdf_func(self, x):
        if len(x.shape) == 1:
            x = x.reshape((1, 2))
        y = np.zeros((x.shape[0], self.dim))
        y[:, 0] = x[:, 0]/self.a
        y[:, 1] = x[:, 1]*self.a + self.a*self.b*(x[:, 0]**2 + self.a**2)
        return self.G0.logpdf(y)

  def _banana_grad_logpdf(self, x):
        if len(x.shape) == 1:
            x = x.reshape((1, 2))
        y = np.zeros((x.shape[0], self.dim))
        y[:, 0] = x[:, 0]/self.a
        y[:, 1] = x[:, 1]*self.a + self.a*self.b*(x[:, 0]**2 + self.a**2)
        grad = self.G0.gradient(y)
        gradx0, gradx1 = grad[0, :]/self.a + grad[1, :]*self.a*self.b*2*x[:, 0], grad[1, :]*self.a
        grad[0, :], grad[1, :] = gradx0, gradx1
        if x.shape[0] == 1:
            return grad.flatten()
        else:
            return grad
