import numpy as np
from cuqi.problem import BayesianProblem
from cuqi.distribution import Distribution, JointDistribution, UserDefinedDistribution, Gaussian
 
class Benchmarks(UserDefinedDistribution):
    def __init__(self,model_type,dim, log_pdf = None,grad_logpdf = None,prior_distribution = None,model = None, data = None, finite_gradient = False,**kwargs):
            # Init from abstract distribution class
            if model_type == "distribution":
                super().__init__(dim = dim,logpdf_func = log_pdf, gradient_func = grad_logpdf, **kwargs)
            
            elif model_type == "bayesian" :
                self.x0 = prior_distribution
                y = model
                BP = BayesianProblem(y, self.x0)
                BP.set_data(y = data) 
                if finite_gradient == True:
                     BP.posterior.enable_FD()
                self.posterior = BP.posterior
                super().__init__(dim = dim, logpdf_func = BP.posterior.logpdf, gradient_func = BP.posterior.gradient, **kwargs)