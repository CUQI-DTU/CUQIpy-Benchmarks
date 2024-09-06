import numpy as np
from cuqi.problem import BayesianProblem
from cuqi.distribution import Distribution, JointDistribution, UserDefinedDistribution, Gaussian
 
class Benchmarks(UserDefinedDistribution):
    def __init__(self,model_type,dim, log_pdf = None,grad_logpdf = None,prior_distribution = None,model = None, noise=None, forward_operator = None,data = None, **kwargs):
            # Init from abstract distribution class
            if model_type == "distribution":
                super().__init__(dim = dim,logpdf_func = log_pdf, gradient_func = grad_logpdf, **kwargs)
            
            elif model_type == "bayesian" :
                x = prior_distribution
                y = model
                BP = BayesianProblem(y, x)
                BP.set_data(y = data) 
                super().__init__(dim = dim, logpdf_func = BP.posterior.logpdf, gradient_func = BP.posterior.gradient, **kwargs)