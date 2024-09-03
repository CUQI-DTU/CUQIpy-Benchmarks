import numpy as np
from cuqi.distribution import Distribution, DistributionGallery, Gaussian, JointDistribution, UserDefinedDistribution
 
class Benchmarks(UserDefinedDistribution):
    def __init__(self,model_type,dim = None,log_pdf = None,grad_logpdf = None,**kwargs):
            # Init from abstract distribution class
            if model_type == "distribution":
                super().__init__(dim = dim,logpdf_func = log_pdf, gradient_func = grad_logpdf, **kwargs)

    

    