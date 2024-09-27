import dolfin as dl
import numpy as np
from cuqi.geometry import _WrappedGeometry
# import cuqipy_fenics 
from .Expression2D import Expression2D


class StepExpansion2D(_WrappedGeometry):
    """A geometry class that builds step expansion in 2D domain
    Parameters
    -----------
    geometry : cuqi.fenics.geometry.Geometry
        An input geometry on which the step expansion is built (the geometry must have a mesh attribute)

    num_steps: int
        Number of expansion terms to represent the step expansion realization (the num_steps have must be square of integer)


    Example
    -------
    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        from cuqi.fenics.geometry import StepExpansion2D, FEniCSContinuous
        from cuqi.distribution import Gaussian
        import dolfin as dl


        mesh = dl.UnitSquareMesh(32,32)
        V = dl.FunctionSpace(mesh, 'DG', 0)

        grid_x = np.linspace(0, 1, 15)[1:-1]
        grid_y = np.linspace(0, 1, 15)[1:-1]
        geometry = FEniCSContinuous(V, labels=['$\\xi_1$', '$\\xi_2$'])
        StepExpansionGeometry = StepExpansion2D(geometry, 
                                    num_steps=64)

        StepExpansionField = Gaussian(mean=np.zeros(StepExpansionGeometry.num_steps),
                    cov=np.eye(StepExpansionGeometry.num_steps),
                    geometry=StepExpansionGeometry)

        samples = StepExpansionField.sample()
        samples.plot()


    """

    def __init__(self, geometry, num_steps = 64): 
        super().__init__(geometry)
        if not hasattr(geometry, 'mesh'):
            raise NotImplementedError
        self._num_steps = num_steps
        self._build_basis() 
        
    @property
    def funvec_shape(self):
        """The shape of the geometry (shape of the vector representation of the
        function value)."""
        return self.geometry.funvec_shape
    
    @property
    def par_shape(self):
        return (self.num_steps,)


    @property
    def num_steps(self):
        return self._num_steps

    @property
    def function_space(self):
        return self.geometry.function_space
    
    @property
    def step_vec(self):
        return self._step_vec
    
    @property
    def physical_dim(self):
        """Returns the physical dimension of the geometry, e.g. 1, 2 or 3"""
        return self.geometry.physical_dim

    def __repr__(self) -> str:
        return "{} on {}".format(self.__class__.__name__,self.geometry.__repr__())

    def par2fun(self,a):
        return self.geometry.par2fun(self.par2field(a))

    def fun2vec(self,fun):
        """ Maps the function value (FEniCS object) to the corresponding vector
        representation of the function (ndarray of the function DOF values)."""
        return self.geometry.fun2vec(fun)
    
    def vec2fun(self,funvec):
        """ Maps the vector representation of the function (ndarray of the
        function DOF values) to the function value (FEniCS object)."""
        return self.geometry.vec2fun(funvec)

    def gradient(self, direction, wrt):
        direction = self.geometry.gradient(direction, wrt)
        return self._step_vec.T@direction
        
    def par2field(self, a):
        """Applies linear transformation of the parameters a to
        generate a realization of the step expansion (given that a is the coefficient of conductivity)"""

        if self._step_vec is None:
            self._build_basis() 
	   
        a = self._process_values(a)
        Ns = a.shape[-1]
        field_list = np.empty((self.geometry.par_dim,Ns))

        for idx in range(Ns):
            # For more details about the formulation below, see section 4.3 in
            # Chen, V., Dunlop, M. M., Papaspiliopoulos, O., & Stuart, A. M.
            # (2018). Dimension-robust MCMC in Bayesian inverse problems.
            # arXiv preprint arXiv:1803.03344.
            field_list[:,idx] = self.step_vec@(a[...,idx] )

        if len(field_list) == 1:
            return field_list[0]
        else:
            return field_list

    def _build_basis(self):
        """Builds the basis of step expansion"""
        n_ksi_1 = 32 # number of vertices on the ksi_1 dimension
        n_ksi_2 = 32 # number of vertices on the ksi_2 dimension
        mesh = dl.UnitSquareMesh(n_ksi_1, n_ksi_2) # create FEniCS mesh
        parameter_function_space = dl.FunctionSpace(mesh, 'DG', 0) # function space for solution a
        u = dl.Function(parameter_function_space)
    
        self._step_vec = np.zeros( [ u.vector().get_local().shape[0], self.num_steps ] )
        val = int(np.sqrt(self.num_steps))
        for i in range( self.num_steps ):   
            u.interpolate(Expression2D(degree=0, x_lim=[(i%val) /val , (i%val + 1) / val], y_lim=[i//val /val , (i//val + 1) / val]))
            self._step_vec[:,i] = u.vector().get_local()

        

        