import os
import matplotlib.pyplot as plt
import dolfin as dl
import numpy as np
from cuqi import geometry
from cuqi.model import PDEModel
from cuqi.distribution import Gaussian,Lognormal, Posterior
from cuqi.array import CUQIarray
from cuqipy_fenics.pde import SteadyStateLinearFEniCSPDE
from cuqipy_fenics.geometry import FEniCSContinuous
import cuqipy_fenics
ufl = cuqipy_fenics.utilities._LazyUFLLoader()
from .._benchmarks import Benchmarks
from .StepExpansion import StepExpansion

class Wolfgang(Benchmarks):
  def __init__(self,n_steps = 64, n_mesh = 1024, n_obs = 169,**kwargs):

    self.n_obs = n_obs
    # Get the absolute path to the project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(project_dir, '../data', 'data_wolfgang.npy')

    # Load the data
    data = CUQIarray(np.load(file_path))

    # Define the mesh
    n_ksi_1 = int(np.sqrt(n_mesh)) # number of vertices on the ksi_1 dimension
    n_ksi_2 = int(np.sqrt(n_mesh)) # number of vertices on the ksi_2 dimension
    mesh = dl.UnitSquareMesh(n_ksi_1, n_ksi_2) # create FEniCS mesh

    self.solution_function_space = dl.FunctionSpace(mesh, 'Lagrange', 1) # function space for solution u
    self.parameter_function_space = dl.FunctionSpace(mesh, 'DG', 0) # function space for solution a

    # Dirichlet boundary condition
    dirichlet_bc_expr = dl.Expression("0", degree=1)
    dirichlet_bc = dl.DirichletBC(self.solution_function_space,
                                  dirichlet_bc_expr,
                                  u_boundary) 
    # Forcing term
    self.f = dl.Constant(10)

    # FEniCS measure for integration
    self.dksi = dl.Measure('dx', domain=mesh)

    # Define Poisson equation
    self.PDE = SteadyStateLinearFEniCSPDE( 
        self.form,
        mesh, 
        parameter_function_space=self.parameter_function_space,
        solution_function_space=self.solution_function_space,
        dirichlet_bcs=dirichlet_bc,
        observation_operator=self.observation)
    
    # Define CUQI geometry 
    fenics_continuous_geo = FEniCSContinuous(self.parameter_function_space,
                                            labels=['$\\xi_1$', '$\\xi_2$'])
    self.domain_geometry = StepExpansion(fenics_continuous_geo,
                                        num_steps=n_steps)
    # self.range_geometry = FEniCSContinuous(self.solution_function_space,
    #                                   labels=['$\\xi_1$', '$\\xi_2$'])
    grid_x = np.linspace(0, 1, 15)[1:-1]
    grid_y = np.linspace(0, 1, 15)[1:-1]
    self.range_geometry = geometry.Continuous2D((grid_x, grid_y))
    
    


    # prior distribution

    mean = 0
    std = 2
    a = Lognormal(np.zeros(64), std**2, geometry=self.domain_geometry)

    # model 
    self.cuqi_model = PDEModel(PDE = self.PDE, domain_geometry=self.domain_geometry, range_geometry=self.range_geometry)
    y = Gaussian(mean=self.cuqi_model(a), cov=1, geometry = self.range_geometry)

    super().__init__(model_type = "bayesian", dim = n_steps, prior_distribution = a, model = y,  data = data, finite_gradient = False,**kwargs)

  
  # The weak form of the PDE
  def form(self,a,u,p):
    return a*ufl.inner(ufl.grad(u), ufl.grad(p))*self.dksi - self.f*p*self.dksi
  

  # defining the observation functions
  def observation(self,par,obs):
      z = []
      val = int(np.sqrt(self.n_obs)) + 1
      for i in range(1,val):
        for j in range(1,val):
          z.append(obs([i /val,j /val]))
      return np.array(z)
  
def u_boundary(ksi,on_boundary):
    return on_boundary
     

