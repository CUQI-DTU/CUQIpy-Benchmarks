# CUQIpy-Benchmarks

This repository contains benchmark scripts for the CUQIpy library, focusing on assessing the performance of various sampling methods. The benchmarks are designed to evaluate the effectiveness and efficiency of sampling algorithms implemented in CUQIpy across different problem setups.


To run the benchmarks, you need Python 3.7 or higher and the latest version of CUQIpy installed. Install CUQIpy via pip:

```bash
pip install cuqipy
```

Clone this repository to your local machine:

```bash
git clone https://github.com/CUQI-DTU/CUQIpy-Benchmarks.git
cd CUQIpy-Benchmarks
```

To run the Jupyter notebook scripts, navigate to the desired benchmark directory: 

```bash
jupyter notebook benchmarks/table-donut.ipynb
```

The directory structure of this repository is as follows:

```plaintext
CUQIpy-Benchmarks/
│
├── benchmarks/
    └── ...
├── demos/
│   ├── table-donut.ipynb        # Jupyter notebook for donut distribution
│   ├── table-banana.ipynb       # Jupyter notebook for banana distribution
│   └── ...
│
├── sandbox/
    └── ...
└── utilities/
    ├── TableAutomization.py     # Module for automating table generation
    ├── __init__.py              # Init file for the utilities package
```
Using the `utilities/` directory contains helper module `TableAutomization.py` this repository analyses different sampling methods from different points of view. 
In this context, the definition of a __benchmark__ is a fully specified distribution that can be sampled. There are 3 types of benchmark problems that will be worked on:

__Type 0__ : 
- simple target distributions
- this is not an inverse problem
- givn that there is already a target distribution
- the user choosees the initial point/ the initial point distribution
__Type 1__ :
- inverse problems
- likelihood, prior and  posterior
- no need for a initial point
__Type 2__ :
- PDE-type inverse problems
- given a fixed likelihood, prior and posterior for the PDE
- analyzes different sampling methods for a fixed pde problem




