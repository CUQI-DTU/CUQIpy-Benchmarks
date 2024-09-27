# CUQIpy-Benchmarks

This repository contains benchmark scripts for the CUQIpy library, focusing on assessing the performance of various sampling methods. The benchmarks are designed to evaluate the effectiveness and efficiency of sampling algorithms implemented in CUQIpy across different problem setups.


To run the benchmarks, you need the latest version of CUQIpy installed. Install CUQIpy via pip:

```bash
pip install cuqipy
```

Click Release and download Source code (zip) and open it in your environment.

Or clone this repository to your local machine:

```bash
git clone https://github.com/CUQI-DTU/CUQIpy-Benchmarks.git
cd CUQIpy-Benchmarks
```

The directory structure of this repository is as follows:

```plaintext
CUQIpy-Benchmarks/
│
├── benchmarksClass/
    └── ...
├── demos/
│   ├── table-donut.ipynb        # Jupyter notebook for donut distribution
│   ├── table-banana.ipynb       # Jupyter notebook for banana distribution
│   └── ...
│
├── sandbox/
    └── ...
└── utilities/
    ├── _mcmcComparison.py     # Module for automating table generation
    ├── _criteria.py     # Module for automating table generation
    ├── _plot.py     # Module for automating table generation
    ├── __init__.py              # Init file for the utilities package
```
Using the `utilities/` directory contains helper modules . This repository analyses different sampling methods from different points of view. 
In this context, the definition of a __benchmark__ is a fully specified distribution that can be sampled. There are 2 types of benchmark problems that will be worked on:

__Type 0__ : 
- simple target distributions
- this is not an inverse problem
- givn that there is already a target distribution
- the user choosees the initial point/ the initial point distribution


__Type 1__ :
- inverse problems
- likelihood, prior and  posterior
- can be PDE problems



[CUQIpy_Benchmarks_Poster.pdf](https://github.com/user-attachments/files/17147237/DTU_Poster_tania_goia.pdf)
