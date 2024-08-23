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

The `benchmarks/` directory contains the scripts and notebooks for various benchmarks, while the `utilities/` directory contains helper module `TableAutomization.py` that are used in the benchmark analysis.


