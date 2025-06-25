# Singular Spectrum Analysis Library (SSALib)

[![Tests](https://github.com/ADSCIAN/ssalib/actions/workflows/python-tests.yml/badge.svg)](https://github.com/ADSCIAN/ssalib/actions/workflows/python-tests.yml)
[![Python Version](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12|%203.13-blue)](https://www.python.org)
[![Coverage](https://img.shields.io/badge/coverage-97%25-green)](https://github.com/ADSCIAN/ssalib/actions)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Development Status](https://img.shields.io/badge/Development%20Status-beta-blue)](https://pypi.org/project/ssalib/)

## Overview

The Singular Spectrum Analysis Library (SSALib) is a Python package for
univariate (i.e., single) time series decomposition, designed for
multidisciplinary applications like natural sciences, economics, epidemiology,
and more. SSALib can be used to explore and extract trends, periodic patterns,
and noise from time series.

![decomposed_signal.png](https://raw.githubusercontent.com/ADSCIAN/ssalib/main/images/decomposed_signal.png)
_Figure: Example of Decomposition and Pattern Extraction (standardized) using
the Sea Surface Temperature Time Series_

## Key Features

- Univariate SSA implementation with both Broemhead & King and Vautard and Ghil
  approaches
- Multiple SVD solver options (NumPy, SciPy, scikit-learn)
- Monte Carlo SSA for significance testing
- Built-in visualization tools for analysis
- Include example datasets
- Comprehensive test coverage
- Type-annotated codebase

## Quick Start

### Requirements

- Python ≥ 3.9
- NumPy
- SciPy < 1.16.0
- Pandas
- Matplotlib
- Scikit-learn
- Statsmodels

### Installation

Use

```bash
pip install git+https://github.com/ADSCIAN/ssalib.git
```

or

```bach
pip install ssalib
```

### Basic Usage

```python
from ssalib import SingularSpectrumAnalysis
from ssalib.datasets import load_sst

# Load example data: Mean Sea Surface Temperature
ts = load_sst()

# Create SSA instance and decompose
ssa = SingularSpectrumAnalysis(ts)
ssa.decompose()

# Visualize results, in this case, singular values
fig, ax = ssa.plot(kind='values')

# Reconstruct groups
ssa.reconstruct(groups={'trend': [0, 1], 'seasonality': [2, 3]})

# Export
df_ssa = ssa.to_frame()
```

### Available Datasets

| Dataset   | Loading Function   | Description                                                                | Time Range               | Source                                                            | License   |
|-----------|--------------------|----------------------------------------------------------------------------|--------------------------|-------------------------------------------------------------------|-----------|
| Mortality | `load_mortality()` | Daily counts of deaths in Belgium.                                         | 1992-01-01 to 2023-12-31 | [STATBEL](https://statbel.fgov.be/en/open-data/number-deaths-day) | Open Data |  
| SST       | `load_sst()`       | Monthly mean sea surface temperature globally between 60° North and South. | 1982-01-01 to 2023-12-31 | [Climate Reanalyzer](https://climatereanalyzer.org/)              | CC-BY     |

### Available SVD Methods

SSALib supports multiple SVD solvers:

| Solver Name          | Underlying Method                                                                                                                     | Status    |
|----------------------|---------------------------------------------------------------------------------------------------------------------------------------|-----------|
| `numpy_standard`     | [`numpy.linalg.svd`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)                                          | Default   |
| `scipy_standard`     | [`scipy.linalg.svd`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html)                                      | Available |
| `scipy_sparse`       | [`scipy.sparse.linalg.svds`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html)                      | Available |
| `sklearn_randomized` | [`sklearn.utils.extmath.randomized_svd`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html) | Available |

Select the solver with the `svd_solver` argument.

```python
from ssalib import SingularSpectrumAnalysis
from ssalib.datasets import load_sst

# Load example data
ts = load_sst()

# Create SSA instance with solver 'sklearn_randomized'
ssa = SingularSpectrumAnalysis(ts, svd_solver='sklearn_randomized')
```

### Available Visualizations

| `kind`        | Description                                                                   | Decomposition Required | Reconstruction Required |
|---------------|-------------------------------------------------------------------------------|:----------------------:|:-----------------------:|
| `matrix`      | Plot the matrix or its group reconstruction                                   |        Optional        |        Optional         |
| `paired`      | Plot pairs (x,y) of successive left-eigenvectors                              |          Yes           |           No            |
| `periodogram` | Plot periodogram associated with eigenvectors                                 |          Yes           |           No            |
| `timeseries`  | Plot original, preprocessed, or reconstructed time series                     |        Optional        |        Optional         |
| `values`      | Plot the singular values ranked by value norm or dominant component frequency |          Yes           |           No            |
| `vectors`     | Plot the left eigen vectors                                                   |          Yes           |           No            |
| `wcorr`       | Plot the weighted correlation matrix                                          |          Yes           |           No            |

Pass the `kind` argument to the `SingularSpectrumAnalysis.plot` method.

## Documentation

For more in-depth examples and tutorials, check the Jupyter notebooks in the
`notebooks` folder:

- [Tutorial 1: Introduction to SSA](/notebooks/01_basic_ssa_introduction.ipynb)
- [Tutorial 2: Plotting Guide](/notebooks/02_ssa_plotting_guide.ipynb)
- [Tutorial 3: SVD Matrix Construction and Window Sizes](/notebooks/03_ssa_svd_matrices.ipynb)
- [Tutorial 4: Comparison of SVD Solvers and Speed Performances](/notebooks/04_svd_solver_comparison.ipynb)
- [Tutorial 5: Comparison of SSALib and Rssa](/notebooks/05_Rssa_comparison.ipynb)

In more advanced tutorials, we cover:

- [A1: Testing Significance with
  `MonteCarloSSA`](/notebooks/A1_montecarlo_ssa.ipynb)

## References

The main references used to develop SSALib were:

1. Golyandina, N., & Zhigljavsky, A. (2020). Singular Spectrum Analysis for Time
   Series. Berlin, Heidelberg:
   Springer. https://doi.org/10.1007/978-3-662-62436-4
2. Hassani, H. (2007). Singular Spectrum Analysis: Methodology and Comparison.
   Journal of Data Science, 5(2),
   239–257. https://doi.org/10.6339/JDS.2007.05(2).396
3. Broomhead, D. S., & King, G. P. (1986). Extracting qualitative dynamics from
   experimental data. Physica D: Nonlinear Phenomena, 20(2),
   217–236. https://doi.org/10.1016/0167-2789(86)90031-X
4. Vautard, R., & Ghil, M. (1989). Singular spectrum analysis in nonlinear
   dynamics, with applications to paleoclimatic time series. Physica D:
   Nonlinear Phenomena, 35(3). https://doi.org/10.1016/0167-2789(89)90077-8
5. Allen, M. R., & Smith, L. A. (1996). Monte Carlo SSA: Detecting irregular
   oscillations in the Presence of Colored Noise. Journal of Climate, 9(12),
   3373–3404.
   [https://doi.org/10.1175/1520-0442(1996)009<3373:MCSDIO>2.0.CO;2](https://doi.org/10.1175/1520-0442(1996)009<3373:MCSDIO>2.0.CO;2)

## How to Cite

Awaiting the release of the first non-beta version, you can refer to SSALib
using:

```bibtex
@software{ssalib2025,
  author    = {Delforge, Damien AND Alonso, Alice AND de Viron, Olivier AND Vanclooster, Marnik AND Speybroeck, Niko},
  title     = {{SSALib}: A {Python} Library for {Singular Spectrum Analysis}},
  year      = {2025},
  version   = {0.1.2b1},
  url       = {https://github.com/ADSCIAN/ssalib}
}
```

