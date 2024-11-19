# Visually Assisted Singular Spectrum AnaLysis (VASSAL)

[![Tests](https://github.com/ADSCIAN/vassal/actions/workflows/python-tests.yml/badge.svg)](https://github.com/ADSCIAN/vassal/actions/workflows/python-tests.yml)
[![Python Version](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue)](https://www.python.org)
[![Coverage](https://img.shields.io/badge/coverage-88%25-green)](https://github.com/ADSCIAN/vassal/actions)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Development Status](https://img.shields.io/badge/Development%20Status-Alpha-yellow)](https://pypi.org/project/your-package-name/)


> [!NOTE]
> This repository contains an early alpha development version of VASSAL,
> designed for educational purposes. VASSAL has undergone extensive testing
> with the pytest framework; however, users may report any experienced issues
> using [GitHub Issues](https://github.com/ADSCIAN/vassal/issues).

## Overview

VASSAL (Visually Assisted Singular Spectrum AnaLysis) is a Python package
implementing the basic Singular Spectrum Analysis (SSA) univariate timeseries
decomposition technique. It relies on different Singular Value Decomposition
(SVD) methods from existing Python scientific packages and provides a convenient
API along with plotting capabilities.

## Key Features

- Basic SSA implementation with both BK and VG approaches
- Multiple SVD solver options (NumPy, SciPy, scikit-learn)
- Built-in visualization tools for analysis
- Include example datasets
- Comprehensive test coverage
- Type-annotated codebase

## Quick Start

### Requirements

- Python ≥ 3.10
- NumPy
- SciPy
- pandas
- matplotlib
- scikit-learn

### Installation

```bash
pip install git+https://github.com/ADSCIAN/vassal.git
```

### Basic Usage

```python
from vassal import SingularSpectrumAnalysis
from vassal.datasets import load_sst

# Load example data
ts = load_sst()

# Create SSA instance and decompose
ssa = SingularSpectrumAnalysis(ts)
ssa.decompose()

# Visualize components
fig, ax = ssa.plot(kind='values')

# Reconstruct Groups
ssa.reconstruct(groups={'trend': [0, 1], 'seasonality': [2, 3]})

# Export
df_ssa = ssa.to_frame()
```

### Available Datasets

| Dataset   | Loading Function   | Description                                                                | Time Range               | Source                                                            | License   |
|-----------|--------------------|----------------------------------------------------------------------------|--------------------------|-------------------------------------------------------------------|-----------|
| Mortality | `load_mortality()` | Daily counts of deaths in Belgium.                                         | 1992-01-01 to 2023-12-31 | [STATBEL](https://statbel.fgov.be/en/open-data/number-deaths-day) | Open Data |  
| SST       | `load_sst()`       | Monthly mean sea surface temperature globally between 60° North and South. | 1982-01-01 to 2023-12-31 | [Climate Reanalyzer](https://climatereanalyzer.org/)              | CC-BY     |
| Sunspots  | `load_sunspots()`  | Monthly mean total sunspot number.                                         | 1749-01 to 2023-12       | [Royal Observatory of Belgium](https://www.sidc.be/SILSO/)        | CC-BY-NC  |

### Available SVD Methods

VASSAL supports multiple SVD solvers:

| Solver Name          | Underlying Method                          | Status      |
|----------------------|--------------------------------------------|-------------|
| `numpy_standard`     | [`numpy.linalg.svd`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)      | Default     |
| `scipy_standard`     | [`scipy.linalg.svd`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html)      | Available   |
| `scipy_sparse`       | [`scipy.sparse.linalg.svds`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html) | Available   |
| `sklearn_randomized` | [`sklearn.utils.extmath.randomized_svd`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html) | Available   |

Select the solver with the `svd_solver` argument.

```python
from vassal import SingularSpectrumAnalysis
from vassal.datasets import load_sst

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

## References

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
   Nonlinear Phenomena, 35(3)

## How to Cite

TODO

