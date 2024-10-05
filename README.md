# Visually Assisted Singular Spectrum Analysis Library (VASSAL)

> [!NOTE]
> You are currently viewing at an early development version of VASSAL, designed
> for educational purposes. VASSAL has undergone extensive testing with the 
> pytest framework; however, it may still exhibit unexpected behavior. If you 
> encounter any issues, please report them using GitHub Issues.

## Overview

The `vassal` Python package implements the basic Singular Spectrum
Analysis (SSA) univariate timeseries decomposition technique, relying on 
different Singular Value Decomposition (SVD) methods form existing Python 
scientific packages, and provides a convenient API along with plotting 
capabilities.

## What is SSA?

SSA is a great time series decomposition technique typically used to explore
a signal and separate its deterministic components (e.g., trend or seasonality)
from noise. Learn more in the `vassal` project documentation. SSA is a 
three-step process:
1. Matrix Construction (Time-delayed Embedding);
2. Singular Value Decomposition (SVD);
3. Grouping & Signal Reconstruction;

See more in the [SSA Introduction Tutorial]().

## Installation

```bash
pip install vassal
```

## Features
  
The `vassal` package implements the `SingularSpectrumAnalysis` class, with the
following features implemented.

### Embedding Methods 

The `vassal` package supports univariate time-series decomposition relying on the
SVD of a lagged trajectory matrix (Broomhead & King, 1986). The lagged 
Trajectory matrix is built using a window size parameter window:

```python
from vassal import SingularSpectrumAnalysis
from vassal.datasets import load_sst

sst = load_sst()  # Sea Surface Temperature
ssa = SingularSpectrumAnalysis(sst, window=100)
ssa.svd_matrix
```
  
### Singular Value Decomposition (SVD) methods

By default, `vassal` depends on the NumPy implementation of SVD, yet, provides
alternative algorithms, including truncated SVD algorithms for speed performance.

* [`numpy.linalg.svd`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)  
* [`scipy.linalg.svd`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html)  
* [`scipy.sparse.linalg.svds`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html)
* [`sklearn.utils.extmath.randomized_svd`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html)
* [`dask.array.linalg.svd`](https://docs.dask.org/en/stable/generated/dask.array.linalg.svd.html) (optional)
* [`dask.array.linalg.svd_compressed`](https://docs.dask.org/en/latest/generated/dask.array.linalg.svd_compressed.html) (optional)

### Visualization

The `SingularSpectrumAnalysis` class has a `plot` method allowing for multiple
plot `kind`.

| `kind`       | Description                                               | Decomposition Required | Reconstruction Required |
|--------------|-----------------------------------------------------------|:----------------------:|:-----------------------:|
| `paired`     | Plot pairs (x,y) of successive left-eigenvectors          |          Yes           |           No            |
| `timeseries` | Plot original, preprocessed, or reconstructed time series |        Optional        |        Optional         |
| `values`     | Plot the singular values                                  |          Yes           |           No            |
| `vectors`    | Plot the left eigen vectors                               |          Yes           |           No            |
| `wcorr`      | Plot the weighted correlation matrix                      |          Yes           |           No            |

