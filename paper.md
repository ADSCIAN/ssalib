---
title: 'SSALib: a Python Library for Time Series Decomposition using Singular Spectrum Analysis'
tags:
  - Python
  - time series
  - Singular Spectrum Analysis
  - Singular Value Decomposition
  - time series decomposition
authors:
  - name: Damien Delforge
    orcid: 0000-0002-3552-9444
    corresponding: true
    affiliation: "1, 2"
  - name: Alice Alonso
    orcid: 0000-0001-8869-6759
    affiliation: "2, 3"
  - name: Olivier de Viron
    orcid: 0000-0003-3112-9686
    affiliation: "4"
  - name: Marnik Vanclooster
    orcid: 0000-0003-1358-8723
    affiliation: "3"
  - name: Niko Speybroeck
    orcid: 0000-0003-3322-3502
    affiliation: "1"

affiliations:
  - name: Université catholique de Louvain, Institute of Health & Society, Brussels, Belgium.
    index: 1
    ror: "02495e989"
  - name: AD Scientific Consulting & Environmental Systems Analytics (ADSCIAN), Brussels, Belgium.
    index: 2
  - name: Université catholique de Louvain, Earth & Life Institute, Louvain-la-Neuve, Belgium.
    index: 3
    ror: "02495e989"
  - name: Littoral, Environnement et Sociétés, La Rochelle Université and CNRS (UMR7266), La Rochelle, France
    index: 4
    ror: "00r8amq78"

date: 27 October 2025
bibliography: paper.bib
---

# Summary

Singular Spectrum Analysis (SSA) is a method developed in the 1980s for
analyzing and decomposing time series. Using time-delayed
trajectories or covariance matrices, SSA takes advantage of temporal
dependencies to identify structured components such as trends and cycles.
Time series decomposition has various applications, including denoising,
filtering, signal modeling, interpolation (or gap filling), and extrapolation
(or forecasting). The Singular Spectrum Analysis Library (SSALib) is a Python
package that simplifies SSA implementation and visualization for the
decomposition of univariate time series, featuring component significance
testing.

# Statement of Needs

SSA is a non-parametric method that provides a low-assumption framework for
exploring, discovering, and decomposing linear, nonlinear, or pseudo-periodic
patterns in time series data, in contrast to methods that require strong
_a priori_ hypotheses about signal components
[@elsner_singular_1996; @golyandina_singular_2020].
The SSALib package includes Monte Carlo SSA to support statistical inference and
reduce subjective user guidance. Its Python Application Programming Interface is
designed to streamline the SSA workflow and facilitate time series exploration,
including built-in plotting features.

SSALib is particularly relevant for researchers and practitioners working in
domains where time series analysis is central, i.e., climate and environmental
sciences, geophysics, neuroscience, econometrics, or epidemiology.

# Mathematical Background

The mathematical background of Singular Spectrum Analysis (SSA) has been
primarily developed during the 1980–2000 period
[@golyandina_particularities_2020; @elsner_singular_1996; @golyandina_singular_2020].
The basic Singular Spectrum Analysis (SSA) algorithm, as described by
@broomhead_extracting_1986 (BK-SSA) or
@vautard_singular_1989 (VG-SSA), applies to univariate time series. It consists
of three major steps [@hassani_singular_2007; @golyandina_singular_2020]. The
first step is the time-delayed matrix construction. The second step consists in
a Singular Value Decomposition of the trajectory matrix. The BK-SSA approach is
based on a time-delayed trajectory matrix with dimensions depending on the
window parameter and the number of unit lags. This matrix consists of lagged
copies of time series segments of a specified length, forming a Hankel matrix,
i.e., with equal anti-diagonal values. In contrast, the VG-SSA approach captures
time dependencies by constructing a special type of covariance matrix that has a
Toeplitz structure, meaning that its diagonal values are identical. The
eigenvalues of the SVD depend on the variance captured by each component, either
composed of one eigenvector, for monotonic trends, or two eigenvectors,
representing nonlinear trends, pseudo-periodic cycles or oscillations. 
In the third step, the eigenvectors are then grouped, for
pseudo-periodic components, and their contributions to the time series are
reconstructed via projection.

For testing the significance of the retrieved mode, @allen_monte_1996
proposed a Monte Carlo approach, by comparison of the variance captured by the
eigenvectors on the original time series with that captured in many random
autoregressive (AR) surrogate time series [@schreiber_surrogate_2000]. Many
extensions have been proposed for the methods, paving the way for future
developments, such as multivariate (or multichannel) SSA (M-SSA), 
SSA-based interpolation and extrapolation, 
or causality tests [@golyandina_singular_2020].

# Implementation Details

The Singular Spectrum Analysis Library (SSALib) Python package interfaces
time series as `numpy.Array` [@harris_array_2020] or `pandas.Series`
[@mckinney_data_2010] objects. It uses decomposition algorithms from
acknowledged Python scientific packages like `numpy` [@harris_array_2020],
`scipy` [@virtanen_scipy_2020], and `sklearn` [@pedregosa_scikit-learn_2011].
In particular, `sklearn` features a randomized SVD algorithm for efficient
decomposition [@halko_finding_2010]. Visualization features rely on
`matplotlib`, drawing inspiration from the R `rSSA` package
[@golyandina_singular_2018].

SSALib also incorporates the Monte Carlo SSA approach [@allen_monte_1996] for
identifying significant components by comparison to randomly generated data
(i.e., surrogate data), relying on `statsmodels` [@seabold_statsmodels_2010]
for fitting AR processes and generate the surrogate data. In
SSALib, an AR process of a specified maximum order is fitted
relying on a state space modeling framework [@durbin_time_2012], which enables
fitting AR processes from time series that contain masked or missing values.

# Related Work

@golyandina_singular_2020 mention some existing software dedicated to
SSA, such as the GUI-based SSA-MTM toolkit, Caterpillar-SSA software, and the
rSSA R package. In Python, most SSA implementations are basic and part of large
software packages, including `pyts` [@faouzi_pyts_2020], `pyleoclim`
[@khider_pyleoclim_2023], or `pyactigraphy` [@hammad_pyactigraphy_2024], or are
available primarily as unmaintained and untested projects. To address this gap,
SSALib was developed as a fully dedicated and tested Python package for SSA
that is suitable for both teaching and research purposes.

# References
