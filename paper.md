---
title: 'SSALib: a Python Library for Timeseries Decomposition using Singular Spectrum Analysis'
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
    affiliation: "1, 3"
  - name: Olivier de Viron
    orcid: 0000-0003-3112-9686
    affiliation: "4"
  - name: Marnik Vanclooster
    orcid: 0000-0003-1358-8723
    affiliation: "3"
  - name: Niko Speybroeck
    orcid: 0000-0003-3322-3502
    affiliation: "2"

affiliations:
  - name: AD Scientific Consulting & Environmental Systems Analytics (ADSCIAN), Brussels, Belgium.
    index: 1
  - name: University of Louvain (UCLouvain), Institute of Health & Society, Brussels, Belgium.
    index: 2
    ror: 02495e989
  - name: University of Louvain (UCLouvain), Earth & Life Institute, Louvain-la-Neuve, Belgium.
    index: 3
    ror: 02495e989
  - name: Littoral, Environnement et Sociétés, La Rochelle Université and CNRS (UMR7266), La Rochelle, France
    index: 4
    ror: 00r8amq78
      
date: 23 April 2025
bibliography: paper.bib
---

# Summary

Singular Spectrum Analysis (SSA) is a method developed in the 1980s for
analyzing and decomposing time-series data. Using time-delayed
trajectories or covariance matrices, SSA takes advantage of temporal
dependencies to identify structured components such as trends and cycles. 
Time-series decomposition has various applications, including denoising, 
filtering, signal modeling, interpolation (or gap filling), and extrapolation 
(or forecasting). The Singular Spectrum Analysis Library (SSALib) is a Python 
package that simplifies SSA implementation and visualization for the 
decomposition of univariate time series, featuring component significance 
testing.  

# Statement of Needs

SSA is a non-parametric method that allows for the analysis and decomposition of
time series into nonlinear trends and pseudo-periodic signatures, without prior
knowledge of their underlying dynamics 
[@elsner_singular_1996; @golyandina_singular_2020]. The basic Singular Spectrum 
Analysis (SSA) algorithm for univariate time series, as described by 
@broomhead_extracting_1986 and @vautard_singular_1989, applies to univariate 
time series. It consists of three major steps [@golyandina_singular_2020].The 
first step is the Time-Delayed Matrix Construction. The second step consists in 
a Singular Value Decomposition of the trajectory matrix.The BK-SSA approach is 
based on a time-delayed trajectory matrix with dimensions depending on the 
window parameter and the number of unit lags. This matrix consists of lagged 
copies of time series segments of a specified length, forming a Hankel matrix 
where the anti-diagonal values are equal. In contrast, the VG-SSA approach 
captures time dependencies by constructing a special type of covariance matrix 
that has a Toeplitz structure, meaning that its diagonal values are identical. 
The eigenvalues of the SVD depends on the variance captured by each mode, 
either composed of one (trend) or two (trend or pseudo-periodic cycles) 
eigenvectors (or components). In the third step, the eigenvectors are 
then grouped, for pseudo-periodic components, and their contributions to the 
time series are reconstructed by projection. 

For testing the significance of the retrieved mode, @allen_monte_1996 
proposed a Monte-Carlo approach, by comparison of the variance captured by the 
eigenvector on the original time series with that captured in many random 
autoregressive surrogate time series. Many extensions have been proposed for 
the methods, paving the way for future developments, such as multi-time 
series method (M-SSA), SSA-based interpolation and extrapolation, or causality 
tests.

# Technical Details

The Singular Spectrum Analysis Library (SSALib) Python package interfaces 
time series as `numpy.Array` [@harris_array_2020] or `pandas.Series` 
[@mckinney_data_2010] objects. It uses decomposition algorithms from 
acknowledged Python scientific packages like `numpy` [@harris_array_2020], 
`scipy` [@virtanen_scipy_2020], and `sklearn` [@pedregosa_scikit-learn_2011]. 
In particular, scikit-learn features a randomized SVD algorithm for efficient 
decomposition [@halko_finding_2010]. Visualization features relies on 
Matplotlib, drawing inspiration from the R rSSA package 
[@golyandina_singular_2018].

SSALib also incorporates the Monte Carlo SSA approach [@allen_monte_1996] for 
identifying significant components by comparison to randomly generated data 
(i.e., surrogate data), relying on `statsmodels` [@seabold_statsmodels_2010] 
for fitting autoregressive (AR) processes and generate the surrogate data. In
SSALib, an autoregressive (AR) process of a specified maximum order is fitted
relying on a state space modeling framework [@durbin_time_2012], which enable 
fitting AR processes from time series that contains missing values.

# Related Works

@golyandina_singular_2020 mention some existing software dedicated to
SSA, such as the GUI-based SSA-MTM toolkit, Caterpillar-SSA software, and the
rSSA R package. In Python, most SSA implementations are basic and part of large
software packages, including `pyts` [@faouzi_pyts_2020], `pyleoclim`
[@khider_pyleoclim_2023], or `pyactigraphy` [@hammad_pyactigraphy_2024], or are
available primarily as unmaintained and untested projects. To address this gap,
SSALib was developed as a fully dedicated SSA Python package
that is both tested and suitable for teaching and research purposes.

# References
