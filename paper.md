---
title: 'SSALib: a Python Library for Timeseries Decomposition using Singular Spectrum Analysis'
tags:
  - Python
  - time series
  - singular spectrum analysis
  - singular value decomposition
  - time series decomposition
authors:
  - name: Damien Delforge
    orcid: 0000-0002-3552-9444
    corresponding: true
    affiliation: "1, 2"
  - name: Alice Alonso
    orcid: 0000-0001-8869-6759
    affiliation: "1, 3"
  - name: Niko Speybroeck
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
date: 21 December 2024
bibliography: paper.bib
---

# Summary & Statement of Needs

Singular Spectrum Analysis (SSA) is a method developed in the 1980s for
analyzing and decomposing time-series data 
[@broomhead_extracting_1986, @vautard_singular_1989]. Using time-delayed 
trajectories or covariance matrices, SSA takes advantage of temporal 
dependencies to identify structured components such as trends and cycles 
[@elsner_singular_1996, golyandina_singular_2020]. Time-series decomposition 
has various applications, including denoising, filtering, signal modeling, 
interpolation (or gap filling), and extrapolation (or forecasting).

SSA is a non-parametric method that allows for the decomposition and analysis of
time series without prior knowledge of their underlying dynamics. Another
advantage of SSA is the ability to extract nonlinear trends and phase- or
amplitude-modulated cycles. The Singular Spectrum Analysis Library (`ssalib`) 
is a Python package that simplifies SSA implementation and visualization 
through an easy-to-use API, operating time series as `numpy.Array` 
[@harris_array_2020] or `pandas.Series` [mckinney_data_2010] objects, and 
requiring minimal knowledge of linear algebra. It uses decomposition algorithms 
from robust Python scientific packages like `numpy` [@harris_array_2020], 
`scipy` [@virtanen_scipy_2020], and `sklearn` [@pedregosa_scikit-learn_2011]. 
SSALib also incorporates the Monte Carlo SSA approach [@allen_monte_1996] for 
identifying significant components by comparison to randomly generated data 
(i.e., surogate data), relying on `statsmodels` [@seabold_statsmodels_2010] for 
fitting autoregressive processes and generate the surrogate data.

The basic Singular Spectrum Analysis (SSA) algorithm for univariate time series,
as described by @broomhead_extracting_1986 and @vautard_singular_1989, requires
a linear algebra library and only a few lines of code for implementation.
However, developing dedicated SSA software offers several advantages. As time
series analysis has become increasingly common across various fields, SSA
software needs to address the needs of a broader audience focused on practical
applications rather than just technical implementation. In addition, SSA has
evolved beyond being a single method and has transformed into a modular
analytical framework, consisting of interchangeable steps that can be combined
into multiple variants. Consequently, both experts and newcomers would benefit
from SSA software that allows for configurable analyses, saving time in the
process. Moreover, SSA empirical nature relies heavily on data visualization.
This makes the implementation of software essential for providing users with
established visualization features.

@golyandina_singular_2020 mention some existing software dedicated to
SSA, such as the GUI-based SSA-MTM toolkit, Caterpillar-SSA software, and the 
rSSA R package. In Python, most SSA implementations are basic and part of large
software packages, including `pyts` [@faouzi_pyts_2020], `pyleoclim` 
[@khider_pyleoclim_2023], or `pyactigraphy` [@hammad_pyactigraphy_2024], or are 
available primarily as unmaintained and untested projects. To address this gap, 
`ssalib` was developed as a fully dedicated SSA Python package 
that is both tested and suitable for teaching and research purposes.

# Technical Details

The Singular Spectrum Analysis (SSA) approach consists of three major steps 
[@golyandina_singular_2020]: (1) Time-Delayed Matrix Construction, (2)
Matrix Decomposition, and (3) Components Grouping and Reconstruction.
The `ssalib` implements the approaches of @broomhead_extracting_1986, referred 
to as BK, and @vautard_singular_1989, referred to as VG. These variants differ 
in the matrices they utilize during the first step. The BK approach is based on 
a time-delayed trajectory matrix with dimensions depending on the window 
parameter and the number of unit lags. This matrix consists of lagged copies of 
time series segments of a specified length, forming a Hankel matrix where the
anti-diagonal values are equal. In contrast, the VG approach captures time
dependencies by constructing a special type of covariance matrix that has a
Toeplitz structure, meaning that its diagonal values are identical.

Regarding Step 2, `ssalib` relies on Singular Value Decomposition (SVD) with
methods implemented in the NumPy, SciPy, and Scikit-learn libraries. In
particular, scikit-learn features a randomized SVD algorithm for efficient
decomposition [halko_finding_2010]. Step 3 involves visualizations created with
Matplotlib, drawing inspiration from the R rSSA package 
[@golyandina_singular_2018].

Significance testing is based on the work of Allen and Smith (1996). In 
`ssalib`, an autoregressive (AR) process of a specified maximum order is fitted 
relying on a state space modeling framework [@durbin_time_2012] and utilizing 
the `statsmodels` library [@seabold_statsmodels_2010]. The AR random surrogates 
are also generated using statsmodels, and their time-delayed matrices are 
projected onto the singular system of the original time series. This comparison 
of the original distribution of singular values with the many random 
projections allows for the inference of significance.