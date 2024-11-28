"""Plotting Tools for Singular Spectrum Analysis"""
from __future__ import annotations

import abc
import logging
from enum import Enum
from functools import cached_property
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from scipy.signal import periodogram
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

from ssalib.error import DecompositionError

logger = logging.getLogger(__name__)


class SSAPlotType(Enum):
    """Available plot types for SingularSpectrumAnalysis
    """
    MATRIX = 'matrix'
    PAIRED = 'paired'
    PERIODOGRAM = 'periodogram'
    TIMESERIES = 'timeseries'
    VALUES = 'values'
    VECTORS = 'vectors'
    WCORR = 'wcorr'

    @property
    def requires_decomposition(self):
        """Whether this plot type requires decomposition"""
        return self in {
            self.PAIRED,
            self.PERIODOGRAM,
            self.VALUES,
            self.VECTORS,
            self.WCORR
        }

    @property
    def supports_ax(self) -> bool:
        """Whether this plot type supports the ax argument."""
        return self in {
            self.MATRIX,
            self.TIMESERIES,
            self.VALUES,
            self.WCORR
        }

    @classmethod
    def available_plots(cls) -> list[str]:
        return [solver.value for solver in cls]


class PlotSSA(metaclass=abc.ABCMeta):
    """Abstract Base Class for Singular Spectrum Analysis Plotting.

    This class defines the interface for creating plots in the context of
    Singular Spectrum Analysis (SSA). Any subclass of PlotSSA must implement
    the `plot` method.
    """
    # Child-defined arguments
    na_mask: NDArray[bool]
    n_components: int | None = None  # number of components
    eigenvalues: NDArray[float] | None = None  # array of eigenvalues
    squared_frobenius_norm: float | None = None  # sum of eigenvalues
    svd_matrix: NDArray[float]  # SVD matrix
    _ix: pd.Index | None  # time series index
    _n: int | None = None  # timeseries length
    _na_strategy: str  # missing data strategy
    _window: int | None = None  # SSA window length
    u_: NDArray[float] | None = None  # left eigenvectors
    s_: NDArray[float] | None = None  # array of singular values
    vt_: NDArray[float] | None = None  # right eigenvectors
    _svd_matrix_kind: 'SSAMatrixType'  # SVD matrix kind
    _timeseries_pp: NDArray  # preprocessed timeseries

    autoregressive_model: SARIMAXResults  # Surrogate model (MonteCarloSSA)
    n_surrogates: int  # Number of surrogates (MontecarloSSA)

    _N_COMPONENTS_SENTINEL = object()  # track if n_components was user-defined
    _N_COMPONENTS_DEFAULT: int = 10  # default number of components to plot

    available_plots: list[str] = SSAPlotType.available_plots
    _PLOT_METHOD_MAPPING = {
        SSAPlotType.MATRIX: '_plot_matrix',
        SSAPlotType.PAIRED: '_plot_paired_vectors',
        SSAPlotType.PERIODOGRAM: '_plot_periodogram',
        SSAPlotType.TIMESERIES: '_plot_timeseries',
        SSAPlotType.VALUES: '_plot_values',
        SSAPlotType.VECTORS: '_plot_vectors',
        SSAPlotType.WCORR: '_plot_wcorr'
    }

    def __validate_main_plot_inputs(
            self,
            kind: SSAPlotType | str,
            n_components: int | None | object,  # object _N_COMPONENTS_SENTINEL
            ax: Axes | None
    ) -> None:
        """Validate input parameters for plotting.
        """
        # Validate plot kind
        if isinstance(kind, str):
            try:
                kind = SSAPlotType(kind)
            except ValueError:
                valid_kinds = [plot_type.value for plot_type in SSAPlotType]
                raise ValueError(
                    f"Unknown plot kind '{kind}'. "
                    f"Valid plot kinds are: {', '.join(valid_kinds)}"
                )

        # Validate decomposition requirement
        if kind.requires_decomposition and self.n_components is None:
            raise DecompositionError(
                f"Decomposition must be performed before calling the 'plot' "
                f"method with kind='{kind.value}'. "
                f"Call 'decompose' method first"
            )

        # Validate n_components parameter
        if kind.requires_decomposition:
            if (not isinstance(n_components, (int, type(None))) and
                    n_components is not self._N_COMPONENTS_SENTINEL):
                raise TypeError(
                    f"Argument 'n_components' must be integer or None, "
                    f"got {type(n_components)}"
                )
            if (isinstance(n_components, int) and
                    n_components > self.n_components):
                raise ValueError(
                    f"Argument 'n_components' must be less than or equal to the "
                    f"number of components ({self.n_components}), "
                    f"got {n_components}"
                )
        elif n_components is not self._N_COMPONENTS_SENTINEL:
            logger.warning(
                f"Parameter 'n_components' is not supported for plot kind "
                f"'{kind.value}' and will be ignored"
            )

        # Validate axes parameter
        if not kind.supports_ax:
            if ax is not None:
                logger.warning(
                    f"Parameter 'ax' is not supported for plot kind "
                    f"'{kind.value}' and will be ignored"
                )
        elif ax is not None and not isinstance(ax, Axes):
            raise TypeError(
                f"Parameter 'ax' must be an instance of matplotlib.axes.Axes, "
                f"got {type(ax)}"
            )

    def plot(
            self,
            kind: SSAPlotType | str = SSAPlotType.VALUES,
            n_components: int | None = _N_COMPONENTS_SENTINEL,
            ax: Axes = None,
            **plot_kwargs: Any
    ) -> tuple[Figure, Axes | NDArray[Axes]]:
        """Main method of the plotting API of SingularSpectrumAnalysis

        The `plot` method generates plots of various kinds to explore the
        eigentriple features and reconstructed time series from
        `SingularSpectrumAnalysis` instance.

        Parameters
        ----------
        kind : SSAPlotType | str, default SSAPlotType.VALUES
            The type of plot to produce, user options include:
            * 'matrix': Plots the decomposed or reconstructed matrix.
            * 'paired': Plots pairs of successive left eigenvectors against
              each other.
            * 'periodogram': Plots power spectral density of the original
              series or the one associated with each eigenvector.
            * 'timeseries': Displays reconstructed time series based on
              component groups defined with the
              'SingularSpectrumAnalysis.reconstruct' method.
            * 'values': Plots singular values to inspect their magnitudes.
            * 'vectors': Plots the left eigenvectors.
            * 'wcorr': Displays a weighted correlation matrix using a heatmap.
        n_components : int | None, default 10
            Number of eigentriple components to use in the plot. Only valid for
            kind 'paired', 'periodogram', 'values', 'vectors', and
            'wcorr'. If None, the maximum number of components is used.
        ax : Axes, optional
            An existing matplotlib Axes object to draw the plot on. If None, a
            new figure and axes are created. This parameter is ignored for
            subplots, i.e., kind 'paired', 'vectors', and 'periodogram'.

        Other Parameters
        ----------------
        'indices' : int | range | list[int], optional
            For kind 'matrix' only. If passed, plot the reconsconstructed
            matrix for the component index (int), a range of component indices
            (range), or a list of component indices (list[int]). If None, plot
            the original matrix being decomposed.
        'scale' : Literal['loglog', 'plot', 'semilogx', 'semilogy'], optional
            For kind 'periodogram' only. Scale used for each periodogram
            subplot.
        'include' : list[str] | None, optional, default None
            For kind 'timeseries' only. List of time series names to include
            in the time series plot. Passed to the
            SingularSpectrumAnalysis.to_frame method. If None and 'exclude' is
            None, all time series are included.
        'exclude' : list[str] | None, optional, default None
            For kind 'timeseries' only. List of time series names to exclude
            in the time series plot. Passed to the
            SingularSpectrumAnalysis.to_frame method. If None and 'include' is
            None, all time series are included.
        'rescale' : bool, optional, default False
            For kind 'timeseries' only. Whether to rescale the series with
            the original standard deviation and mean.
        'rank_by' : Literal['values', 'freq'], optional, default 'values'.
            For kind 'values' only. Whether to sort the singular values
            by decreasing values or by increasing frequencies.
        'confidence_level' : float | None, optional, default 0.95
            For kind 'values' only and class 'MonteCarloSSA'. Defines the
            confidence level for defining the percentile interval of the
            surrogate value distribution.
        'two_tailed' : bool, optional, default True
            For kind 'values' only and class 'MonteCarloSSA'. Controls how
            surrogate value distribution percentile intervals are calculated.
            - If True, uses two-tailed confidence intervals with
              (1-confidence_level) / 2 on each tail. Example: For 95%
              confidence (confidence_level = 0.95), uses 2.5th and 97.5th
              percentiles.
            - If False, uses one-tailed confidence intervals with
              intervals are defined between 0 and
              100 * (1 - confidence_level) / 2. Example: For 95% confidence,
              (confidence_level = 0.95), uses the minimum value and 95th
              percentile.
        'errorbar_kwargs' : dict[str, Any], optional, default None
            For kind 'values' only and class 'MonteCarloSSA'. Dictionary of
            keyword arguments to pass to the matplotlib errorbar function to
            control the display of surrogate confidence intervals.
        plot_kwargs : Any, optional
            Additional keyword arguments for customization of the plot, passed
            to the respective main plotting function. The specific function used
            depends on the 'kind' of plot. See corresponding documentation for
            details.
            - 'matrix': `matplotlib.pyplot.imshow`
            - 'paired', 'values', 'vectors': `matplotlib.pyplot.plot`
            - 'periodogram': `matplotlib.pyplot.semilogy`
            - 'timeseries': `pandas.DataFrame.plot`
            - 'wcorr': `matplotlib.pyplot.imshow`

        Returns
        -------
        tuple[Figure, Axes]
            A tuple containing the matplotlib Figure and Axes objects with the
            generated plot. This allows further customization after the
            function returns.

        Raises
        ------
        DecompositionError
            If the ´plot´ method is called before decomposition for plot 'kind'
            that does not allow it.

        Examples
        --------
        .. plot::
            :include-source: True

            >>> from ssalib.ssa import SingularSpectrumAnalysis
            >>> from ssalib.datasets import load_sst
            >>> sst = load_sst()
            >>> ssa = SingularSpectrumAnalysis(sst)
            >>> _ = ssa.decompose()
            >>> ssa.available_plots()
            ['matrix', 'paired', 'periodogram', 'timeseries', 'values', 'vectors', 'wcorr']

            >>> ssa.plot(kind='values', n_components=30, marker='.', ls='--') # doctest: +SKIP

        """

        self.__validate_main_plot_inputs(
            kind=kind,
            n_components=n_components,
            ax=ax
        )

        kind = SSAPlotType(kind)

        # Adjust _N_COMPONENTS_DEFAULT based on self.n_components
        if (self.n_components is not None and
                self.n_components < self._N_COMPONENTS_DEFAULT):
            self._N_COMPONENTS_DEFAULT = self.n_components

        if n_components is None and kind.requires_decomposition:
            n_components = self.n_components

        if (n_components is self._N_COMPONENTS_SENTINEL and
                kind.requires_decomposition):
            plot_kwargs['n_components'] = self._N_COMPONENTS_DEFAULT
        if (n_components is not self._N_COMPONENTS_SENTINEL and
                kind.requires_decomposition):
            plot_kwargs['n_components'] = n_components

        if ax is not None and kind.supports_ax:
            plot_kwargs['ax'] = ax

        plot_method = getattr(self, self._PLOT_METHOD_MAPPING[kind])
        fig, ax = plot_method(**plot_kwargs)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        return fig, ax

    @abc.abstractmethod
    def get_confidence_interval(
            self,
            n_components: int | None = None,
            confidence_level: float = 0.95,
            two_tailed: bool = True,
            return_lower: bool = True,
    ) -> tuple[NDArray[float], NDArray[float]] | NDArray[float]:
        pass

    @abc.abstractmethod
    def to_frame(
            self,
            include: list[str] | None = None,
            exclude: list[str] | None = None,
            rescale: bool = False
    ) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def _reconstruct_group_matrix(
            self,
            group_indices: int | slice | range | list[int]
    ) -> NDArray[float]:
        pass

    @abc.abstractmethod
    def _reconstruct_group_timeseries(
            self,
            group_indices: int | slice | range | list[int]
    ) -> NDArray[float]:
        pass

    @abc.abstractmethod
    def wcorr(self, n_components: int) -> NDArray:
        pass

    def _plot_matrix(
            self,
            indices: int | range | list[int] = None,
            ax: Axes | None = None,
            **plot_kwargs: Any
    ):
        """Plot decomposed or reconstructed matrices.
        """
        if not ax:
            fig = plt.figure()
            ax = fig.gca()
        else:
            fig = ax.get_figure()

        if indices is None:  # plot decomposed
            matrix = self.svd_matrix
            subtitle = f'({self._svd_matrix_kind.value}, Original)'
        else:  # plot reconstructed
            if self.n_components is None:
                raise DecompositionError(
                    "Cannot plot reconstructed matrix prior to decomposition. "
                    "Make sure to call the 'decompose' method first."
                )
            matrix = self._reconstruct_group_matrix(group_indices=indices)
            subtitle = f'({self._svd_matrix_kind.value}, Group:{indices})'

        im = ax.imshow(matrix, **plot_kwargs)
        ax.set_aspect('equal')
        ax.set_title(f'SVD Matrix {subtitle}')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        fig.colorbar(im)

        return fig, ax

    def _plot_paired_vectors(
            self,
            n_components: int,
            **plot_kwargs: Any
    ) -> tuple[Figure, Axes]:
        """Plot successive paired left eigenvectors.
        """
        pairs = list(zip(range(0, n_components - 1), range(1, n_components)))
        u = self.u_
        eigenvalues = self.eigenvalues
        squared_frobenius_norm = self.squared_frobenius_norm

        rows, cols = self._auto_subplot_layout(len(pairs))

        fig, axes = plt.subplots(rows, cols, figsize=(1.7 * cols, 1.6 * rows))

        for i in range(rows * cols):
            ax = axes.ravel()[i] if isinstance(axes, np.ndarray) else axes
            try:
                j, k = pairs[i]
                ax.plot(u[:, k], u[:, j], **plot_kwargs)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect('auto', 'box')
                ax.axis('off')
                contribution_1 = eigenvalues[j] / squared_frobenius_norm * 100
                contribution_2 = eigenvalues[k] / squared_frobenius_norm * 100

                title = (f'EV{j} ({contribution_1:.1f}%) vs.'
                         f' {j + 1} ({contribution_2:.1f}%)')

                ax.set_title(title, {'fontsize': 'small'})
            except IndexError:
                ax.axis('off')

        return fig, axes

    def _plot_periodogram(
            self,
            n_components: int | None,
            scale: Literal['loglog', 'plot', 'semilogx', 'semilogy'] = 'loglog',
            **plot_kwargs
    ) -> tuple[Figure, Axes] | tuple[Figure, list[Axes]]:
        """Plot the power spectral density of signals associated with
        eigenvectors.
        """
        if not isinstance(scale, str):
            raise TypeError(
                f"Parameter scale must be a string, got {type(scale)}"
            )
        if scale not in ['loglog', 'plot', 'semilogx', 'semilogy']:
            raise ValueError(
                f"Parameter scale must be one of 'loglog', 'plot', "
                f"'semilogx', or 'semilogy', got '{scale}'"
            )

        freq_original, psd_original = self.periodogram

        unit = None
        if self._ix is not None:
            if isinstance(self._ix, pd.DatetimeIndex):
                unit = pd.infer_freq(self._ix)  # Automated unit inference
        if unit is None:
            unit = ''

        rows, cols = self._auto_subplot_layout(n_components)

        fig, axes = plt.subplots(rows, cols, figsize=(1.5 * cols, 1.5 * rows))

        for i in range(rows * cols):
            ax = axes.ravel()[i]
            ax.axis('off')
            plot_method = getattr(ax, scale)

            if i >= n_components:
                continue
            plot_method(freq_original[1:], psd_original[1:], lw=.5,
                        color='lightgrey')
            freq, psd = periodogram(self._reconstruct_group_timeseries([i]))
            plot_method(freq[1:], psd[1:], **plot_kwargs)
            dominant_freq = freq[np.argmax(psd)]
            period = 1 / dominant_freq
            title = f'EV{i} (T={period:.3g}{unit})'
            ax.set_title(title, {'fontsize': 'small'})
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('auto', 'box')

        return fig, fig.get_axes()

    def _plot_timeseries(
            self,
            ax: Axes | None = None,
            include: list[str] | None = None,
            exclude: list[str] | None = None,
            rescale: bool = False,
            **plot_kwargs
    ) -> tuple[Figure, Axes]:
        """Plot all or selected timeseries as a single or a subplot
        """
        data = self.to_frame(include, exclude, rescale)

        axes = data.plot(ax=ax, **plot_kwargs)

        if isinstance(axes, np.ndarray):
            fig = axes[0].get_figure()
            for ax in axes:
                ax.legend(loc='upper left')
        else:
            fig = axes.get_figure()
            axes.legend(loc='best')

        return fig, axes

    def _plot_values(
            self,
            n_components: int,
            rank_by: Literal['values', 'freq'] = 'values',
            ax: Axes | None = None,
            confidence_level: float | None = None,
            two_tailed: bool = True,
            errorbar_kwargs: dict[str, Any] | None = None,
            **plot_kwargs
    ) -> tuple[Figure, Axes]:
        """Plot component norms by decreasing values or increasing frequencies.
        """
        s = self.s_

        if rank_by == 'freq':
            dominant_frequencies = self.get_dominant_frequencies(n_components)
            order = np.argsort(dominant_frequencies)
            s = s[order][:n_components]
            x_values = dominant_frequencies[order]
            x_label = 'Dominant Frequency (Cycle/Unit)'
        else:
            order = np.arange(n_components)
            x_values = order
            s = s[:n_components]
            x_label = 'Component Index'

        if not ax:
            fig = plt.figure()
            ax = fig.gca()

        ax.semilogy(x_values, s, **plot_kwargs)
        ax.set_ylabel('Component Norm')
        ax.set_xlabel(x_label)
        if rank_by == 'values':
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        if not hasattr(self, 'n_surrogates'):  # Basic SingularSpectrumAnalysis
            if confidence_level is not None:
                raise ValueError(
                    "Parameter 'confidence_level' is only valid for "
                    "class 'MonteCarloSSA'."
                )
            if errorbar_kwargs is not None:
                raise ValueError(
                    "Parameter 'errorbar_kwargs' is only valid for "
                    "class 'MonteCarloSSA'."
                )

        else:  # MonteCarloSSA
            if confidence_level is None:
                confidence_level: float = 0.95
            lower, upper = self.get_confidence_interval(
                n_components,
                confidence_level,
                two_tailed,
            )

            center = (lower + upper) / 2
            errorbar_length = (upper - lower) / 2

            if errorbar_kwargs is None:
                errorbar_kwargs = {}

            ax.errorbar(
                x_values,
                center[order],
                yerr=errorbar_length[order],
                fmt='none',
                capsize=1,
                elinewidth=.5,
                capthick=.5,
                color='k',
                label=f'AR{len(self.autoregressive_model.arparams)} Surrogate '
                      f'{confidence_level * 100:g}% CI'
                      f'\nn={self.n_surrogates}',
                **errorbar_kwargs
            )
            ax.legend()

        fig = ax.get_figure()

        return fig, ax

    def _plot_vectors(
            self,
            n_components: int,
            **plot_kwargs
    ):
        """Plot left eigenvectors.
        """
        u = self.u_
        squared_frobenius_norm = self.squared_frobenius_norm

        rows, cols = self._auto_subplot_layout(n_components)

        fig, axes = plt.subplots(rows, cols, figsize=(1.5 * cols, 1.5 * rows))

        for i in range(rows * cols):
            ax = axes.ravel()[i]
            try:
                ax.plot(u[:, i], **plot_kwargs)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect('auto', 'box')
                ax.axis('off')
                contribution = self.eigenvalues[
                                   i] / squared_frobenius_norm * 100

                title = f'EV{i} ({contribution:.1f}%)'
                ax.set_title(title, {'fontsize': 'small'})
            except IndexError:
                ax.axis('off')

        return fig, fig.get_axes()

    def _plot_wcorr(
            self,
            n_components: int,
            ax: Axes | None = None,
            **plot_kwargs
    ):
        """Plot the weighted correlation matrix.
        """
        wcorr = self.wcorr(n_components)

        if not ax:
            fig = plt.figure()
            ax = fig.gca()
        else:
            fig = ax.get_figure()

        if 'vmin' not in plot_kwargs:
            plot_kwargs['vmin'] = -1
        if 'vmax' not in plot_kwargs:
            plot_kwargs['vmax'] = 1
        if 'cmap' not in plot_kwargs:
            plot_kwargs['cmap'] = 'PiYG'

        im = ax.pcolor(wcorr, **plot_kwargs)
        ax.set_aspect('equal')

        # set ticks
        ticks = np.arange(wcorr.shape[0])
        ax.set_xticks(ticks + 0.5, minor=False)
        ax.set_yticks(ticks + 0.5, minor=False)
        ax.set_xticklabels(ticks.astype(int), fontsize='x-small')
        ax.set_yticklabels(ticks.astype(int), fontsize='x-small')

        ax.set_title('W-Correlation Matrix')

        fig.colorbar(im)

        return fig, ax

    @staticmethod
    def _auto_subplot_layout(n_plots: int) -> tuple[int, int]:
        """Calculate the optimal layout for a given number of subplots

        The method favors a 3-column layout until nine plots and then favors a
        squared layout with possibly more columns than rows.

        Parameters
        ----------
        n_plots : int
            The number of subplots required.

        Returns
        -------
        tuple[int, int]
            A tuple containing the number of rows and columns for the layout.

        """
        if n_plots <= 9:
            rows = np.ceil(n_plots / 3).astype(int)
            cols = 3 if n_plots > 3 else n_plots
        else:
            cols = np.ceil(np.sqrt(n_plots)).astype(int)
            rows = np.ceil(n_plots / cols).astype(int)

        return rows, cols

    @cached_property
    def periodogram(self) -> tuple[NDArray[float], NDArray[float]]:
        """Return frequency and power spectral density of the original series.

        Returns
        -------
        tuple[NDArray[float], NDArray[float]]
            Tuple with the array of frequencies and the associated power
            spectral density.

        Notes
        -----

        - If the standardize argument was passed, the periodogram is calculated
          on the z-standardized series.
        - Uses the scipy periodogram function [1]_.

        References
        ----------
        .. [1] "scipy.signal.periodogram", SciPy documentation,
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lombscargle.html
        """
        if any(self.na_mask):
            logging.warning(
                f"Periodogram is estimated on a series imputed with strategy "
                f"'{self._na_strategy}'"
            )

        freq_original, psd_original = periodogram(self._timeseries_pp)

        return freq_original, psd_original

    def get_dominant_frequencies(
            self,
            n_components: int | None = None
    ) -> np.ndarray:
        """Return the dominant frequency associated with n eigenvector.
        """

        if self.n_components is None:
            raise DecompositionError(
                f"Cannot access 'dominant_frequencies' prior to decomposition. "
                f"Make sure to call the decompose' and 'reconstruct' method "
                f"first."
            )

        if n_components is None:
            n_components = self.n_components

        dominant_freqs = []
        for i in range(n_components):
            freq, psd = periodogram(self._reconstruct_group_timeseries([i]))
            dominant_freqs.append(freq[np.argmax(psd)])

        return np.array(dominant_freqs)


if __name__ == '__main__':
    from doctest import testmod

    testmod()
