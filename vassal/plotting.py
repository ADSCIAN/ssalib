"""Plotting Tools for Singular Spectrum Analysis
"""
from __future__ import annotations

import abc
import logging
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from scipy.signal import periodogram

from vassal.log_and_error import DecompositionError, ignored_argument_warning


class PlotSSA(metaclass=abc.ABCMeta):
    """Plotting base class for SingularSpectrumAnalysis

    PlotSSA is an abstract base class that defines the plotting interface for
    the SingularSpectrumAnalysis class. Any plot can be plotted using the
    ´plot´ method.

    """
    # Inherited arguments:
    # --------------------
    # n_components: int | None = None  # number of components
    # eigenvalues: np.ndarray | None = None  # array of eigenvalues
    # squared_frobenius_norm: float | None = None  # sum of eigenvalues
    # svd_matrix: np.ndarray  # SVD matrix
    # wcorr: np.ndarray  # weighted correlation matrix
    # _n: int | None = None  # timeseries length
    # _w: int | None = None  # SSA window length
    # u_: np.ndarray | None = None  # left eigenvectors
    # s_: np.ndarray | None = None  # array of singular values
    # vt_: np.ndarray | None = None  # right eigenvectors
    # _svd_matrix_kind: str  # SVD matrix kind, either BK or VG

    _plot_kinds_map = {
        'matrix': '_plot_matrix',
        'paired': '_plot_paired_vectors',
        'periodogram': '_plot_periodogram',
        'timeseries': '_plot_timeseries',
        'values': '_plot_values',
        'vectors': '_plot_vectors',
        'wcorr': '_plot_wcorr'
    }

    _n_components_required = ['paired', 'periodogram', 'values', 'vectors',
                              'wcorr']

    def plot(
            self,
            kind: str = 'values',
            n_components: int | None = 10,
            ax: Axes = None,
            scale: Literal['loglog', 'semilogx', 'semilogy'] = None,
            **plt_kw: Any
    ) -> tuple[Figure, Axes]:
        """Main method of the plotting API of SingularSpectrumAnalysis

        The ´plot´ method generates plots of various kinds to explore the
        eigentriple features and reconstructed time series from
        ´SingularSpectrumAnalysis´ instance.

        Parameters
        ----------
        kind : str, default 'values'
            The type of plot to produce, options include:

            * 'matrix': Plots the decomposed or reconstructed matrix.
            * 'paired': Plots pairs of successive left eigenvectors against
              each other.
            * 'periodogram': Plots power spectral density associated with each
              eigenvector.
            * 'timeseries': Displays reconstructed time series based on defined
              component groups.
            * 'values': Plots singular values to inspect their magnitudes.
            * 'vectors': Plots the left eigenvectors.
            * 'wcorr': Displays a weighted correlation matrix using a heatmap.

        n_components : int | None, default 10
            Number of eigentriple components to use in the plot. Only valid for
            kind 'matrix', 'paired', 'periodogram', 'values', 'vectors', and
            'wcorr'.
        ax : Axes, optional
            An existing matplotlib Axes object to draw the plot on. If None, a
            new figure and axes are created. This parameter is ignored for
            subplots, i.e., kind 'paired', 'periodogram', and 'vectors'.
        scale: str, optional
            Scale for 'periodogram' kind of plots. One of:

            - 'loglog': logarithmic scaling on both axes
            - 'plot': linear scaling on both axes
            - 'semilogx': logarithmic scaling on x-axis only
            - 'semilogy': logarithmic scaling on y-axis only
        plt_kw : Any, optional
            Additional keyword arguments for customization of the plot, passed
            to the respective plotting function. The specific function used
            depends on the 'kind' of plot:

            - 'matrix': `matplotlib.pyplot.imshow`
            - 'paired', 'values', 'vectors': `matplotlib.pyplot.plot`
            - 'periodogram': `matplotlib.pyplot.semilogy`
            - 'timeseries': `pandas.DataFrame.plot`
            - 'wcorr': `matplotlib.pyplot.imshow`

            See Examples.

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

            >>> from vassal.ssa import SingularSpectrumAnalysis
            >>> from vassal.datasets import load_sst
            >>> sst = load_sst()
            >>> ssa = SingularSpectrumAnalysis(sst)
            >>> ssa.decompose()
            >>> ssa.available_plots()
            ['matrix', 'paired', 'periodogram', 'timeseries', 'values', 'vectors', 'wcorr']

            >>> ssa.plot(kind='values', n_components=30, marker='.', ls='--')

        """

        if n_components is None:
            n_components = self.n_components  # - 1 ?

        # Raise error if no decomposition except for allowed plot kinds
        if self.n_components is None and kind in self._n_components_required:
            raise DecompositionError(
                "Decomposition must be performed before calling the 'plot' "
                f"method with with kind='{kind}'. Make sure to call the"
                "'decompose' method before the 'plot' method.")

        if kind in self._n_components_required:
            n_components = self.__validate_n_components(n_components)

        if kind not in self._plot_kinds_map.keys():
            valid_kinds = ','.join(self._plot_kinds_map.keys())
            raise ValueError(f"Unknown plot kind '{kind}'. "
                             f"Valid plot kinds are {valid_kinds}.")

        plot_method = getattr(self, self._plot_kinds_map[kind])
        fig, ax = plot_method(
            n_components=n_components,
            ax=ax,
            scale=scale,
            **plt_kw
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        return fig, ax

    @abc.abstractmethod
    def to_frame(
            self,
            include: list[str] | None = None,
            exclude: list[str] | None = None,
            recenter: bool = False,
            rescale: bool = False
    ) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def _reconstruct_group_matrix(
            self,
            group_indices: int | slice | range | list[int]
    ) -> NDArray:
        pass

    @abc.abstractmethod
    def _reconstruct_group_timeseries(
            self,
            group_indices: int | slice | range | list[int]
    ) -> NDArray:
        pass

    @ignored_argument_warning('n_components', 'scale', log_level='info')
    def _plot_matrix(
            self,
            indices: int | range | list[int] = None,
            ax: Axes | None = None,
            cmap: str | Colormap | None = 'viridis',
            **plt_kw
    ):
        """Plot decomposed or reconstructed matrices.
        # TODO: refactor and improve (including main doc)
        """
        if not ax:
            fig = plt.figure()
            ax = fig.gca()
        else:
            fig = ax.get_figure()

        if indices is None:
            matrix = self.svd_matrix
            subtitle = f'({self._svd_matrix_kind}, Original)'
        else:
            if self.n_components is None:
                raise DecompositionError(
                    "Cannot plot reconstructed matrix prior to decomposition. "
                    "Make sure to call the 'decompose' method first."
                )
            matrix = self._reconstruct_group_matrix(group_indices=indices)
            subtitle = f'({self._svd_matrix_kind}, Group:{indices})'

        im = ax.imshow(matrix, cmap=cmap, **plt_kw)
        ax.set_aspect('equal')
        ax.set_title(f'SVD Matrix {subtitle}')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        fig.colorbar(im)

        return fig, ax

    @ignored_argument_warning('ax', 'scale')
    def _plot_paired_vectors(
            self,
            n_components: int,
            **plt_kw
    ) -> tuple[Figure, Axes]:
        """Plot successive paired left eigenvectors.
        """
        pairs = list(zip(range(0, n_components - 1), range(1, n_components)))
        u = self.u_
        eigenvalues = self.eigenvalues
        squared_frobenius_norm = self.squared_frobenius_norm

        rows, cols = self._auto_subplot_layout(len(pairs))

        fig, axes = plt.subplots(rows, cols, figsize=(1.5 * cols, 1.5 * rows))

        for i in range(rows * cols):
            ax = axes.ravel()[i] if isinstance(axes, np.ndarray) else axes
            try:
                j, k = pairs[i]
                ax.plot(u[:, k], u[:, j], **plt_kw)
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

    @ignored_argument_warning('ax')
    def _plot_periodogram(
            self,
            n_components: int,
            scale: Literal['loglog', 'plot', 'semilogx', 'semilogy'] | None,
            **plt_kw
    ) -> tuple[Figure, Axes]:
        """Plot the power spectral density of signals associated with eigenvectors.

        # TODO: offer the possiblity to show the original periodogram if
        # TODO: ncomp is 0 or None?

        """

        freq_original, psd_original = periodogram(self._timeseries_pp)

        unit = None

        if self._ix is not None:
            if isinstance(self._ix, pd.DatetimeIndex):
                unit = pd.infer_freq(self._ix)

        if unit is None:
            unit = ''

        if scale is None:
            scale = 'loglog'

        rows, cols = self._auto_subplot_layout(n_components)

        fig, axes = plt.subplots(rows, cols, figsize=(1.5 * cols, 1.5 * rows))

        for i in range(rows * cols):
            ax = axes.ravel()[i]
            ax.axis('off')
            plot_method = getattr(ax, scale)

            if i >= self.n_components:
                continue
            plot_method(freq_original[1:], psd_original[1:], lw=.5,
                        color='lightgrey')
            freq, psd = periodogram(self[i])
            plot_method(freq[1:], psd[1:], **plt_kw)
            dominant_freq = freq[
                np.argmax(psd)]  # TODO: rely on proprety dominant_frequencies
            period = 1 / dominant_freq
            title = f'EV{i} (T={period:.1f}{unit})'
            ax.set_title(title, {'fontsize': 'small'})
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('auto', 'box')

        return fig, fig.get_axes()

    @ignored_argument_warning('n_components', 'scale', log_level='info')
    def _plot_timeseries(
            self,
            ax: Axes,
            include: list[str] | None = None,
            exclude: list[str] | None = None,
            subplots: bool = False,
            recenter: bool = False,
            rescale: bool = False,
            **plt_kw
    ) -> tuple[Figure, Axes]:
        """Plot all or selected timeseries as a single or a subplot
        """
        data = self.to_frame(include, exclude, recenter, rescale)

        if subplots and include is not None:
            data = data.reindex(columns=include)

        axes = data.plot(subplots=subplots, ax=ax, **plt_kw)

        if isinstance(axes, np.ndarray):
            fig = axes[0].get_figure()
            for ax in axes:
                ax.legend(loc='upper left')
        else:
            fig = axes.get_figure()
            axes.legend(loc='best')

        fig.tight_layout()

        return fig, axes

    @ignored_argument_warning('scale')
    def _plot_values(
            self,
            n_components: int,
            rank_by: Literal['values', 'freq'] = 'values',
            conf_level: float | None = None,
            two_tailed: bool = True,
            ax: Axes | None = None,
            **plt_kw
    ) -> tuple[Figure, Axes]:
        """Plot component norms.
        # TODO: update tutorial and test the new freq feature (rank_by etc)
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

        ax.semilogy(x_values, s, **plt_kw)
        ax.set_ylabel('Component Norm')
        ax.set_xlabel(x_label)
        if rank_by == 'values':
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        if hasattr(self, 'n_surrogates'):
            lower, upper = self.get_confidence_interval(
                n_components,
                conf_level,
                two_tailed,
            )

            center = (lower + upper) / 2
            errorbar_length = (upper - lower) / 2

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
                      f'{conf_level * 100:g}% CI'
                      f'\nn={self.n_surrogates}')
            ax.legend()

        fig = ax.get_figure()

        return fig, ax

    @ignored_argument_warning('ax', 'scale')
    def _plot_vectors(
            self,
            n_components: int,
            **plt_kw
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
                ax.plot(u[:, i], **plt_kw)
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

    @ignored_argument_warning('scale')
    def _plot_wcorr(
            self,
            n_components: int,
            ax: Axes | None = None,
            cmap: str | Colormap | None = 'PiYG',
            vmin: float | None = -1.,
            vmax: float | None = 1.,
            **plt_kw
    ):
        """Plot the weighted correlation matrix.
        """
        wcorr = self.wcorr(n_components)

        if not ax:
            fig = plt.figure()
            ax = fig.gca()
        else:
            fig = ax.get_figure()

        im = ax.pcolor(wcorr, vmin=vmin, vmax=vmax, cmap=cmap, **plt_kw)
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

    def __validate_n_components(
            self,
            n_components: int,
    ):
        """Validate the number of components requested for plotting.

        Parameters
        ----------
        n_components : int
            User defined number of components

        Returns
        -------
        n_components : int
            Validated number of components

        Raises
        ------
        ValueError
            If n_components is not a strictly positive integer.

        Warnings
        --------
        Out of range n_components is warned to the users and automatically
        corrected to the maximum allowed.

        """
        if not isinstance(n_components, int) or n_components < 1:
            raise ValueError("Parameter 'n_components' must be a strictly "
                             "positive integer.")
        if n_components > self.n_components:
            logger = logging.getLogger(self.__module__)
            log_func = getattr(logger, 'warning')
            log_func(
                f"Parameter 'n_components={n_components}' is out of range. "
                f"Value has been set to the maximum value "
                f"'n_components={self.n_components}'.")
            n_components = self.n_components

        return n_components

    @staticmethod
    def _auto_subplot_layout(n_plots: int) -> tuple[int, int]:
        """Calculate the optimal layout for a given number of subplots

        The method favors a 3-column layout until 9 plots, and then favors a
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

            # Adjust to favor more columns than rows if not a perfect square
            if rows * cols < n_plots:
                rows = cols
            elif rows > cols:
                cols = rows

        return rows, cols

    @classmethod
    def available_plots(cls) -> list[str]:
        """ List of available plot kinds.
        """
        return list(cls._plot_kinds_map.keys())

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
            freq, psd = periodogram(self[i])
            dominant_freqs.append(freq[np.argmax(psd)])

        return np.array(dominant_freqs)


if __name__ == '__main__':
    from doctest import testmod

    testmod()
