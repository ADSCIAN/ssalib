import abc
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from vassal.log_and_error import DecompositionError, ignored_argument_warning
from vassal.wcorr import correlation_weights, weighted_correlation_matrix


class PlotSSA(metaclass=abc.ABCMeta):
    """
    PlotSSA is an abstract base class that defines the plotting interface for
    the SingularSpectrumAnalysis class.
    """
    n_components: int | None = None  # number of components
    eigenvalues: np.ndarray | None = None  # array of eigenvalues
    total_variance: float | None = None  # sum of eigenvalues
    _n: int | None = None  # timeseries length
    _w: int | None = None  # SSA window length
    _u: np.ndarray | None = None  # left eigenvectors
    _s: np.ndarray | None = None  # array of singular values
    _v: np.ndarray | None = None  # right eigenvectors

    _plot_kinds_map = {
        'paired': '_plot_paired_vectors',
        'timeseries': '_plot_timeseries',
        'values': '_plot_values',
        'vectors': '_plot_vectors',
        'wcorr': '_plot_wcorr'
    }
    _n_components_required = ['paired', 'values', 'vectors', 'wcorr']

    def plot(
            self,
            kind: str = 'values',
            n_components: int | None = 10,
            ax: Axes = None,
            **plt_kw
    ) -> tuple[Figure, Axes]:
        """
        Generates plots of various kinds to explore the eigentriple features
        and reconstructed time series from Singular Spectrum Analysis.

        Parameters
        ----------
        kind : str, default 'values'
            The type of plot to produce, options include:
            - 'paired': Plots pairs of successive left eigenvectors against
                each other.
            - 'timeseries': Displays reconstructed time series based on defined
                component groups.
            - 'values': Plots singular values to inspect their magnitudes.
            - 'vectors': Plots the left eigenvectors.
            - 'wcorr': Displays a weighted correlation matrix using a heatmap.
        n_components : int | None, default 10
            Number of eigentriple components to use in the plot. Only valid for
            kind 'paired', 'values', 'vectors', and 'wcorr'. If None, use the
            maximum value
        ax : Axes, optional
            An existing matplotlib Axes object to draw the plot on. If None, a
            new figure and axes are created. This parameter is ignored for
            subplots, i.e., kind 'paired' and 'vectors'.
        plt_kw : dict, optional
            Additional keyword arguments for customization of the plot, passed
            to the respective plotting function.
            The specific function used depends on the 'kind' of plot:
            - 'paired', 'values', 'vectors': `matplotlib.pyplot.plot`
            - 'timeseries': `pandas.DataFrame.plot`
            - 'wcorr': `matplotlib.pyplot.imshow`
            See the corresponding documentation for more information.

        Returns
        -------
        tuple[Figure, Axes]
            A tuple containing the matplotlib Figure and Axes objects with the
            generated plot. This allows further customization after the
            function returns.

        Examples
        --------

        >>> from vassal.ssa import SingularSpectrumAnalysis
        >>> from vassal.datasets import load_sst
        >>> sst = load_sst()
        >>> ssa = SingularSpectrumAnalysis(sst)
        >>> u, s, v = ssa.decompose()
        >>> ssa.plot(kind='values')  # doctest: +SKIP

        """

        if n_components is None:
            n_components = self.n_components

        # Raise error if no decomposition except for plot kind 'timeseries'
        if self.n_components is None and kind != 'timeseries':
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

    def wcorr(
            self,
            n_components: int
    ) -> np.ndarray:
        """
        Parameters
        ----------
        n_components : int
            The number of components used to compute the weighted correlation
            matrix.

        Returns
        -------
        wcorr : np.ndarray
            The weighted correlation matrix.

        References
        ----------


        #TODO add

        """
        timeseries = np.array(
            [self._reconstruct_group([i]) for i in range(n_components)])
        weights = correlation_weights(self._n, self._w)
        wcorr = weighted_correlation_matrix(timeseries, weights=weights)
        return wcorr

    @abc.abstractmethod
    def _reconstruct_group(
            self,
            group_indices: int | slice | range | list[int]
    ) -> np.ndarray:
        pass

    @ignored_argument_warning('ax')
    def _plot_paired_vectors(
            self,
            n_components: int,
            **plt_kw
    ) -> tuple[Figure, Axes]:
        pairs = list(zip(range(0, n_components - 1), range(1, n_components)))
        u = self._u
        eigenvalues = self.eigenvalues
        total_variance = self.total_variance

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
                captured_var1 = eigenvalues[j] / total_variance * 100
                captured_var2 = eigenvalues[k] / total_variance * 100

                title = (f'EV{j} ({captured_var1:.1f}%) vs.'
                         f' {j + 1} ({captured_var2:.1f}%)')

                ax.set_title(title, {'fontsize': 'small'})
            except IndexError:
                ax.axis('off')

        return fig, axes

    @ignored_argument_warning('n_components', log_level='info')
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
        data = self.to_frame(include, exclude, recenter, rescale)
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

    def _plot_values(
            self,
            n_components: int,
            ax: Axes = None,
            **plt_kw
    ) -> tuple[Figure, Axes]:
        s = self._s

        if not ax:
            fig = plt.figure()
            ax = fig.gca()

        ax.semilogy(s[:n_components], **plt_kw)
        ax.set_ylabel('Singular Values')
        ax.set_xlabel('Eigentriple Index')

        fig = ax.get_figure()

        return fig, ax

    @ignored_argument_warning('ax')
    def _plot_vectors(
            self,
            n_components: int,
            **plt_kw
    ):
        u = self._u
        total_variance = self.total_variance

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
                captured_variance = self.eigenvalues[i] / total_variance * 100

                title = f'EV{i} ({captured_variance:.1f}%)'
                ax.set_title(title, {'fontsize': 'small'})
            except IndexError:
                ax.axis('off')

        return fig, fig.get_axes()

    def _plot_wcorr(
            self,
            n_components: int,
            ax: Axes | None = None,
            cmap: str | Colormap | None = 'PiYG',
            vmin: float | None = -1.,
            vmax: float | None = 1.,
            **plt_kw
    ):
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
        """
        Validate the number of components used for plotting.

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
        """
        Calculate the optimal layout for a given number of subplots that favors
        a 3-column layout until 9 plots, and then favors a squared layout with
        possibly more columns than rows.

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


if __name__ == '__main__':
    from doctest import testmod

    testmod()
