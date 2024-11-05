import logging
from inspect import signature

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

matplotlib.use('Agg')

from vassal.log_and_error import DecompositionError
from vassal.plotting import PlotSSA
from vassal.ssa import SingularSpectrumAnalysis


def test_PlotSSA_instantiation():
    """Test that PlotSSA cannot be instantiated as it's an abstract class."""
    with pytest.raises(TypeError) as excinfo:
        PlotSSA()

    # Check that the error message mentions all required abstract methods
    error_msg = str(excinfo.value)
    assert all(method in error_msg for method in [
        '_reconstruct_group_matrix',
        '_reconstruct_group_timeseries',
        'to_frame'
    ]), "Error message should mention all abstract methods"


@pytest.mark.parametrize("abstract_method, concrete_method", [
    (PlotSSA.to_frame, SingularSpectrumAnalysis.to_frame),
    (PlotSSA._reconstruct_group_matrix,
     SingularSpectrumAnalysis._reconstruct_group_matrix),
    (PlotSSA._reconstruct_group_timeseries,
     SingularSpectrumAnalysis._reconstruct_group_timeseries),
])
def test_methods_signature(abstract_method, concrete_method):
    abstract_sig = signature(abstract_method)
    concrete_sig = signature(concrete_method)
    assert abstract_sig == concrete_sig, \
        (f"Signature mismatch for '{abstract_method.__name__}' and "
         f"'{concrete_method.__name__}': {abstract_sig} != {concrete_sig}")


def test_plot_kind_hasattr(ssa_no_decomposition):
    assert hasattr(ssa_no_decomposition, '_plot_kinds_map')
    for value in ssa_no_decomposition._plot_kinds_map.values():
        assert hasattr(ssa_no_decomposition, value)
        assert callable(getattr(ssa_no_decomposition, value))


@pytest.mark.parametrize("plot_kind", PlotSSA.available_plots())
def test_plot_methods(ssa_with_reconstruction, plot_kind) -> None:
    fig, ax = ssa_with_reconstruction.plot(kind=plot_kind)
    assert isinstance(fig, Figure)
    assert isinstance(ax, (Axes, np.ndarray, list))
    plt.close()


@pytest.mark.parametrize("plot_kind", PlotSSA.available_plots())
def test_plot_methods_low_window(timeseries50, plot_kind) -> None:
    ssa = SingularSpectrumAnalysis(timeseries50, window=2)
    ssa.decompose()
    fig, ax = ssa.plot(kind=plot_kind)
    assert isinstance(fig, Figure)
    assert isinstance(ax, (Axes, np.ndarray, list))
    plt.close()


def test_unknown_plot_kind(ssa_with_reconstruction):
    with pytest.raises(ValueError, match="Unknown plot kind"):
        ssa_with_reconstruction.plot(kind='unknown')


@pytest.mark.parametrize("plot_kind", PlotSSA.available_plots())
def test_unknown_plot_kind(ssa_no_decomposition, plot_kind):
    if plot_kind in PlotSSA._n_components_required:
        with pytest.raises(
                DecompositionError,
                match="Decomposition must be performed before "
                      f"calling the 'plot' method with with kind='{plot_kind}'."
        ):
            ssa_no_decomposition.plot(kind=plot_kind)


@pytest.mark.parametrize("plot_kind", PlotSSA.available_plots())
def test_n_components_none(ssa_with_decomposition, plot_kind):
    ssa_with_decomposition.plot(kind=plot_kind, n_components=None)


@pytest.mark.parametrize("plot_kind", PlotSSA.available_plots())
def test_n_components_out_of_range(ssa_with_decomposition, plot_kind, caplog):
    if plot_kind in PlotSSA._n_components_required:
        with caplog.at_level(logging.WARNING):
            ssa_with_decomposition.plot(kind=plot_kind, n_components=11)
        assert len(caplog.records) == 1
        assert "Parameter 'n_components=11' is out of range." in \
               caplog.records[0].message


@pytest.mark.parametrize("plot_kind", PlotSSA.available_plots())
def test_n_components_wrong_type(ssa_with_decomposition, plot_kind):
    if plot_kind in PlotSSA._n_components_required:
        with pytest.raises(
                ValueError,
                match="Parameter 'n_components' must be a strictly positive "
                      "integer."
        ):
            ssa_with_decomposition.plot(
                kind=plot_kind,
                n_components='wrong_type')


def test_auto_subplot_layout():
    PlotSSA._auto_subplot_layout(3)
    PlotSSA._auto_subplot_layout(9)
    PlotSSA._auto_subplot_layout(13)
    PlotSSA._auto_subplot_layout(17)
