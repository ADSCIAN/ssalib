import logging
import re
from inspect import signature

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

matplotlib.use('Agg')

from vassal.log_and_error import DecompositionError
from vassal.plotting import PlotSSA, SSAPlotType
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
    assert hasattr(ssa_no_decomposition, '_PLOT_METHOD_MAPPING')
    for plot_type in SSAPlotType:
        assert plot_type in ssa_no_decomposition._PLOT_METHOD_MAPPING.keys()
    for value in ssa_no_decomposition._PLOT_METHOD_MAPPING.values():
        assert hasattr(ssa_no_decomposition, value)
        assert callable(getattr(ssa_no_decomposition, value))


@pytest.mark.parametrize("plot_kind", SSAPlotType.available_plots())
def test_plot_methods(ssa_with_reconstruction, plot_kind) -> None:
    fig, ax = ssa_with_reconstruction.plot(kind=plot_kind)
    assert isinstance(fig, Figure)
    assert isinstance(ax, (Axes, np.ndarray, list))
    plt.close()


@pytest.mark.parametrize("plot_kind", SSAPlotType.available_plots())
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


@pytest.mark.parametrize("plot_kind", SSAPlotType.available_plots())
def test_requires_decomposition(ssa_no_decomposition, plot_kind):
    if SSAPlotType(plot_kind).requires_decomposition:
        with pytest.raises(
                DecompositionError,
                match=f"Decomposition must be performed before calling the "
                      f"'plot' method with kind='{plot_kind}'. Call "
                      f"'decompose' method first"
        ):
            ssa_no_decomposition.plot(kind=plot_kind)


@pytest.mark.parametrize("plot_kind", SSAPlotType.available_plots())
def test_n_components_none(ssa_with_decomposition, plot_kind):
    ssa_with_decomposition.plot(kind=plot_kind, n_components=None)


@pytest.mark.parametrize("plot_kind", SSAPlotType.available_plots())
def test_n_components_out_of_range(ssa_with_decomposition, plot_kind, caplog):
    if SSAPlotType(plot_kind).requires_decomposition:
        expected_match = (
            "Argument 'n_components' must be less than or equal to "
            "the number of components (10), got 11")
        with pytest.raises(
                ValueError,
                match=re.escape(expected_match)
        ):
            ssa_with_decomposition.plot(kind=plot_kind, n_components=11)


@pytest.mark.parametrize("plot_kind", SSAPlotType.available_plots())
def test_n_components_wrong_type(ssa_with_decomposition, plot_kind):
    if SSAPlotType(plot_kind).requires_decomposition:
        with pytest.raises(
                TypeError,
                match="Argument 'n_components' must be integer or None, "
                      "got <class 'str'>"
        ):
            ssa_with_decomposition.plot(
                kind=plot_kind,
                n_components='wrong_type'
            )


@pytest.mark.parametrize("plot_kind", SSAPlotType.available_plots())
def test_n_components_ignored_warning(ssa_with_decomposition, plot_kind,
                                      caplog):
    # Trigger the case where n_components is ignored for the specific plot kind
    with caplog.at_level(logging.WARNING):
        ssa_with_decomposition.plot(kind=plot_kind, n_components=5)

        # Collect the log messages from the records
        log_messages = [record.message for record in caplog.records]

        # Verify that the appropriate warning was logged
        if not SSAPlotType(plot_kind).requires_decomposition:
            assert any(
                "Parameter 'n_components' is not supported for plot kind" in message
                for message in log_messages
            ), "Warning for 'n_components' not logged as expected"
        else:
            assert not any(
                "Parameter 'n_components' is not supported for plot kind" in message
                for message in log_messages
            ), "Unexpected warning for 'n_components'"


@pytest.mark.parametrize("plot_kind", SSAPlotType.available_plots())
def test_ax_ignored_warning(ssa_with_decomposition, plot_kind, caplog):
    # Create a matplotlib axis object.
    fig, ax = plt.subplots()

    # Trigger the case where ax is ignored for the specific plot kind
    with caplog.at_level(logging.WARNING):
        ssa_with_decomposition.plot(kind=plot_kind, ax=ax)

        # Collect the log messages from the records
        log_messages = [record.message for record in caplog.records]

        # Verify that the appropriate warning was logged, based on supports_ax
        if not SSAPlotType(plot_kind).supports_ax:
            assert any(
                "Parameter 'ax' is not supported for plot kind" in message
                for message in log_messages
            ), "Warning for 'ax' not logged as expected"
        else:
            assert not any(
                "Parameter 'ax' is not supported for plot kind" in message
                for message in log_messages
            ), "Unexpected warning for 'ax'"
    # Close the figure to free memory
    plt.close(fig)

@pytest.mark.parametrize("plot_kind", [
    "periodogram"])  # Parameterize to only focus on periodogram
def test_imputation_warning(ssa_with_decomposition_fill_mean, plot_kind,
                            caplog):
    # Trigger the case where the periodogram with imputed values is plotted
    with caplog.at_level(logging.WARNING):
        ssa_with_decomposition_fill_mean.plot(kind=plot_kind)

        # Collect the log messages from the records
        log_messages = [record.message for record in caplog.records]

        # Verify that the appropriate warning was logged about the imputation strategy
        assert any(
            f"Periodogram is estimated on a series imputed with strategy 'fill_mean'" in message
            for message in log_messages
        ), "Warning for imputation strategy not logged as expected"


def test_auto_subplot_layout():
    PlotSSA._auto_subplot_layout(3)
    PlotSSA._auto_subplot_layout(9)
    PlotSSA._auto_subplot_layout(13)
    PlotSSA._auto_subplot_layout(17)
