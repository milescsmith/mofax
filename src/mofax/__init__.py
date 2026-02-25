from importlib.metadata import PackageNotFoundError, version

from .core import mofa_model
from .plot_data import plot_data_overview
from .plot_factors import plot_factors
from .plot_mefisto import plot_group_kernel, plot_sharedness
from .plot_utils import _plot_grid_from_1d
from .plot_variance import plot_r2, plot_r2_barplot, plot_r2_pvalues
from .plot_weights import plot_weights

try:
    if isinstance(__package__, str):
        __version__ = version(__package__)
    else:
        __version__ = "unknown"
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"


__all__: list[str] = [
    "_plot_grid_from_1d",
    "mofa_model",
    "plot_data_overview",
    "plot_factors",
    "plot_group_kernel",
    "plot_r2",
    "plot_r2_barplot",
    "plot_r2_pvalues",
    "plot_sharedness",
    "plot_weights",
]
