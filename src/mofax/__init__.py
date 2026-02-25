from importlib.metadata import PackageNotFoundError, version

from .core import mofa_model
from .plot_data import plot_data_overview
from .plot_factors import (
    plot_factors_correlation,
    plot_factors_covariates_correlation,
    plot_factors_dotplot,
    plot_factors_matrix,
    plot_factors_scatter,
    plot_factors_umap,
    plot_factors_violin,
    plot_projection,
)
from .plot_mefisto import plot_group_kernel, plot_interpolated_factors, plot_sharedness, plot_smoothness
from .plot_variance import plot_r2, plot_r2_barplot, plot_r2_pvalues
from .plot_weights import (
    plot_weights,
    plot_weights_correlation,
    plot_weights_dotplot,
    plot_weights_heatmap,
    plot_weights_ranked,
    plot_weights_scaled,
    plot_weights_scatter,
)

try:
    if isinstance(__package__, str):
        __version__ = version(__package__)
    else:
        __version__ = "unknown"
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"


__all__: list[str] = [
    "mofa_model",
    "plot_data_overview",
    "plot_factors",
    "plot_factors_correlation",
    "plot_factors_covariates_correlation",
    "plot_factors_dotplot",
    "plot_factors_matrix",
    "plot_factors_scatter",
    "plot_factors_umap",
    "plot_factors_violin",
    "plot_group_kernel",
    "plot_interpolated_factors",
    "plot_projection",
    "plot_r2",
    "plot_r2_barplot",
    "plot_r2_pvalues",
    "plot_sharedness",
    "plot_smoothness",
    "plot_weights",
    "plot_weights_correlation",
    "plot_weights_dotplot",
    "plot_weights_heatmap",
    "plot_weights_ranked",
    "plot_weights_scaled",
    "plot_weights_scatter",
]
