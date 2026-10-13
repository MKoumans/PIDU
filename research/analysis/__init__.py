"""Analysis package: builds the data tables, timing benchmarks and figures for
the case-study chapters from previously produced training/LMA outputs.

Run as a script with `python -m research.analysis` (see analysis.main for the CLI).
"""

from .datasets import (
    flatten_oscillator_params,
    get_p_distribution,
    get_sample_info,
    load_dataset,
)
from .export import export_timings_latex, export_to_excel
from .main import get_data_figures, main
from .timings import collect_mape_by_method, get_timings, timings_to_latex_table
from .tracking import get_unfolds_loss_comparison, get_usecase2_parameter_tracking

__all__ = [
    'flatten_oscillator_params',
    'get_p_distribution',
    'get_sample_info',
    'load_dataset',
    'export_timings_latex',
    'export_to_excel',
    'get_data_figures',
    'main',
    'collect_mape_by_method',
    'get_timings',
    'timings_to_latex_table',
    'get_unfolds_loss_comparison',
    'get_usecase2_parameter_tracking',
]
