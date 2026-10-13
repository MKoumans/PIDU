"""Export of analysis results to Excel workbooks and LaTeX tables."""

from pathlib import Path

import pandas as pd

from .timings import timings_to_latex_table

SHEET_PREFIXES = {
    'get_sample_info': 'sam_info',
    'get_p_distribution': 'p_dist',
    'get_timings': 'time',
    'get_usecase2_parameter_tracking': 'p_track',
    'get_unfolds_loss_comparison': 'unfolds_loss',
}
EXCEL_SHEET_NAME_MAX = 31


def export_to_excel(results: dict[str, dict[str, pd.DataFrame]], export_path: Path):
    """Write every result DataFrame to one sheet of a single workbook."""
    print(f'Exporting data figures to {export_path}...')
    with pd.ExcelWriter(export_path) as writer:
        for func_name, data_dict in results.items():
            prefix = SHEET_PREFIXES.get(func_name, 'func')
            for key, df in data_dict.items():
                sheet_name = f'{prefix}_{key}'[-EXCEL_SHEET_NAME_MAX:]
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f' - Exported sheet {sheet_name}')
    print(f'Data figures exported to {export_path}')


def export_timings_latex(timing_summary: pd.DataFrame, export_path: Path) -> bool:
    """Write the timing summary as a LaTeX table; returns False when empty."""
    if timing_summary.empty:
        print('Skipping timing table: no LMA timing files or learned-model checkpoints were found.')
        return False

    print(f'Exporting timing table to {export_path}...')
    label = f"tab:timings_{export_path.stem.split('_')[0].replace('usecase', 'case')}"  # usecase<N>_timings_table.tex
    export_path.write_text(timings_to_latex_table(timing_summary, label=label))
    print(f'LaTeX table exported to {export_path}')
    return True
