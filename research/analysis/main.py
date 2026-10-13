"""Command-line entry point that produces the data tables and figures for the
case-study chapters.

Reads the outputs of the training runs (pidu.train) and the LMA
baseline (pidu.baselines.lma), and writes comparison tables, timing
benchmarks and figures to outputs/case<usecase>/figures/.

Usage:
    python -m research.analysis -c case1 -u 1 -e
    python -m research.analysis -c case2 -u 2 -e
    python -m research.analysis --summary
"""

import argparse

from .datasets import get_p_distribution, get_sample_info, load_dataset
from .export import export_timings_latex, export_to_excel
from .timings import get_timings
from .tracking import get_unfolds_loss_comparison, get_usecase2_parameter_tracking
from .utils import figures_dir
from pidu.paths import DATASETS_DIR

EXP_DS = 'usecase2/case2_experimental'  # available on request; without it, use case 2 skips its parts


def parse_args():
    parser = argparse.ArgumentParser(description='Get data figures for case studies.')

    parser.add_argument('-c', '--chapter', type=str, default='case1', help='Chapter identifier (case0|case1|case2).')
    parser.add_argument('-u', '--usecase', type=int, default=1, help='Use case number.')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='Device to use (cpu|cuda).')
    parser.add_argument('--sizes', type=int, nargs=3, metavar=('N_TRAIN', 'N_VAL', 'N_TEST'), default=(14, 10, 10),
                        help='log2 train, validation, and test dataset sizes.')
    parser.add_argument('-e', '--export', action='store_true', help='Export results to Excel file.')
    parser.add_argument('--summary', action='store_true',
                        help='Print a summary of the data produced per use case, then exit.')

    return parser.parse_args()


def _build_tasks(args) -> dict:
    """Map each analysis function name to a zero-argument callable for the chapter."""
    if args.chapter == 'case0':
        # Supplementary/discussion material.
        return {'get_unfolds_loss_comparison': get_unfolds_loss_comparison}

    n_train, n_val, n_test = args.sizes
    y_mode = 'both' if args.chapter == 'case2' else 'noised'
    has_exp = args.chapter == 'case2' and (DATASETS_DIR / f'{EXP_DS}.npz').is_file()
    if args.chapter == 'case2' and not has_exp:
        print(f'{EXP_DS}.npz not found (available on request): skipping its parameter distribution and the parameter tracking.')

    def named_ds(split: str, n: int, y_return_mode: str = 'both'):
        return load_dataset(
            filename=f'usecase{args.usecase}/case{args.usecase}_{split}_n2_{n}',
            device=args.device,
            y_return_mode=y_return_mode,
        )

    test_ds = named_ds('test', n_test, y_return_mode=y_mode)

    def p_distribution():
        datasets = {
            'ds_train': named_ds('train', n_train),
            'ds_val': named_ds('val', n_val),
            'ds_test': test_ds,
        }
        if has_exp:
            datasets['ds_exp'] = load_dataset(filename=EXP_DS, device=args.device, y_return_mode='both')
        return get_p_distribution(datasets)

    def timings():  # use case 2 is timed on the experimental set, use case 1 on the simulated test set
        timing_ds = load_dataset(filename=EXP_DS, device=args.device, y_return_mode='both') if has_exp else test_ds
        return get_timings(args.usecase, args.device, timing_ds)

    tasks = {
        'get_sample_info': lambda: get_sample_info(test_ds, sample_idx=0),
        'get_p_distribution': p_distribution,
        'get_timings': timings,
    }
    if has_exp:
        tasks['get_usecase2_parameter_tracking'] = lambda: get_usecase2_parameter_tracking(args.device)

    return tasks


def get_data_figures(args):
    """Run all analysis functions for the requested chapter and optionally export."""
    results = {}
    for func_name, task in _build_tasks(args).items():
        results[func_name] = task()
        print(f'Finished {func_name}')

    if not args.export:
        return results

    export_dir = figures_dir(args.usecase)
    export_to_excel(results, export_dir / f'usecase{args.usecase}_data_figures.xlsx')

    if 'get_timings' in results:
        export_timings_latex(
            results['get_timings']['timing_summary'],
            export_dir / f'usecase{args.usecase}_timings_table.tex',
        )

    return results


def summary():
    """Print a human-readable summary of the data produced per chapter."""
    print(
        'analysis package'
        '\n================'
        '\nProduces figures and data tables for each case-study chapter.'
        '\nNo files are written when called with --summary.'
        '\nUse --sizes N_TRAIN N_VAL N_TEST to select the generated datasets (default: 14 10 10).'
        '\n'
        '\nData functions (see the analysis submodules for details)'
        '\n---------------------------------------------------------'
        '\n  datasets.get_sample_info(ds, sample_idx)   -> dict[str, DataFrame]'
        '\n      One sample per system: signal, region boundaries, system & oscillator params.'
        '\n  datasets.get_p_distribution(ds_dict)       -> dict[str, DataFrame]'
        '\n      True-parameter distributions per dataset/system/layer.'
        '\n  timings.get_timings(usecase, device, ds)   -> dict[str, DataFrame]'
        '\n      Forward-pass timing benchmarks for LMA and CNN/DU/PIDU, plus MAPE.'
        '\n  timings.timings_to_latex_table(summary)    -> str'
        '\n      LaTeX table of the timing summary.'
        '\n  tracking.get_usecase2_parameter_tracking(device) -> dict[str, DataFrame]'
        '\n      GT vs. method oscillator-parameter comparison + 3x3 tracking plot.'
        '\n  tracking.get_unfolds_loss_comparison()     -> dict[str, DataFrame]'
        '\n      Train/val last-5-epoch loss summary per unfold count.'
        '\n'
        '\nPer-chapter data produced by get_data_figures()'
        '\n------------------------------------------------'
        '\n  case0  ->  get_unfolds_loss_comparison'
        '\n  case1  ->  get_sample_info, get_p_distribution, get_timings (+ LaTeX table on export)'
        '\n  case2  ->  case1 functions + get_usecase2_parameter_tracking'
    )


def main():
    args = parse_args()
    if args.summary:
        summary()
        return
    get_data_figures(args)
