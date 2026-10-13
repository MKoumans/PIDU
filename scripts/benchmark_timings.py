"""Time the released CNN/DU/PIDU checkpoints (b = 1 and b = 256) -> outputs/fig-summary/timing_case<N>.csv.

The input of scripts/collect_figure_summary.py. Timings depend on the hardware; the TRF timings are not
produced here but read from the LMA runs (timing_per_sample.csv).

    python scripts/benchmark_timings.py [--usecase 1 2] [--device cuda] [--batch-sizes 1 256]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

import torch  # noqa: E402

from research.analysis.datasets import load_dataset  # noqa: E402
from research.analysis.timings import get_timings  # noqa: E402

TEST_SIZE_LOG2 = 10


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--usecase', type=int, nargs='+', default=[1, 2])
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 256])
    ap.add_argument('--out-dir', type=Path, default=ROOT / 'outputs' / 'fig-summary')
    a = ap.parse_args()

    a.out_dir.mkdir(parents=True, exist_ok=True)
    for case in a.usecase:
        ds = load_dataset(filename=f'usecase{case}/case{case}_test_n2_{TEST_SIZE_LOG2}', device=a.device,
                          y_return_mode='both' if case == 2 else 'noised')
        df = get_timings(case, a.device, ds, batch_sizes=a.batch_sizes)['timing_summary']
        df.to_csv(a.out_dir / f'timing_case{case}.csv', index=False)
        print(f'Wrote {a.out_dir}/timing_case{case}.csv')


if __name__ == '__main__':
    main()
