"""Download (or upload) the released datasets, LMA results and checkpoints.

Files on the Hub mirror their path in the repository, so a download into the
repository root puts everything where the code looks for it:

  data repo  (dataset)  data/datasets/usecase<N>/*.npz      simulated datasets
                        data/labels/*.xlsx                  label mappings (dataset generation)
                        outputs/case<N>/outputs/loss_lma_<scheme>  LMA baseline results (full-dataset runs)
                        outputs/case<N>/outputs/loss_{CNN,DU,PIDU}_*  test exports of the trained runs
                        outputs/case<N>/loss/loss_{CNN,DU,PIDU}_<i>_001  loss trajectories (per-epoch train/val/test loss)
                        outputs/case<N>/logs/train_<TYPE>_<i>.log  training logs of all runs
                        models/usecase<N>/<TYPE>-t01-<i>.pt  checkpoints of all runs
  model repo (model)    models/usecase<N>/<TYPE>-t01-II.pt  released CNN / DU / PIDU checkpoint (seed II, RELEASED_SEEDS)
                        outputs/case<N>/logs/train_<TYPE>_II.log  its training log

The experimental dataset and exports with measured spectra (EXCLUDE) are never uploaded. An upload mirrors the local files: files on
the Hub that match an artifact's patterns but no longer exist locally are deleted.

    pidu-hub download                      # everything, both use cases
    pidu-hub download --what models --usecase 1    # released checkpoints only
    pidu-hub upload                        # maintainers; needs `hf auth login`
"""
import argparse
from fnmatch import fnmatch
from pathlib import Path

from pidu.paths import DATASETS_DIR, MODELS_DIR, ROOT

DATA_REPO = 'MKoumans/pidu-data'
MODEL_REPO = 'MKoumans/pidu-optical-parameter-extraction'
USECASES = (1, 2)
# kept out of uploads: the experimental dataset and every export that contains the measured SrTiO3
# spectra (fits of the experimental set are plotted in png_unfolds_exp/, and the use case 2 *_ypreds.xlsx
# hold the spectra of the last evaluated set, the experimental one)
EXCLUDE = ['*experimental*', '*png_unfolds_exp*', 'outputs/case2/*_ypreds.xlsx']

# seed of the released checkpoint per use case and model type, i.e. the training log that goes with it
RELEASED_SEEDS = {1: {'CNN': 0, 'DU': 0, 'PIDU': 0}, 2: {'CNN': 0, 'DU': 0, 'PIDU': 0}}


def released_checkpoint_name(usecase: int, model_type: str) -> str:
  """Repository path of the released checkpoint, e.g. models/usecase2/PIDU-t01-00.pt (run loss_PIDU_0_001)."""
  return f'models/usecase{usecase}/{model_type}-t01-{RELEASED_SEEDS[usecase][model_type]:02d}.pt'

# what -> (repo_id, repo_type, patterns per use case)
ARTIFACTS = {
  'data': (DATA_REPO, 'dataset', lambda uc: [f'data/datasets/usecase{uc}/*.npz', 'data/labels/*.xlsx']),
  'lma': (DATA_REPO, 'dataset', lambda uc: [f'outputs/case{uc}/outputs/loss_lma_{s}/**' for s in ('baseline', 'noisy', 'random')]),
  'runs': (DATA_REPO, 'dataset', lambda uc: [f'outputs/case{uc}/outputs/loss_{m}_*/**' for m in ('CNN', 'DU', 'PIDU')]),
  'loss': (DATA_REPO, 'dataset', lambda uc: [f'outputs/case{uc}/loss/loss_{m}_*_001/**' for m in ('CNN', 'DU', 'PIDU')]),
  'logs': (DATA_REPO, 'dataset', lambda uc: [f'outputs/case{uc}/logs/train_{m}_*.log' for m in ('CNN', 'DU', 'PIDU')]),
  'checkpoints': (DATA_REPO, 'dataset', lambda uc: [f'models/usecase{uc}/{m}-t01-*.pt' for m in ('CNN', 'DU', 'PIDU')]),
  'models': (MODEL_REPO, 'model', lambda uc: [released_checkpoint_name(uc, m) for m in RELEASED_SEEDS[uc]]
             + [f'outputs/case{uc}/logs/train_{m}_{i}.log' for m, i in RELEASED_SEEDS[uc].items()]),
}
# what -> patterns per use case of the Hub files an upload replaces (default: the artifact's own patterns)
MIRRORED = {
  'models': lambda uc: [f'models/usecase{uc}/*', f'outputs/case{uc}/logs/*'],
}


def released_checkpoint(usecase: int, model_type: str = 'PIDU', download_missing: bool = True) -> Path:
  """Local path of the released checkpoint; downloaded from the model repo if it is not there yet."""
  path = MODELS_DIR / Path(released_checkpoint_name(usecase, model_type)).relative_to('models')
  if not path.exists() and download_missing:
    from huggingface_hub import hf_hub_download
    print(f"[hub] {path.name} not found locally; downloading from {MODEL_REPO}")
    hf_hub_download(MODEL_REPO, released_checkpoint_name(usecase, model_type), local_dir=MODELS_DIR.parent)
  return path


def dataset_file(name: str, download_missing: bool = True) -> Path:
  """Local .npz of a dataset ('usecase2/case2_test_n2_10_wsn'); downloaded from the data repo if it is
  not there yet. The experimental dataset is not on the Hub, so it is never downloaded."""
  path = DATASETS_DIR / f'{name}.npz'
  repo_path = f'data/datasets/{name}.npz'
  if not path.exists() and download_missing and not any(fnmatch(repo_path, p) for p in EXCLUDE):
    from huggingface_hub import hf_hub_download
    print(f"[hub] {name} not found locally; downloading from {DATA_REPO}")
    hf_hub_download(DATA_REPO, repo_path, repo_type='dataset', local_dir=DATASETS_DIR.parent.parent)
  return path


def download(what=tuple(ARTIFACTS), usecases=USECASES, local_dir=ROOT):
  """Fetch the selected artifacts into `local_dir` (default: the repository root)."""
  from huggingface_hub import HfApi, snapshot_download
  from huggingface_hub.utils import filter_repo_objects

  for name in what:
    repo_id, repo_type, patterns = ARTIFACTS[name]
    allow = [p for uc in usecases for p in patterns(uc)]
    files = list(filter_repo_objects(HfApi().list_repo_files(repo_id, repo_type=repo_type), allow_patterns=allow))
    if not files:
      raise SystemExit(f"[hub] {repo_id} has no {name} files for use case(s) {list(usecases)}")
    snapshot_download(repo_id, repo_type=repo_type, local_dir=local_dir, allow_patterns=allow)
    print(f"[hub] {name}: {len(files)} files from {repo_id} -> {local_dir}")


def upload(what=tuple(ARTIFACTS), usecases=USECASES, local_dir=ROOT):
  """Push the selected local artifacts to the Hub (creates the repos if needed), deleting Hub files of
  those artifacts that do not exist locally (e.g. an LMA split folder of an earlier run)."""
  from huggingface_hub import HfApi

  api = HfApi()
  for name in what:
    repo_id, repo_type, patterns = ARTIFACTS[name]
    api.create_repo(repo_id, repo_type=repo_type, exist_ok=True)
    allow = [p for uc in usecases for p in patterns(uc)]
    mirrored = [p for uc in usecases for p in MIRRORED.get(name, patterns)(uc)]
    api.upload_folder(
      repo_id=repo_id, repo_type=repo_type, folder_path=local_dir,
      allow_patterns=allow, ignore_patterns=EXCLUDE, delete_patterns=mirrored,
      commit_message=f"Upload {name} (use cases {list(usecases)})",
    )
    print(f"[hub] {name}: {local_dir} -> {repo_id}")


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('action', choices=['download', 'upload'])
  parser.add_argument('--what', nargs='+', choices=list(ARTIFACTS), default=list(ARTIFACTS))
  parser.add_argument('--usecase', nargs='+', type=int, choices=USECASES, default=list(USECASES))
  args = parser.parse_args()
  (download if args.action == 'download' else upload)(args.what, args.usecase)


if __name__ == '__main__':
  main()
