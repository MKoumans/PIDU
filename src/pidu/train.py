"""Train PIDU, DU or CNN on one use case (entry point `pidu-train`, or `python -m pidu.train`).

Creates or loads the train/val/test (and experimental) datasets, trains the
model, saves the checkpoint to models/usecase<N>/, and exports the val/test
predictions to outputs/case<N>/outputs/<output-name>/.
"""
import os
import argparse
import numpy as np

import yaml
import torch

from pidu import config
from pidu.config import usecase, active_models
from pidu.paths import ROOT, DATASETS_DIR, MODELS_DIR, case_dir
from pidu.data.utils import pt2np, np2pt, get_matches, savefig_debug
from pidu.data.simulation import load_sim_ds, init_usecase, create_dataset

from pidu.model.networks.framework import MultiModelFramework
from pidu.model.networks.cnn_basic import CNNBasic
from pidu.model.networks.io import load_checkpoint, load_pt_model, save_pt_model
from pidu.model.trainer import Trainer
from pidu.model.utils import create_masks_osc, create_masks_sys, print_memory_usage, save_model_by_trail

PROJECT_DIR = str(ROOT)

torch.set_printoptions(precision=2, sci_mode=False)

MODEL_REGISTRY = {
  # NAME: (MODEL_CLASS, USE_PHYSICS)
  'PIDU': (MultiModelFramework, True),
  'DU':   (MultiModelFramework, False),
  'CNN':  (CNNBasic, False),
}


def get_dataset(
  usecase: int,
  trainset: str|int|None=None,
  valset: str|int|None=None,
  testset: str|int|None=None,
  expset: str|None|bool=None,
  **kwargs
):
  """Load or generate the train/val/test (and optional experimental) datasets for a use case.

  For each of `trainset`, `valset`, `testset`: a string is treated as a dataset name to
  load via `load_sim_ds`, an int overrides the corresponding sample-count exponent
  (`n_train`/`n_val`/`n_test`) and triggers dataset generation via `create_dataset`, and
  `None` generates a dataset using the default size. `expset` follows a similar pattern,
  except `False` (or `None` when not otherwise specified) disables the experimental set,
  a string loads it by name, and any other value generates it.

  Args:
    usecase: Use case number (1, 2, or 3) that determines physical system parameters.
    trainset: Dataset name to load, sample-count exponent to generate, or None for default size.
    valset: Dataset name to load, sample-count exponent to generate, or None for default size.
    testset: Dataset name to load, sample-count exponent to generate, or None for default size.
    expset: Dataset name to load, False/None to skip, or any other value to generate the
      experimental dataset.
    **kwargs: n_train (int, default 12), n_val (int, default 8), n_test (int, default 10)
      sample-count exponents (dataset size is 2**n); device (torch.device) to place datasets
      on; verbose (bool, default True); no_structured_noise (bool, default False) to disable
      structured noise injection for use cases 2 and 3.

  Returns:
    A 4-tuple of:
      - dict mapping 'train'/'val'/'test' (and 'exp' if generated/loaded) to
        `SimulationDatasetBatch` instances.
      - dataset_info dict describing dataset structure (from `get_info_for_model`).
      - device the datasets are placed on.
      - dtype of the dataset tensors.

  Raises:
    AssertionError: If the requested device does not match the datasets' device, or if the
      train/val/test/exp datasets are not all on the same device.
  """
  n_train = kwargs.get('n_train', 12)
  n_val = kwargs.get('n_val', 8)
  n_test = kwargs.get('n_test', 10)
  device = kwargs.get('device')
  p_gen_params, sys_pars = init_usecase(usecase=usecase, device=device)
  verbose = kwargs.get('verbose', True)
  no_structured_noise = kwargs.get('no_structured_noise', False)

  structured_noise = [False] * len(sys_pars)
  if not no_structured_noise:
    if usecase == 2:
      structured_noise = [False, True]
    elif usecase == 3:
      structured_noise = [True, True]

  if isinstance(trainset, str):
    trainset = load_sim_ds(trainset, y_return_mode='noised', device=device)
  else:
    if isinstance(trainset, int):
      n_train = trainset
    trainset = create_dataset(
      usecase=usecase,
      setname=f"case{usecase}_train_n2_{n_train}",
      n_samples=2**n_train,
      system_params=sys_pars,
      p_gen_params=p_gen_params,
      verbose=verbose,
      export=True,
      structured_noise=structured_noise,
      device=device,
    )

  if isinstance(valset, str):
    valset = load_sim_ds(valset, y_return_mode='noised', device=device)
  else:
    if isinstance(valset, int):
      n_val = valset
    valset = create_dataset(
      usecase=usecase,
      setname=f"case{usecase}_val_n2_{n_val}",
      n_samples=2**n_val,
      system_params=sys_pars,
      p_gen_params=p_gen_params,
      verbose=verbose,
      export=True,
      structured_noise=structured_noise,
      device=device,
    )

  if isinstance(testset, str):
    testset = load_sim_ds(testset, device=device)
  else:
    if isinstance(testset, int):
      n_test = testset
    testset = create_dataset(
      usecase=usecase,
      setname=f"case{usecase}_test_n2_{n_test}",
      n_samples=2**n_test,
      system_params=sys_pars,
      p_gen_params=p_gen_params,
      verbose=verbose,
      export=True,
      structured_noise=structured_noise,
      device=device,
    )

  if isinstance(expset, str):
    expset = load_sim_ds(expset, device=device)
  elif expset is False:
    expset = None
  else:
    expset = create_dataset(
      usecase=usecase,
      setname=f"case{usecase}_experimental",
      n_samples=2**n_test,
      system_params=sys_pars,
      p_gen_params=p_gen_params,
      verbose=verbose,
      export=True,
      structured_noise=structured_noise,
      device=device,
    )

  _device, dtype = trainset.get_device_dtype()
  dataset_info = trainset.get_info_for_model()

  assert device == _device, f"Device mismatch between datasets and specified device: {_device} vs {device}"
  assert trainset.device == valset.device == testset.device, "Datasets are not on the same device."

  if expset is not None:
    assert trainset.device == expset.device, "Datasets are not on the same device."

  to_return = {'train': trainset, 'val': valset, 'test': testset}
  if expset is not None:
    to_return['exp'] = expset

  return to_return, dataset_info, device, dtype


def get_model(
  model_name:str|None=None,
  model_type:str|None=None,
  ds_info=None,
  device=None,
  dtype=None,
  usecase:int=usecase,
  **kwargs
):
    """Load an existing model checkpoint or construct a new model from `MODEL_REGISTRY`.

    If `model_name` is given, the corresponding saved model is loaded via `load_pt_model`
    and all other arguments are ignored. Otherwise a new model is built from `model_type`
    (looked up in `MODEL_REGISTRY` to determine the model class and whether the physics
    layer is used), configured with backbone, masking, and unfold settings, and summarized.

    Args:
      model_name: Name of a previously saved model to load. If provided, takes precedence
        over `model_type` and no new model is constructed.
      model_type: Key into `MODEL_REGISTRY` (e.g. 'PIDU', 'DU', 'CNN') selecting the model
        class and whether the differentiable physics layer is used.
      ds_info: Dataset info dict (as returned by `get_dataset`) describing layer/scale/region
        counts, used to build oscillator and system masks.
      device: Device to place the model on.
      dtype: Dtype for model parameters and masks.
      usecase: Use case number (default: `pidu.config.usecase`).
      **kwargs: cnn_size (float, default 1.0) width scale factor for CNNBasic; active_models
        (list[str], default ['system1']) list of active sub-systems; unfolds (int, default 3)
        number of unfold iterations.

    Returns:
      The constructed (or loaded) model instance, an instance of one of the classes in
      `MODEL_REGISTRY`.

    Raises:
      ValueError: If `model_type` is not a key in `MODEL_REGISTRY`.
    """
    if model_name is not None:
      return load_pt_model(model_name)

    p_gen_params, sys_pars = init_usecase(usecase=usecase, device=device)

    if model_type not in MODEL_REGISTRY:
      raise ValueError(f"Unknown model_type: {model_type}")
    
    model, use_physics = MODEL_REGISTRY[model_type]

    model_info = {
      'physics': use_physics,
      'backbone_act_func': 'tanh',
      'backbone_norm': 'instancenorm', # batchnorm|layernorm|instancenorm|groupnorm|none
      'dropout_rate': 0.2,
      'cnn_size': kwargs.get('cnn_size', 1.0),
      'active_models': kwargs.get('active_models', ['system1']),
      'mask_x_sys': create_masks_sys(
        usecase=usecase,
        n_layers=ds_info['n_layers'],
        n_sys_pars=ds_info['n_system_parameters'],
        dt=dtype,
        dv=device
      ),
      'mask_x_osc': create_masks_osc(
        n_layers=ds_info['n_layers'],
        n_scales=ds_info['n_scales'],
        n_regions_max=ds_info['n_regions_per_scale_max'],
        n_attribs_label=ds_info['n_attribs_label'],
        std_val={'system1': 1, 'system2': 1},
        dt=dtype,
        dv=device
      ),
      'create_wn_mask': True,
      'use_residual': True,
      'conf_threshold': 0.75,
    }

    system_info = {
      'p_gen': p_gen_params,
      'dataset_info': ds_info,
      'systems': sys_pars
    }

    framework = model(
      n_unfolds=kwargs.get('unfolds', 3),
      system_info=system_info,
      model_info=model_info,
      dtype=dtype,
      device=device,
      verbose=0,
    )
    framework.summary()

    return framework


def perform_training(
  usecase:int,
  model_type:str,
  outputname:str,
  headless:bool=False,
  *,
  n_epochs:int,
  datasets:dict,
  dataset_info:dict,
  training_system:dict,
  active_models:list[str],
  **kwargs
) -> tuple[torch.nn.Module, dict]:
  """Build (or load) a model, train it, save the result, and evaluate on val/test/exp sets.

  Args:
    usecase: Use case number (1, 2, or 3), used for output paths and model saving.
    model_type: Key into `MODEL_REGISTRY` selecting the model architecture (passed to
      `get_model`).
    outputname: Base filename used for the training history/loss output.
    headless: If True, disables intermediate plotting/saving (for headless HPC runs).
    n_epochs: Number of training epochs.
    datasets: Dict of datasets as returned by `get_dataset`.
    dataset_info: Dataset info dict as returned by `get_dataset`.
    training_system: Trainer settings as returned by `build_training_system`.
    active_models: Active sub-systems of the use case.
    **kwargs: model_name (str|None) to load an existing model instead of building a new one;
      device, dtype for model/tensor placement; cnn_size (float, default 1.0); unfolds (int,
      default 3); save_idx (list[int]) epoch indices to save intermediate results for;
      save_trail_id (int|None) trail identifier used for output naming and model saving;
      model_id (int|None) explicit model id for deterministic filenames; exp_ds (bool) whether
      to include the experimental dataset in the final test evaluation.

  Returns:
    A tuple of the trained model and the training `history` object (as returned by
    `Trainer.train`).
  """
  print(print_memory_usage(verbose=1))
  model = get_model(
    model_name=kwargs.get('model_name', None),
    model_type=model_type,
    ds_info=dataset_info,
    device=kwargs.get('device'),
    dtype=kwargs.get('dtype'),
    usecase=usecase,
    active_models=active_models,
    cnn_size=kwargs.get('cnn_size', 1.0),
    unfolds=kwargs.get('unfolds', 3),
  )

  print(print_memory_usage(verbose=1))
  train_config = {
    'save_epoch': np.arange(0, n_epochs, 10),
    'save_idx': kwargs.get('save_idx', []),
    'save_model': None,
    'plot_intermediate': True,
    'plot_dir': str(case_dir(usecase) / "intermediate"),
    'history_save_intermediate': True,
    'outputname_history': outputname,
    'save_trail_id': kwargs.get('save_trail_id', None),
  }

  if headless:
    train_config.update({
      'save_epoch': [],
      'save_idx': [],
      'plot_intermediate': False,
    })

  trainer = Trainer(model, training_system=training_system, loss_matrix=True)
  trainer.load_dataset(
    to_import=datasets,
  )

  history = trainer.train(n_epochs, **train_config)

  history.info(
    plt_loss=True,
    show_fig=False,
    plt_grads=True,
    save_fig=True,
    yscale='log',
  )

  trail = kwargs.get('save_trail_id', 0)
  trail = 0 if trail is None else int(trail)
  model_to_save = trainer.base_model if hasattr(trainer, 'base_model') else model
  save_path = save_model_by_trail(
    model=model_to_save,
    usecase=usecase,
    model_type=model_type,
    trail=trail,
    model_id=kwargs.get('model_id', None),
  )
  print(f"Saved model to: {save_path}")

  trainer.test(['val','test'] + (['exp'] if kwargs.get('exp_ds', False) else []), to_export=True)

  return model, history


def perform_evaluation(
  usecase:int,
  model_type:str,
  outputname:str,
  *,
  n_epochs:int,
  datasets:dict,
  training_system:dict,
  trail:int,
  model_id:int,
  **kwargs
) -> torch.nn.Module:
  """Load a saved checkpoint and export its val/test (and exp) predictions, as training does.

  Loads models/usecase<N>/<model_type>-t<trail>-<model_id>.pt and writes the same
  outputs/case<N>/outputs/<outputname>_<trail>/ folder as `perform_training`, so released
  checkpoints can stand in for training runs.

  Args:
    n_epochs: Epochs of the original run (sets the loss schedule as at the end of training).
    datasets: Dict of datasets as returned by `get_dataset`; the train set is not used.
    **kwargs: device; exp_ds (bool) whether to include the experimental dataset.
  """
  path = MODELS_DIR / f"usecase{usecase}" / f"{model_type}-t{trail:02d}-{model_id:02d}.pt"
  print(f"Evaluating checkpoint: {path}")
  model = load_checkpoint(path, map_location=kwargs.get('device'))

  # init_mode would re-initialise the loaded weights
  trainer = Trainer(model, training_system={**training_system, 'init_mode': None}, loss_matrix=True)
  trainer.load_dataset(to_import={k: v for k, v in datasets.items() if k != 'train'})
  trainer.init_evaluation(
    n_epochs,
    history_save_intermediate=True,
    outputname_history=outputname,
    save_trail_id=trail,
  )
  trainer.test(['val','test'] + (['exp'] if kwargs.get('exp_ds', False) else []), to_export=True)

  return model


def load_config(config_path:str) -> dict:
  """Load a YAML run config, normalising hyphenated keys to underscores.

  Args:
    config_path: Path to the YAML file; relative paths are resolved against the repo root.

  Returns:
    Dict of config values with keys normalised from hyphens to underscores.
  """
  if not os.path.isabs(config_path):
    config_path = os.path.join(PROJECT_DIR, config_path)
  with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f) or {}
  return {k.replace('-', '_'): v for k, v in cfg.items()}


# Trainer settings and their defaults, used when a key is absent from the config/CLI.
TRAINING_SYSTEM_DEFAULTS = {
  'wgn': 0.0,
  'verbose': 1,
  'lr': 1e-4,
  'scheduler_type': 'StepLR', # StepLR|CosineAnnealingWR|none
  'scheduler_step_size1': 10,
  'scheduler_gamma1': 0.95,
  'optimisation_mode': 'scale-optics-type',
  'optimiser': 'adamw',
  'optimiser_algo': 'base',
  'betas': (0.9, 0.999),
  'eps': 1e-4,
  'weight_decay': 1e0,
  'clip_grad_norm': 1.0,
  'init_mode': 'xavier', # see pidu.model.initializer; --initialiser sets it from the CLI
  'batch_size': 256,
  'shuffle': True,
  'track_gradient': False,
  'loss_alpha': 0.05,
  'mixed_precision': False,
}


def build_training_system(cfg:dict, run_folder:str) -> dict:
  """Build the Trainer `training_system` dict from a config dict.

  Args:
    cfg: Config values (e.g. from `load_config` or `vars(parse_args())`); any key
      missing or None falls back to `TRAINING_SYSTEM_DEFAULTS`.
    run_folder: Output folder for the run.

  Returns:
    The `training_system` dict expected by `model.Trainers.Trainer`.
  """
  def get(key):
    val = cfg.get(key)
    return TRAINING_SYSTEM_DEFAULTS[key] if val is None else val

  return {
    'wgn_factor': float(get('wgn')),
    'verbose': get('verbose'),
    'lr': float(get('lr')),
    'scheduler_type': get('scheduler_type'),
    'scheduler_step_size1': get('scheduler_step_size1'),
    'scheduler_gamma1': get('scheduler_gamma1'),
    'optimisation_mode': get('optimisation_mode'),
    'optimiser': get('optimiser'),
    'optimiser_algo': get('optimiser_algo'),
    'betas': tuple(get('betas')),
    'eps': float(get('eps')),
    'weight_decay': float(get('weight_decay')),
    'clip_grad_norm': get('clip_grad_norm'),
    'init_mode': cfg.get('initialiser') or get('init_mode'),
    'batch_size': get('batch_size'),
    'shuffle': get('shuffle'),
    'track_gradient': get('track_gradient'),
    'run_folder': run_folder,
    'loss_alpha': get('loss_alpha'),
    'mixed_precision': get('mixed_precision'),
  }


def parse_args():
  """Parse command-line arguments, optionally seeded with defaults from a YAML config file.

  A `--config` argument is pre-parsed first; if given, its YAML contents (with keys
  normalized from hyphens to underscores) override the hardcoded defaults, and any
  explicit CLI arguments take final precedence over both.

  Returns:
    An `argparse.Namespace` with all parsed CLI arguments (usecase, epochs, model, device,
    dataset settings, optimisation hyperparameters, and output/trail identifiers).
  """
  pre_parser = argparse.ArgumentParser(add_help=False)
  pre_parser.add_argument('--config', type=str, default=None, help='Path to a YAML config file (CLI args override config values)')
  pre_args, _ = pre_parser.parse_known_args()

  config_defaults = {}
  if pre_args.config is not None:
    config_defaults = load_config(pre_args.config)

  defaults = {
    'usecase': usecase,
    'epochs': 100,
    'model': 'PIDU',
    'debug': False,
    'verbose': 1,
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'mixed_precision': False,
    'headless': False,
    'load_ds': 'usecase{usecase}/case{usecase}_{ds_name}_n2_{number}',
    'ds_sizes': [14, 10, 10],
    'exp_ds': False,
    'nwsn': False,
    'wgn': 0.000,
    'unfolds': 3,
    'optimisation_mode': 'scale-optics-type',
    'optimiser_algo': 'base',
    'weight_decay': 1e0,
    'run_folder': None,
    'lr': 1e-4,
    'cnn_size': 1.0,
    'initialiser': 'xavier',
    'save_idx': [],
    'output_name': 'loss',
    'trail_id': None,
    'model_id': None,
    'seed': 199,
    'evaluate': False,
  }
  defaults.update(config_defaults)

  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str, default=None, help='Path to a YAML config file (CLI args override config values)')
  parser.add_argument('--usecase', type=int, help='Usecase number (1|2|3)')
  parser.add_argument('--epochs', type=int, help='Number of training epochs')
  parser.add_argument('--model', type=str, help='Modeltype to use (PIDU|DU|CNN)')
  parser.add_argument('--debug', action='store_true', help='Enable debug mode')
  parser.add_argument('--verbose', type=int, help='Verbosity level (0|1|2)')
  parser.add_argument('--device', type=str, help='Device to use for training (cpu|cuda)')
  parser.add_argument('--mixed-precision', action='store_true', help='Enable automatic mixed precision (AMP) during training')
  parser.add_argument('--sm', dest='headless', action='store_true', help='Headless mode for cluster runs: no intermediate plots/saves')

  parser.add_argument('--load-ds', type=str, help='Path to load dataset from (if any)')
  parser.add_argument('--ds-sizes', type=int, nargs=3, metavar=('N_TRAIN', 'N_VAL', 'N_TEST'), help='Sizes of the datasets as powers of 2 (e.g., 14 means 2^14 samples for training set)')
  parser.add_argument('--exp-ds', action='store_true', help='Include experimental dataset')
  parser.add_argument('--nwsn', action='store_true', help='Use noised datasets without structured noise')
  parser.add_argument('--wgn', type=float, help='Std of extra white Gaussian noise added to the input spectra during training')

  # Hyperparameters
  parser.add_argument('--unfolds', type=int, help='Number of unfolds to use in the model')
  parser.add_argument('--optimisation-mode', type=str, help='Optimisation mode to use (complete|unfold|scale|optics-type|system|scale-optics-type|none)')
  parser.add_argument('--optimiser-algo', type=str, help='Optimiser algorithm to use (base|pcgrad|nash)')
  parser.add_argument('--weight-decay', type=float, help='Weight decay for the optimizer')
  parser.add_argument('--run-folder', type=str, help='Override the run folder path')
  parser.add_argument('--lr', type=float, help='Learning rate')
  parser.add_argument('--cnn-size', type=float, help='Size factor for CNNBasic widths (1.0 keeps the current architecture)')
  parser.add_argument('--initialiser', type=str, help='Initialiser to use:\n kaiming|xavier|orthogonal|normal|delta_orthogonal|fixup|sparse|spectral|variance_scaling|midas)')

  parser.add_argument('--save_idx', type=int, nargs='+', help='Indices of epochs to save intermediate results for')
  parser.add_argument('--output-name', type=str, help='Name of the output file')
  parser.add_argument('--trail-id', type=int, help='ID for the training trail')
  parser.add_argument('--model-id', type=int, help='Model run index to make model filenames deterministic in parallel runs')
  parser.add_argument('--seed', type=int, help='Random seed for python/numpy/torch to make runs reproducible')
  parser.add_argument('--evaluate', action='store_true', help='Skip training: load models/usecase<N>/<MODEL>-t<trail-id>-<model-id>.pt and export its val/test predictions')

  parser.set_defaults(**defaults)
  args = parser.parse_args()
  return args


def assert_args(args):
  """Validate parsed CLI arguments for mutually incompatible combinations.

  Args:
    args: Parsed arguments namespace as returned by `parse_args`.

  Raises:
    AssertionError: If `optimisation_mode` is 'complete' (single-objective) while
      `optimiser_algo` is a multi-objective algorithm ('pcgrad' or 'nash').
  """
  assert not (args.optimisation_mode == 'complete' and args.optimiser_algo in ['pcgrad', 'nash']), f"Optimisation mode 'complete' returns single objective, cannot be used with multi-objective optimiser '{args.optimiser_algo}'"


def seed_everything(seed):
  """Seed python, numpy and torch (CPU and CUDA) RNGs for reproducible runs.

  Note: full bit-exact reproducibility across different GPUs/driver versions is
  not guaranteed; cuDNN is kept in its default (non-deterministic) mode for speed.
  """
  import random
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


def main():
    """CLI entry point (see `parse_args`)."""
    args = parse_args()
    assert_args(args)

    if args.seed is not None:
      seed_everything(args.seed)
      print(f"Seeded RNGs with seed: {args.seed}")

    if args.debug:
        print("Debug mode enabled.")
        print(f"Arguments: {args}")

    if args.config is None:
      assert config.usecase == args.usecase, f"Usecase mismatch between PIDU_USECASE (pidu.config) and command-line argument: {config.usecase} vs {args.usecase}"

    usecase = args.usecase
    if usecase == 1:
      active_models = ['system1']
    elif usecase in (2, 3):
      active_models = ['system1', 'system2_0', 'system2_1']
    
    n_epochs = args.epochs
    device = torch.device(args.device)
    outputname = args.output_name
    headless = args.headless
    optimisation_mode = args.optimisation_mode
    optimiser_algo = args.optimiser_algo
    weight_decay = args.weight_decay
    cnn_size = args.cnn_size
    model_type = args.model
    run_folder = args.run_folder if args.run_folder is not None else (case_dir(usecase) / 'runs').as_posix()
    verbose = args.verbose

    training_system = build_training_system(vars(args), run_folder=run_folder)

    ds_names = ['train', 'val', 'test']
    ds_set_names = [
      args.load_ds.format(usecase=usecase, ds_name=name, number=args.ds_sizes[i])
      if args.load_ds is not None else None
      for i, name in enumerate(ds_names)
    ]
    if args.nwsn:
      ds_set_names = [f"{n}_nwsn" if n is not None else None for n in ds_set_names]
    trainset_name, valset_name, testset_name = ds_set_names

    expset_name = False
    if args.exp_ds and usecase == 2:
      expset_name = f'usecase{usecase}/case{usecase}_experimental'
      if not (DATASETS_DIR / f'{expset_name}.npz').is_file():
        print(f"Experimental dataset {expset_name}.npz not found (available on request); continuing without it.")
        expset_name = False

    print(f"Loading datasets:\n Trainset: {trainset_name}\n   Valset: {valset_name}\n  Testset: {testset_name}\n  Exp. set: {expset_name}")
    print(f"Model: {model_type}, Usecase: {usecase}, Device: {device}, Epochs: {n_epochs}")

    print(print_memory_usage(verbose=1))
    datasets, dataset_info, device, dtype = get_dataset(
      usecase=usecase,
      trainset=trainset_name,
      valset=valset_name,
      testset=testset_name,
      expset=expset_name,
      device=device,
      no_structured_noise=args.nwsn,
      )

    if args.evaluate:
      assert args.trail_id is not None and args.model_id is not None, "--evaluate needs --trail-id and --model-id"
      perform_evaluation(
        usecase,
        model_type,
        outputname,
        n_epochs=n_epochs,
        datasets=datasets,
        training_system=training_system,
        trail=args.trail_id,
        model_id=args.model_id,
        device=device,
        exp_ds=bool(expset_name),
      )
      return

    _, _ = perform_training(
      usecase,
      model_type,
      outputname,
      headless=headless,
      n_epochs=n_epochs,
      datasets=datasets,
      dataset_info=dataset_info,
      device=device,
      dtype=dtype,
      save_idx=args.save_idx,
      save_trail_id=args.trail_id,
      model_id=args.model_id,
      exp_ds=bool(expset_name),
      cnn_size=cnn_size,
      unfolds=args.unfolds,
      training_system=training_system,
      active_models=active_models,
    )


if __name__ == '__main__':
    main()
