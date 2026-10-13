"""
Model definitions.

  param_activations.py - primitive per-parameter activation modules
  activations.py    - shared activation-function construction/application (ParameterActivations)
  blocks.py         - generic building blocks: act_str_to_fn, ScalableTanh(Output), ResidualBlock1D, Backbone
  heads.py          - detection heads: HeadCnnModel, HeadSSDModel(+HeadSSDScaleSmall/Cell/SysNetwork)
  unfold_model.py   - BaseModel, CNNModel, SSDModel: one per-unfold, per-system detection model
  cnn_basic.py      - CNNBasic: a compact from-scratch CNN baseline
  framework.py      - MultiModelFramework: the multi-system, multi-unfold orchestrator
  io.py             - save_pt_model / load_pt_model
  common.py         - small helpers shared across the above
  experimental.py   - archived, unused architecture experiments (not imported by anything live)

Import directly from the submodule you need, e.g.:
  from pidu.model.networks.framework import MultiModelFramework
  from pidu.model.networks.io import save_pt_model, load_pt_model
"""
