"""Global use-case selection.

`usecase` is read at import time by `pidu.data.simulation` (module-level
parameter ranges, structured-noise branch) and by the LMA baseline. Set it with
the `PIDU_USECASE` environment variable (default 2) before importing `pidu`.
"""
import os

usecase = int(os.environ.get('PIDU_USECASE', 2))

if usecase == 1:
  active_models = ['system1'] # , 'system2_0', 'system2_1'
elif usecase == 2:
  active_models = ['system1', 'system2_0', 'system2_1']
elif usecase == 3:
  active_models = ['system1', 'system2_0', 'system2_1']
else:
  raise ValueError(f"PIDU_USECASE must be 1, 2 or 3, got {usecase}")
