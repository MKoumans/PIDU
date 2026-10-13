"""Small helpers shared by CNNBasic and MultiModelFramework."""
from typing import Any, Dict


def select_dict_from_subset(dict_: Dict[str, Any], model_name: str) -> Any | None:
  """
  Select the entry of `dict_` whose key is a prefix of `model_name`.

  Example:
    dict_.keys() = ['system1', 'system2']
    model_name = 'system1_modelA'
    returns: dict_['system1']
  """
  option = [v for k, v in dict_.items() if model_name.startswith(k)]
  return option[0] if option else None
