import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from pidu.paths import ROOT
PROJECT_ROOT = ROOT

from pidu.data.simulation import init_usecase, load_sim_ds
from pidu.model.networks.cnn_basic import CNNBasic
from pidu.model.networks.framework import MultiModelFramework
from pidu.model.utils import create_masks_osc, create_masks_sys


def latex_escape(value: Any) -> str:
  text = str(value)
  return (
    text.replace("\\", "\\textbackslash{}")
    .replace("_", "\\_")
    .replace("%", "\\%")
    .replace("&", "\\&")
    .replace("#", "\\#")
    .replace("{", "\\{")
    .replace("}", "\\}")
  )


def shape_to_str(shape: Iterable[int]) -> str:
  return "(" + ", ".join(str(int(v)) for v in shape) + ")"


def find_dataset_name(case: int, split: str = "train", n2: Optional[int] = None) -> str:
  ds_root = PROJECT_ROOT / "data" / "datasets" / f"usecase{case}"
  if not ds_root.exists():
    raise FileNotFoundError(f"Dataset directory not found: {ds_root}")

  pattern = f"case{case}_{split}_"
  candidates = [p for p in ds_root.glob(f"{pattern}*.npz") if p.is_file()]
  if not candidates:
    candidates = [p for p in ds_root.glob(f"case{case}_*.npz") if p.is_file()]
  if not candidates:
    raise FileNotFoundError(f"No dataset files found in {ds_root}")

  if n2 is not None:
    target = f"case{case}_{split}_n2_{n2}"
    for c in candidates:
      if c.stem == target or c.stem.startswith(target + "_"):
        return f"usecase{case}/{c.stem}"

  def score(path: Path) -> Tuple[int, int]:
    stem = path.stem
    has_split = 1 if f"_{split}_" in stem else 0
    match = re.search(r"_n2_(\d+)", stem)
    power = int(match.group(1)) if match else -1
    return has_split, power

  selected = sorted(candidates, key=score, reverse=True)[0]
  return f"usecase{case}/{selected.stem}"


def normalize_dataset_name(case: int, dataset: Optional[str], split: str, n2: Optional[int]) -> str:
  if not dataset:
    return find_dataset_name(case=case, split=split, n2=n2)

  name = dataset.replace("\\", "/")
  if name.endswith(".npz"):
    name = name[:-4]

  if name.startswith("usecase"):
    return name

  return f"usecase{case}/{name}"


def build_model_info(case: int, ds_info: Dict[str, Any], dtype: torch.dtype, device: torch.device, cnn_size: float, active_models: List[str], create_wn_mask: bool) -> Dict[str, Any]:
  return {
    "physics": False,
    "backbone_act_func": "tanh",
    "backbone_norm": "instancenorm",
    "dropout_rate": 0.2,
    "cnn_size": cnn_size,
    "active_models": active_models,
    "mask_x_sys": create_masks_sys(
      usecase=case,
      n_layers=ds_info["n_layers"],
      n_sys_pars=ds_info["n_system_parameters"],
      dt=dtype,
      dv=device,
    ),
    "mask_x_osc": create_masks_osc(
      n_layers=ds_info["n_layers"],
      n_scales=ds_info["n_scales"],
      n_regions_max=ds_info["n_regions_per_scale_max"],
      n_attribs_label=ds_info["n_attribs_label"],
      std_val={"system1": 1, "system2": 1},
      dt=dtype,
      dv=device,
    ),
    "create_wn_mask": create_wn_mask,
    "use_residual": True,
    "conf_threshold": 0.75,
  }


def resolve_active_models(case: int, user_value: Optional[str]) -> List[str]:
  if user_value:
    return [x.strip() for x in user_value.split(",") if x.strip()]
  if case == 1:
    return ["system1"]
  return ["system1", "system2_0", "system2_1"]


def tabular(headers: List[str], rows: List[List[str]]) -> str:
  """Generate LaTeX tabular with borders and proper formatting."""
  cols = "|" + "|".join(["l"] * len(headers)) + "|"
  out = []
  out.append("\\begin{tabular}{" + cols + "}")
  out.append("\\hline")
  out.append(" " + " & ".join(headers) + " \\\\")
  out.append("\\hline")
  for row in rows:
    out.append(" " + " & ".join(row) + " \\\\")
  out.append("\\hline")
  out.append("\\end{tabular}")
  return "\n".join(out)


def build_cnn_table(cnn: CNNBasic, ds_info: Dict[str, Any]) -> str:
  """Build unified CNNBasic table with conv, FC, and output layers."""
  rows: List[List[str]] = []
  
  for system_name in cnn.active_systems:
    x_len = int(ds_info["x_len"][system_name])
    seq = cnn.cnn_per_system[system_name]
    
    # Conv/Pool layers
    x = torch.zeros(1, 2, x_len)
    for layer in seq:
      in_shape = shape_to_str(x.shape)
      with torch.no_grad():
        x = layer(x)
      out_shape = shape_to_str(x.shape)
      
      layer_type = layer.__class__.__name__
      details = "-"
      if isinstance(layer, nn.Conv1d):
        details = f"k={layer.kernel_size[0]}, s={layer.stride[0]}, p={layer.padding[0]}"
      elif isinstance(layer, nn.MaxPool1d):
        k = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
        s = layer.stride if isinstance(layer.stride, int) else layer.stride[0]
        details = f"k={k}, s={s}"
      elif isinstance(layer, nn.AdaptiveAvgPool1d):
        out_size = layer.output_size if isinstance(layer.output_size, int) else layer.output_size[0]
        details = f"out={out_size}"
      
      rows.append([
        layer_type,
        details,
        f"${in_shape}$",
        f"${out_shape}$",
      ])
  
  # Separator
  rows.append(["", "", "", ""])
  
  # FC layers
  for system_name in cnn.active_systems:
    fc_seq = cnn.fc_per_system[system_name]
    for layer in fc_seq:
      layer_type = layer.__class__.__name__
      details = "-"
      in_shape = "-"
      out_shape = "-"
      
      if isinstance(layer, nn.Linear):
        in_shape = f"$(1, {layer.in_features})$"
        out_shape = f"$(1, {layer.out_features})$"
      elif isinstance(layer, nn.Dropout):
        details = f"p=${layer.p}$"
      
      rows.append([
        layer_type,
        details,
        in_shape,
        out_shape,
      ])
  
  # Separator
  rows.append(["", "", "", ""])
  
  # Output heads
  for system_name in cnn.active_systems:
    sys_head = cnn.sys_output_heads[system_name] if system_name in cnn.sys_output_heads else None
    if sys_head is not None:
      rows.append([
        "Linear",
        "out\\_sys",
        f"$(1, {sys_head.in_features})$",
        f"$(1, {sys_head.out_features})$",
      ])
    
    osc_head = cnn.osc_output_heads[system_name] if system_name in cnn.osc_output_heads else None
    if osc_head is not None:
      rows.append([
        "Linear",
        "out\\_osc",
        f"$(1, {osc_head.in_features})$",
        f"$(1, {osc_head.out_features})$",
      ])
  
  table = tabular(headers=["Layer", "Details", "Input", "Output"], rows=rows)
  return f"\\subsection*{{CNNBasic}}\n\n{table}"


def build_cnn_model_table(mmf: MultiModelFramework, ds_info: Dict[str, Any]) -> str:
  """Build unified CNNModel detection head table."""
  rows: List[List[str]] = []
  
  for model_name, model_list in mmf.system_models.items():
    model_type = model_list[0].__class__.__name__ if model_list else None
    if model_type != 'CNNModel':
      continue
    
    system_name = next((k for k in mmf.active_systems if model_name.startswith(k)), model_name)
    x_len = int(ds_info["x_len"][system_name])
    
    if not model_list:
      continue
    model_unfold = model_list[0]
    head = getattr(model_unfold, 'detection_head', None)
    if head is None or head.__class__.__name__ != 'HeadCnnModel':
      continue
    
    n_features_in = head.n_features_in
    x = torch.zeros(1, n_features_in, x_len, dtype=head.dt, device=head.dv)
    
    # CNN1
    with torch.no_grad():
      x_cnn1 = head.cnn['cnn1'](x)
      x_cnn1 = head.norm['instance1'](x_cnn1)
      x_cnn1 = head.act['act1'](x_cnn1)
    
    rows.append([
      "Conv1d",
      f"k={head.cnn['cnn1'].kernel_size[0]}, s={head.cnn['cnn1'].stride[0]}, p={head.cnn['cnn1'].padding[0]}",
      f"${shape_to_str(x.shape)}$",
      f"${shape_to_str(x_cnn1.shape)}$",
    ])
    rows.append(["InstanceNorm1d", "-", f"${shape_to_str(x_cnn1.shape)}$", f"${shape_to_str(x_cnn1.shape)}$"])
    rows.append(["Activation", "-", f"${shape_to_str(x_cnn1.shape)}$", f"${shape_to_str(x_cnn1.shape)}$"])
    
    # CNN2
    with torch.no_grad():
      x_cnn2 = head.cnn['cnn2'](x_cnn1)
      x_cnn2 = head.norm['instance2'](x_cnn2)
      x_cnn2 = head.act['act2'](x_cnn2)
    
    rows.append([
      "Conv1d",
      f"k={head.cnn['cnn2'].kernel_size[0]}, s={head.cnn['cnn2'].stride[0]}, p={head.cnn['cnn2'].padding[0]}",
      f"${shape_to_str(x_cnn1.shape)}$",
      f"${shape_to_str(x_cnn2.shape)}$",
    ])
    rows.append(["InstanceNorm1d", "-", f"${shape_to_str(x_cnn2.shape)}$", f"${shape_to_str(x_cnn2.shape)}$"])
    rows.append(["Activation", "-", f"${shape_to_str(x_cnn2.shape)}$", f"${shape_to_str(x_cnn2.shape)}$"])
    
    # CNN3
    with torch.no_grad():
      x_cnn3 = head.cnn['cnn3'](x_cnn2)
      x_cnn3 = head.act['act3'](x_cnn3)
    
    rows.append([
      "Conv1d",
      f"k={head.cnn['cnn3'].kernel_size[0]}, s={head.cnn['cnn3'].stride[0]}, p={head.cnn['cnn3'].padding[0]}",
      f"${shape_to_str(x_cnn2.shape)}$",
      f"${shape_to_str(x_cnn3.shape)}$",
    ])
    rows.append(["Activation", "-", f"${shape_to_str(x_cnn3.shape)}$", f"${shape_to_str(x_cnn3.shape)}$"])
    
    # Flatten
    with torch.no_grad():
      x_flat_cnn = head.flatten(x_cnn3)
      x_flat = x_flat_cnn.flatten(1)
    
    rows.append(["", "", "", ""])
    rows.append([
      "Conv1d (Flatten)",
      "k=1, s=1, p=0",
      f"${shape_to_str(x_cnn3.shape)}$",
      f"${shape_to_str(x_flat.shape)}$",
    ])
    rows.append(["", "", "", ""])
    
    # FC layers
    for fc_name in ['fc1', 'fc2', 'fc3']:
      fc_layer = head.fc[fc_name]
      rows.append([
        "Linear",
        "-",
        f"$(1, {fc_layer.in_features})$",
        f"$(1, {fc_layer.out_features})$",
      ])
  
  if rows:
    table = tabular(headers=["Layer", "Details", "Input", "Output"], rows=rows)
    return f"\\subsection*{{CNNModel}}\n\n{table}"
  return ""


def build_ssd_model_table(mmf: MultiModelFramework, ds_info: Dict[str, Any]) -> str:
  """Build unified SSDModel detection head table."""
  rows: List[List[str]] = []
  
  for model_name, model_list in mmf.system_models.items():
    model_type = model_list[0].__class__.__name__ if model_list else None
    if model_type != 'SSDModel':
      continue
    
    system_name = next((k for k in mmf.active_systems if model_name.startswith(k)), model_name)
    x_len = int(ds_info["x_len"][system_name])
    
    if not model_list:
      continue
    model_unfold = model_list[0]
    head = getattr(model_unfold, 'detection_head', None)
    if head is None or head.__class__.__name__ != 'HeadSSDModel':
      continue
    
    scale_key = list(head.models.keys())[0]
    scale_head = head.models[scale_key]
    
    if scale_head.__class__.__name__ != 'HeadSSDScaleSmall':
      continue
    
    n_features_in = scale_head.n_features_in
    x_len_scale = scale_head.x_len
    stride = scale_head.stride
    n_regions = scale_head.n_regions
    
    rows.append([
      "RegionNorm",
      f"stride={stride}",
      f"$(1, {n_features_in}, {x_len_scale})$",
      f"$(1, {n_features_in}, {stride})$ per region",
    ])
    
    rows.append([
      "Downsample",
      "Upsample to 64",
      f"$(1, {n_features_in}, {stride})$",
      f"$(1, {n_features_in}, 64)$",
    ])
    
    rows.append([
      "CellNetwork",
      f"per region (n\\_regions={n_regions})",
      f"$(1, {n_features_in}, 64)$ + z",
      "$(1, 4)$ per region",
    ])
    
    rows.append(["", "", "", ""])
    
    rows.append([
      "SysNetwork",
      "full signal",
      f"$(1, {n_features_in}, {x_len_scale})$ + z\\_sys",
      "$(1, n\\_sys\\_pars)$",
    ])
  
  if rows:
    table = tabular(headers=["Layer", "Details", "Input", "Output"], rows=rows)
    return f"\\subsection*{{SSDModel}}\n\n{table}"
  return ""


def generate_latex(
  case: int,
  dataset_name: str,
  cnn: CNNBasic,
  mmf: MultiModelFramework,
  ds_info: Dict[str, Any],
  out_file: Path,
) -> str:
  content: List[str] = []
  content.append("% Auto-generated architecture tables")
  content.append(f"% case={case}, dataset={dataset_name}")
  
  cnn_table = build_cnn_table(cnn, ds_info)
  if cnn_table:
    content.append(cnn_table)
  
  cnn_model_table = build_cnn_model_table(mmf, ds_info)
  if cnn_model_table:
    content.append(cnn_model_table)
  
  ssd_model_table = build_ssd_model_table(mmf, ds_info)
  if ssd_model_table:
    content.append(ssd_model_table)
  
  latex_text = "\n\n".join(content) + "\n"
  out_file.parent.mkdir(parents=True, exist_ok=True)
  out_file.write_text(latex_text, encoding="utf-8")
  return latex_text


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Generate LaTeX tables for model architectures.")
  parser.add_argument("--case", type=int, choices=[1, 2], required=True, help="Usecase number (1 or 2).")
  parser.add_argument("--dataset", type=str, default=None, help="Dataset name.")
  parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test", "experimental"], help="Split.")
  parser.add_argument("--n2", type=int, default=None, help="Preferred n2 power.")
  parser.add_argument("--cnn-size", type=float, default=1.0, help="CNN width scaling factor.")
  parser.add_argument("--unfolds", type=int, default=3, help="Number of unfolds.")
  parser.add_argument("--active-models", type=str, default=None, help="Comma-separated active model list.")
  parser.add_argument("--device", type=str, default="cpu", help="Torch device.")
  parser.add_argument("--create-wn-mask", action="store_true", help="Enable wn masking.")
  parser.add_argument("--output", type=str, default=None, help="Output tex file path.")
  return parser.parse_args()


def main() -> None:
  args = parse_args()

  dataset_name = normalize_dataset_name(args.case, args.dataset, args.split, args.n2)
  print(f"Loading dataset: {dataset_name}")
  dataset = load_sim_ds(dataset_name, device=args.device)
  ds_info = dataset.get_info_for_model(print_info=False)
  device, dtype = dataset.get_device_dtype()

  p_gen_params, sys_pars = init_usecase(usecase=args.case, device=device)
  active_models = resolve_active_models(case=args.case, user_value=args.active_models)
  
  model_info = build_model_info(
    case=args.case,
    ds_info=ds_info,
    dtype=dtype,
    device=device,
    cnn_size=args.cnn_size,
    active_models=active_models,
    create_wn_mask=args.create_wn_mask,
  )

  system_info = {
    "p_gen": p_gen_params,
    "dataset_info": ds_info,
    "systems": sys_pars,
  }

  cnn = CNNBasic(
    system_info=system_info,
    model_info=model_info,
    dtype=dtype,
    device=device,
    verbose=0,
  )

  mmf = MultiModelFramework(
    n_unfolds=args.unfolds,
    system_info=system_info,
    model_info=model_info,
    dtype=dtype,
    device=device,
    verbose=0,
  )

  default_out = PROJECT_ROOT / "outputs" / "statistics" / f"architecture_case{args.case}.tex"
  out_file = Path(args.output) if args.output else default_out

  generate_latex(
    case=args.case,
    dataset_name=dataset_name,
    cnn=cnn,
    mmf=mmf,
    ds_info=ds_info,
    out_file=out_file,
  )

  print(f"Wrote LaTeX table file: {out_file}")


if __name__ == "__main__":
  main()
