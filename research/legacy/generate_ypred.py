import os
import argparse
from platform import system_alias
import sys
import pandas as pd
from pandas import DataFrame
from torch import Tensor
from torch import stack, concat

from pidu.data.simulation import load_sim_ds
from pidu.data.utils import np2pt, pt2np
from pidu.data.physics import TmMethodPyTorch

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate predicted y values from model outputs using TMM simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate-ypred.py -u 1 -n loss_DU_0_000 -s test
  python generate-ypred.py --usecase 3 --name-run experiment_01 --set val
        """
    )
    
    parser.add_argument(
        '-d', '--dir',
        type=str,
        required=False,
        help='Directory containing the Excel files to load. If not provided, it will be constructed from usecase, name-run, and set arguments.'
    )
    parser.add_argument(
        '-u', '--usecase', 
        type=str, 
        required=True, 
        help='Use case identifier (e.g., 1, 2, 3)'
    )
    parser.add_argument(
        '-n', '--name-run', 
        type=str, 
        required=True, 
        help='Name of the run to analyze (e.g., loss_DU_0_000)'
    )
    parser.add_argument(
        '-s', '--set', 
        type=str, 
        required=True, 
        choices=['train', 'val', 'test', 'exp'],
        help='Dataset type: train, validation (val), test, or experimental (exp)'
    )
    parser.add_argument(
        '-x', '--xlsx-filter',
        type=str,
        required=False,
        help='Optional filter with which the Excel file should start (e.g., "system"). If not provided, all Excel files in the directory will be loaded.'
    )
    
    return parser.parse_args()

def load_xlsx(dir: str, xlsx_filter: str=None, layers:int=1) -> dict[str, DataFrame]:
    print(f" - Loading data from {dir} ...")
    
    if dir.endswith('.xlsx'):
      dfs: dict[str, dict[str, DataFrame]] = {dir.split('\\')[-1].split('.')[0]: pd.read_excel(dir, sheet_name=None, index_col=0)}
    else:
      files = os.listdir(dir)
    
      if xlsx_filter is not None:
        files = [f for f in files if f.startswith(xlsx_filter)]
    
      dfs = {x.split('.')[0]: pd.read_excel(f"{dir}/{x}", sheet_name=None, index_col=0) for x in files}
    
    # Reshape dataframes to include layers
    for systemname, systemdata in dfs.items():
      # reshape [nx1xa] to [nxlayersxa/layers]
      for samplekey, df in systemdata.items():
        n, a = df.shape
        l = layers
        dfs[systemname][samplekey] = df.to_numpy().reshape((n, l, a//l))
    
    return dfs

def load_tmm_data(usecase: str|int, filename:str='case{usecase}_test_n2_10'):
    print(f" - Loading TMM data for usecase {usecase} ...")
    file_path = f"usecase{usecase}/{filename.format(usecase=usecase)}"  # 
    sim_ds = load_sim_ds(file_path)               # 
    tmm = sim_ds.tmm
    wn = sim_ds.wn
    
    return (tmm, wn), sim_ds

def create_y_pred(data: dict, tmm:dict[str, TmMethodPyTorch], wn:dict[str, Tensor]) -> dict:
    print(" - Creating y_pred data ...")
    y_preds = {}
    mapping = {'fit_sto_transmission_dook1': 'system1', 'paramsfile': 'system1'}
    
    for systemname, systemdata in data.items():
      
      if systemname in mapping: systemname_wn = mapping[systemname]
      else: systemname_wn = systemname
      
      for samplekey, data in systemdata.items():
        key = f"{systemname}_{samplekey}"
        params = data
        params = np2pt(params, dtype=wn[systemname_wn].dtype, device=wn[systemname_wn].device).nan_to_num(0.0)
        y, _, _ = tmm[systemname_wn].forward(
          wn=wn[systemname_wn],
          params=params,
        )
        y_preds[key] = pt2np(concat((wn[systemname_wn][None,:], y), dim=0)).T
    
    return y_preds
  
def export_xlsx(data: dict, export_path: str, filename: str, columns:list[str]|None=None):  
    print(f" - Exporting y_pred data to {export_path}/{filename}.xlsx ...")
    os.makedirs(export_path, exist_ok=True)
    with pd.ExcelWriter(f"{export_path}/{filename}.xlsx") as writer:
        for key, data_system in data.items():
            if columns is None:
                columns = [f"sample_{i:03d}" for i in range(data_system.shape[1])]
            df = pd.DataFrame(data_system, columns=columns)
            if not df.empty:
                df.to_excel(writer, sheet_name=key, index=False)

def main():
    args = parse_args()
    
    dir_import = args.dir or f"outputs/case{args.usecase}/{args.name_run}/{args.set}"
    
    # 1. Load existing data from Excel files
    try:
        data = load_xlsx(dir_import, xlsx_filter=args.xlsx_filter)
    except Exception as e:
        print(f"Error loading data from {dir_import}: {e}")
        return
    
    # 3. Load TMM data
    try:
        (tmm, wn), sim_ds = load_tmm_data(usecase=args.usecase)
    except Exception as e:
        print(f"Error loading TMM data for usecase {args.usecase}: {e}")
        return
    
    # 4. Create y_pred data
    y = create_y_pred(data, tmm, wn)
    
    # 5. Export to Excel
    export_xlsx(
      y, 
      f"outputs/case{args.usecase}/{args.name_run}/{args.set}", 
      f"u{args.usecase}_{args.set}_{args.name_run}",
      # ['wn', 'y_true', 'y_init', 'U1', 'U2', 'U3']
    )
  
if __name__ == "__main__":
    main()