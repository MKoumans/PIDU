from typing import List, Dict
import os
from matplotlib.pylab import f
import torch
import numpy as np
import matplotlib.pyplot as plt
from pidu.data.physics import TmMethodPyTorch
from pidu.paths import EXPERIMENTAL_DIR

REFFIT_DIR = str(EXPERIMENTAL_DIR / 'STO' / 'reffit_fit_files')

thickness = 0.00275 # in micrometers

def parse_asc(
  filename: str, 
  filepath: str, 
  theta:float=0.0, 
  thickness:float=thickness,
  osc_before_idx:int=-1,
  osc_after_idx:int=-1,
) -> List[float]:
  """
  Parse ASC files from fitting software and extract oscillator parameters.
  
  Args:
      filename: Name of the file to parse
      filepath: Path to the file
      theta: Angle of incidence in radians (default: 0.0)
      thickness: Film thickness in micrometers (default: thickness constant)
      
  Returns:
      List containing [theta, thickness, eps_inf, 0, 0, Wo0, Wp0, G0, Wo1, Wp1, G1, ...]
  """
  full_path = os.path.join(filepath, filename)
  
  # Initialize parameters
  eps_inf = 1.0
  oscillator_params = []
  
  # -- Read File --
  encodings_to_try = ['latin-1', 'cp1252', 'ISO-8859-1']
  content = None
  
  for encoding in encodings_to_try:
    try:
      with open(full_path, 'r', encoding=encoding) as file:
        content = file.readlines()
      break  # If successful, exit the loop
    except UnicodeDecodeError:
      continue
  
  if content is None:
    print(f"Could not decode file {full_path} with any of the attempted encodings")
    return [theta, thickness, 1.0, 0.0, 0.0]
      
  # -- Parse File --
  for i, line in enumerate(content):
    line = line.strip()
    
    # Extract epsilon infinity
    if line.startswith("Einf ="):
      parts = line.split("±")[0].strip() if "±" in line else line.split("=")[0].strip()
      try:
          eps_inf = float(parts.split("=")[1].strip())
      except (IndexError, ValueError):
          print(f"Failed to parse Einf value from line: {line}")
    
    # Extract oscillator parameters
    if line.startswith("Wo["):
      try:
          wo_value = float(line.split("±")[0].strip().split("=")[1].strip()) if "±" in line else float(line.split("=")[1].strip())
          
          # Get the next two lines for Wp and G
          if i+1 < len(content) and i+2 < len(content):
              wp_line = content[i+1].strip()
              g_line = content[i+2].strip()
              
              if wp_line.startswith("Wp[") and g_line.startswith("G ["):
                  wp_value = float(wp_line.split("±")[0].strip().split("=")[1].strip()) if "±" in wp_line else float(wp_line.split("=")[1].strip())
                  g_value = float(g_line.split("±")[0].strip().split("=")[1].strip()) if "±" in g_line else float(g_line.split("=")[1].strip())
                  
                  oscillator_params.extend([wo_value, wp_value, g_value])
      except (IndexError, ValueError) as e:
          print(f"Failed to parse oscillator parameters from line: {line}, error: {e}")
  
  # Construct the final parameter list
  osc_before = [0,0,0]
  osc_after = [0,0,0]
  osc_cells = []
  osc_drude = [0,0]
  for i in range(0, len(oscillator_params), 3):
      # w0_value = oscillator_params[i]
      if i == osc_before_idx:
        osc_before = oscillator_params[i:i+3]
      elif i == osc_after_idx:
        osc_after = oscillator_params[i:i+3]
      else:
        osc_cells.extend(oscillator_params[i:i+3])
    
  params = [theta, thickness, eps_inf] + osc_drude + osc_before + osc_after + osc_cells
  return np.array(params)

wn = torch.linspace(0.1, 90, 500, device='cuda')

files = os.listdir(REFFIT_DIR)
files = [f for f in files if f.endswith('.asc')]
tmm = TmMethodPyTorch()

for i_f, filename in enumerate(files):
  params1 = parse_asc(filename, REFFIT_DIR, osc_before_idx=0, osc_after_idx=1)

  result = tmm(params1[None,None,:],wn)

  # fig, ax = plt.subplots(3,1, figsize=(10,5))
  # ax[0].plot(wn.cpu(), result['R'].cpu().squeeze(), label='R')
  # ax[1].plot(wn.cpu(), result['t_fd'].real.cpu().squeeze(), label='t(real)')
  # ax[1].plot(wn.cpu(), result['t_fd'].imag.cpu().squeeze(), label='t(imag)')

  t_real = np.column_stack((wn.cpu(), result['t_fd'].real.cpu().squeeze(), result['t_fd'].real.cpu().squeeze()*0.05)) # 
  t_imag = np.column_stack((wn.cpu(), result['t_fd'].imag.cpu().squeeze(), result['t_fd'].imag.cpu().squeeze()*0.05))
  np.savetxt(f'{REFFIT_DIR}/t_real_{i_f+1:02d}.dat', t_real, header='Wavenumber (1/cm)    t_real    error', comments='')
  np.savetxt(f'{REFFIT_DIR}/t_imag_{i_f+1:02d}.dat', t_imag, header='Wavenumber (1/cm)    t_imag    error', comments='')