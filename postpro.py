# writing the postpro script
import numpy as np
from utilities.postpro_utilities import mass_writer, coordinates_writer, dof_writer, arvd_writer, read_pickled_file, plate_connection, stress_calculator
from torch_geometric.data import Data   
import argparse
from pathlib import Path
import torch
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()  
parser.add_argument('--par_dir', type=str, default='/Users/fammasmaz/Downloads/')
parser.add_argument('--stem', type=str, default='sncf_random_test_2') 
args = parser.parse_args()




def node_creator(out_dir):
    # dof x(3), v(3), a (6)
    # coord x, y, z
    # arvd 1
    # mass 1

    coordinates_writer(out_dir)
    dof_writer(out_dir)
    arvd_writer(out_dir)
    mass_writer(out_dir)
    dof = read_pickled_file(out_dir, 'dof.pkl')
    coord = read_pickled_file(out_dir, 'coordinates.pkl')
    arvd = read_pickled_file(out_dir, 'arvd.pkl')
    mass = read_pickled_file(out_dir, 'mass.pkl')
    x = np.hstack((dof, coord, arvd, mass))
    x = x.astype(np.float32)
    return x

def edge_index_creator(out_dir):
  # information on what is what
  # 0: contact id
  # 1: contact candidate
  # 2: contact antagoniste
  # 3-5: rl
  # 6-8: vl
  # 9: gapTT
  # 10-12: coordinates
  # 13-15: local frame n
  # 16-18: local frame t
  # 19-21: local frame s
  # 22: contact type
  # 23-25: intercenter vectors
  postpro = read_pickled_file(out_dir, 'postpro.pkl', False)
  edge_features = postpro[3]
  edge_index = np.array(edge_features[:,1:3]).T
  edge_index = edge_index - 1
  edge_features = edge_features[:,3:]

  return edge_index.astype(np.float32), edge_features.astype(np.float32)

def hot_vector_plate_conection(out_dir, n_nodes):
   plated = plate_connection(out_dir)
   x = np.zeros((n_nodes, 1))
   x[plated-1] = 1
   x[0] = 1
   return x.astype(np.float32)


def gravitational_force_creator(m, g=9.8):
    f = np.zeros((m.shape[0], 3))
    # first one is rigid plate
    f[1:, 2] = -m[1:, 0]*g
    return f
def stress_calculation(edge_features):
   reaction_forces = edge_features[:, 0:3]
   intercenter_vectors = edge_features[:, -3:]
   psi = stress_calculator(reaction_forces, intercenter_vectors)
   return np.array(psi).astype(np.float32)
   
   


# find the names of the files that start with 'sncf'

par_dir = Path(args.par_dir)
files_posix = [f for f in par_dir.iterdir() if f.stem.startswith(args.stem)]
# in string format
files = [str(f)+'/' for f in files_posix]
j = 0

for out_dir in tqdm(files):
  i = int(Path(out_dir).stem[-2:])
  if i >= j:
    try:
      x = node_creator(out_dir)
      n_nodes = x.shape[0]
      edge_index, edge_features = edge_index_creator(out_dir)
      n = hot_vector_plate_conection(out_dir, n_nodes)
      # combine x and n
      x = np.hstack((x, n))
      f = gravitational_force_creator(x[:, -1].reshape(-1, 1))
      y = x.copy()
      psi = stress_calculation(edge_features)
      data = [Data(x=torch.tensor(x, dtype=torch.float32), edge_index=torch.tensor(edge_index, dtype=torch.long), y=torch.tensor(y, dtype=torch.float32),edge_sup=torch.tensor(edge_features,dtype=torch.float32), f=torch.tensor(f, dtype=torch.float32), psi=torch.tensor(psi, dtype=torch.float32))]
      # number from the folder
    except Exception as e:
      print(e)
      continue
    save_dir = Path(args.par_dir) / 'pt_data'
    torch.save(data, save_dir / f'freefall_{i}.pt')
  else: continue

