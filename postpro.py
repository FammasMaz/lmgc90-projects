# writing the postpro script
import numpy as np
from utilities.postpro_utilities import pickle_procesor, mass_writer, dof_writer, arvd_writer, read_pickled_file, stress_caculator_voigt, stress_calculator_grain
from torch_geometric.data import Data   
import argparse
from pathlib import Path
import torch
from tqdm.auto import tqdm


def str2bool(v):
    if v.lower() in ('yes', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'f', 'n', '0'):
        return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()  
parser.add_argument('--par_dir', type=str, default='/Users/fammasmaz/Downloads/')
parser.add_argument('--stem', type=str, default='sncf_random_test_2') 
parser.add_argument('--psi_grain', type=str2bool, default=False)
args = parser.parse_args()


class stack_creator:
  '''
  This class is used to create the stack of nodes and edges for the graph neural network
  It uses the pickle_procesor class to read the pickle files and create the nodes and edges
  and then define the connection with plates
  '''
  def __init__(self, out_dir):
      self.out_dir = out_dir
      self.picked = pickle_procesor(out_dir)
      self.nb_plates = 8
  def nodes_creator(self):
      # dof x(3), v(3), a (9)
      # coord x, y, z
      # arvd 1
      # mass 1
      # fint 3
      # fext 3
      coord = self.picked.coordinates() # coordinates coming from pickle, includes the plates
      fint = self.picked.fint() # internal forces, includes the plates
      fext = self.picked.fext()
      dof_writer(self.out_dir)
      dof = read_pickled_file(self.out_dir, 'dof.pkl')
      arvd_writer(self.out_dir)
      arvd = read_pickled_file(self.out_dir, 'arvd.pkl')
      mass_writer(self.out_dir)
      mass = read_pickled_file(self.out_dir, 'mass.pkl')
      x = np.hstack((dof, coord, arvd[:,1].reshape(-1,1), mass, fint, fext))
      x = x.astype(np.float32)
      return x, mass
  def edge_index_creator(self):
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
      edge_features = self.picked.edge_features()
      edge_index = self.picked.edge_index()
      return edge_index.astype(np.float32), edge_features.astype(np.float32)
  def hot_vector_plate_conection(self):
      n_nodes = self.picked.n_nodes()
      plated = self.picked.plate_connection()
      x = np.zeros((n_nodes, 1))
      x[plated-1] = 1
      # set last six to 1
      x[-self.nb_plates:] = 1
      return x.astype(np.float32)
  def global_force(self):
      return self.picked.global_force()


  



def gravitational_force_creator(x, mass, nb_plates=6):
      n_nodes = x.shape[0]
      f = np.zeros((n_nodes, 3))
      # last six are rigid plates, set their force zero and rest to gravity
      f[:-nb_plates, 2] = -mass[:-nb_plates, 0]*9.8
      return f.astype(np.float32)

def stress_calculation(edge_features):
   reaction_forces = edge_features[:, 0:3]
   intercenter_vectors = edge_features[:, -3:]
   psi = stress_caculator_voigt(reaction_forces, intercenter_vectors,princ_stress_cal=True)
   return np.array(psi).astype(np.float32)

def stress_calculation_grain(edge_features,global_forces, edge_index, n_nodes):
   intercenter_vectors = edge_features[:, -3:]
   psi = stress_calculator_grain(global_forces, edge_index,intercenter_vectors, n_nodes)
   return np.array(psi).astype(np.float32)




## Post processing
par_dir = Path(args.par_dir)
files_posix = [f for f in par_dir.iterdir() if f.stem.startswith(args.stem)]
# in string format
files = [str(f)+'/' for f in files_posix]
j = 0
for out_dir in tqdm(files):
  i = int(Path(out_dir).stem[-2:])
  if i >= j:
    try:
      x, mass = stack_creator(out_dir).nodes_creator()
      n_nodes = x.shape[0]
      edge_index, edge_features = stack_creator(out_dir).edge_index_creator()
      n = stack_creator(out_dir).hot_vector_plate_conection()
      # combine x and n
      x = np.hstack((x, n))
      f = gravitational_force_creator(x, mass)
      fg = stack_creator(out_dir).global_force()
      if not args.psi_grain:
        psi = stress_calculation(edge_features)
      else:
        psi = stress_calculation_grain(edge_features, fg, edge_index, n_nodes) # psi at granular level
      data = [Data(x=torch.tensor(x, dtype=torch.float32), edge_index=torch.tensor(edge_index, dtype=torch.long), y=torch.tensor(x, dtype=torch.float32),edge_sup=torch.tensor(edge_features,dtype=torch.float32), f=torch.tensor(f, dtype=torch.float32), psi=torch.tensor(psi, dtype=torch.float32))]
      # number from the folder
      save_dir = Path(args.par_dir) / 'closet_pt'
      torch.save(data, save_dir / f'freefall_{i}.pt')
    except Exception as e:
      print(e)
      continue
  else: continue
