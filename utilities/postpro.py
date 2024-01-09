'''Post processing utilities that generate data for the graph neural networks in
the graph format specified by PyTorch Geometric.'''

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch_geometric.data import Data   
import torch
import re
from pathlib import Path    

# vectorize function
# index 2 indicates the first body
def read_bodies_vec(prenom, body_idx, dir,idx_st=2):
    '''
    Read the corresponding .dat files and returns a numpy array
    of shape (3, nb_bodies, nb_timesteps) where nb_bodies is the number of bodies
    and nb_timesteps is the number of timesteps.
    Args:
        prenom: string, prefix of the .dat files
        par_dir: string, parent directory
        dir: string, directory containing the .dat files
        idx_st: int, index of the first body
    Returns:
        bt: numpy array of shape (3, nb_bodies, nb_timesteps)

        '''
    ext = '.dat'
    names = [prenom + str(i).zfill(7) + '.dat' for i in body_idx] if idx_st !=1 else [prenom + '.dat']
    bt = np.array([np.loadtxt(os.path.join(dir,name)) for name in names])
    return bt.transpose(1,2,0)


# return samples
def sampler(par_dir, fname='Vloc_Rloc.OUT',sw = '$icdan'):
    with open(Path(par_dir+'OUTBOX/'+fname)) as file:
        lines = file.readlines()
    samples = []
    current_sample = []
    # each sample is the lines till starting from "$icdan" till the next "$icdan"
    for line in lines:
        if line.startswith(sw):
            samples.append(current_sample)
            current_sample = []
        current_sample.append(line)
    samples.append(current_sample)
    return samples[1:]

def extract_numbers(strings):
    numbers = []
    for string in strings:
        # Replace 'D' with 'E' in the scientific notation
        formatted_string = string.replace('D', 'E')
        # Find all numbers in the string
        found_numbers = re.findall(r'[-+]?\d*\.\d+E[-+]?\d+|[-+]?\d+\.\d*|[-+]?\d+', formatted_string)
        # Convert to float and add to the list
        numbers.extend(float(num) for num in found_numbers)
    return numbers

def contact_writer(par_dir):
    samples = sampler(par_dir)
    indices = [1,6,8,9,10,11,12,13,14,16,18,20,22,24,26,28,30,32,34,36,38]
    # remove if file exists
    if os.path.exists(par_dir + 'OUTBOX/CONTACT_FORCES.DAT'):
        os.remove(par_dir + 'OUTBOX/CONTACT_FORCES.DAT')
    with open(par_dir + 'OUTBOX/CONTACT_FORCES.DAT', 'w') as file:
        for sample in samples:
            if 'PLANx' in sample[2].split():
                continue

            numbers = extract_numbers(sample[1:])
            if 'stick' in sample[2].split():
                stick_idx = 56
            elif 'slip' in sample[2].split():
                stick_idx = 57
            else:
                stick_idx = 58
            numbers_out = [numbers[i] for i in indices]
            # change to scientific notation .6e
            numbers_out = [f'{i:.6e}' for i in numbers_out]
            file.write('    '.join([str(i) for i in numbers_out]))
            file.write('    ')
            file.write('    '.join([str(stick_idx)]))
            file.write('\n')

def contact_writer_wall(par_dir):
    samples = sampler(par_dir)
    indices = [1,6,8,9,10,11,12,13,14,16,18,20,22,24,26,28,30,32,34,36,38]
    # remove if file exists
    if os.path.exists(par_dir + 'OUTBOX/CONTACT_FORCES_WALL.DAT'):
        os.remove(par_dir + 'OUTBOX/CONTACT_FORCES_WALL.DAT')
    with open(par_dir + 'OUTBOX/CONTACT_FORCES_WALL.DAT', 'w') as file:
        for sample in samples:
            if 'PLANx' not in sample[2].split():
                continue
            numbers = extract_numbers(sample[1:])
            if 'stick' in sample[2].split():
                stick_idx = 56
            elif 'slip' in sample[2].split():
                stick_idx = 57
            else:
                stick_idx = 58
            numbers_out = [numbers[i] for i in indices]
            # change to scientific notation .6e
            numbers_out = [f'{i:.6e}' for i in numbers_out]
            file.write('    '.join([str(i) for i in numbers_out]))
            file.write('    ')
            file.write('    '.join([str(stick_idx)]))
            file.write('\n')

def tracker(par_dir, start_idx=2):
    nb_files = len([name for name in os.listdir(Path(par_dir + 'POSTPRO')) if name.startswith('BODY_')])
    with open(par_dir + 'dict.txt') as file:
        dict = json.load(file)  
        track_every = dict['Track Every']
    body_idx = np.arange(start_idx, nb_files*track_every, track_every)
    return track_every, body_idx
    

    
def dof_writer(par_dir):
    samples = sampler(par_dir, fname = 'DOF.OUT',sw='$bdyty')
    indices = [1, 5, 7, 9, 17, 19, 21, 23, 25, 27]
    # remove if file exists
    if os.path.exists(par_dir + 'OUTBOX/DOF.DAT'):
        os.remove(par_dir + 'OUTBOX/DOF.DAT')
    with open(par_dir + 'OUTBOX/DOF.DAT', 'w') as file:
        for sample in samples:
            numbers = extract_numbers(sample[1:])
            numbers_out = [numbers[i] for i in indices]
            # change to scientific notation .6e
            numbers_out = [f'{i:.6e}' for i in numbers_out]
            file.write('    '.join([str(i) for i in numbers_out]))
            file.write('\n')

def edge_index_creator(contacts, contacts_wall):
    # contactenate the first two columns of contacts and contacts_wall
    # and create the edge_index
    edge_index = np.concatenate((contacts[:,:2], contacts_wall[:,:2]), axis=0)
    edge_index = torch.tensor(edge_index, dtype=torch.float32).transpose(0,1)
    return edge_index

def state_variables(dict):
    x = torch.tensor([], dtype=torch.float32)
    for key in dict:
        x = torch.cat((x, torch.tensor(dict[key], dtype=torch.float32)), dim=0)
    return x.transpose(0,1)


def arvd_writer(par_dir):
    samples = sampler(par_dir, fname = 'BODIES.OUT',sw='$blmty')
    # ignore the first sample
    indices = [1]
    # remove if file exists
    if os.path.exists(par_dir + 'OUTBOX/BODIES.DAT'):
        os.remove(par_dir + 'OUTBOX/BODIES.DAT')
    with open(par_dir + 'OUTBOX/BODIES.DAT', 'w') as file:
        # first sample is the plane
        for sample in samples:
            numbers = extract_numbers(sample[1:])
            numbers_out = [numbers[i] for i in indices]
            # change to scientific notation .6e
            numbers_out = [f'{i:.6e}' for i in numbers_out]
            file.write('    '.join([str(i) for i in numbers_out]))
            file.write('\n')

def intercenter_vec(pos, contacts_mod, contacts_wall_mod, track_every):
    inter_vec = np.zeros((contacts_mod.shape[0],3))
    inter_vec_wall = np.zeros((contacts_wall_mod.shape[0],3))
    for i in range(contacts_mod.shape[0]):
        # find the index of the contact point
        inter_vec[i,:] = pos[int(contacts_mod[i,0]//track_every)-1] - pos[int(contacts_mod[i,1]//track_every)-1]
        # same for the wall
    for i in range(contacts_wall_mod.shape[0]):
        inter_vec_wall = pos[int(contacts_wall_mod[i,0]//track_every)-1] - pos[int(contacts_wall_mod[i,1]//track_every)-1]
    return inter_vec, inter_vec_wall
