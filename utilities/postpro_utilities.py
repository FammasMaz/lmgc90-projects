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
import pickle 

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

def coordinates_writer(par_dir):
    samples = sampler(par_dir, fname = 'BODIES.OUT', sw = '$nodty')
    indices = [3,5,7]
    # remove if file exists
    if os.path.exists(par_dir + 'OUTBOX/COORDINATES.DAT'):
        os.remove(par_dir + 'OUTBOX/COORDINATES.DAT')
    # write in dat file and also write in pickle
    coord = np.array([])
    with open(par_dir + 'OUTBOX/COORDINATES.DAT', 'w') as file:
        for sample in samples:
            numbers = extract_numbers(sample[1:])
            numbers_out = [numbers[i] for i in indices]
            # change to scientific notation .6e
            numbers_out = [f'{i:.6e}' for i in numbers_out]
            file.write('    '.join([str(i) for i in numbers_out]))
            file.write('\n')
            coord = np.append(coord, numbers_out)
    coord = coord.reshape(-1,len(indices))
    with open(par_dir + 'OUTBOX/COORDINATES.pkl', 'wb') as file:
        pickle.dump(coord, file)
    

        


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
    

    
def dof_writer(par_dir, fname = 'DOF.LAST'):
    samples = sampler(par_dir, fname = fname,sw='$bdyty')
    indices = [5, 7, 9, 17, 19, 21, 29, 31, 33, 35, 37, 39, 41, 43, 45]
    # remove if file exists
    if os.path.exists(par_dir + 'OUTBOX/DOF.DAT'):
        os.remove(par_dir + 'OUTBOX/DOF.DAT')
    dof = np.array([])
    with open(par_dir + 'OUTBOX/DOF.DAT', 'w') as file:
        for sample in samples:
            numbers = extract_numbers(sample[1:])
            numbers_out = [numbers[i] for i in indices]
            # change to scientific notation .6e
            numbers_out = [f'{i:.6e}' for i in numbers_out]
            file.write('    '.join([str(i) for i in numbers_out]))
            file.write('\n')
            dof = np.append(dof, numbers_out)
    dof = dof.reshape(-1, len(indices))
    with open(par_dir + 'OUTBOX/DOF.pkl', 'wb') as file:
        pickle.dump(dof, file)


def mass_writer(par_dir, density = 2900, plate=True):
    samples = sampler(par_dir, fname = 'BODIES.OUT',sw='$tacty')
    mass = np.array([])
    nb_plates = 6
    # append mass of plate 
    density = density
    with open(par_dir + 'OUTBOX/MASS.DAT', 'w') as file:
        # run for bodies minus the nb_plates at last
        ballast_samples = samples[:-nb_plates]
        for sample in ballast_samples:
            numbers = extract_numbers(sample[1:])
            nb_vertices = int(numbers[1])
            nb_faces = int(numbers[2])
            # vertices is every second number starting from 5 and ending at 5+nb_vertices*3
            vertices = np.array(numbers[4:4+nb_vertices*6:2]).reshape(nb_vertices, 3).astype(float)
            ind_vert = nb_vertices*6 + 4
            faces = np.array(numbers[ind_vert:ind_vert+nb_faces*6:2]) -1
            faces = faces.reshape(nb_faces, 3).astype(int)   
            # swap the third and second column
            faces[:,[1, 2]] = faces[:,[2, 1]]
            total_volume = sum(tetrahedron_volume(vertices[face[0]], vertices[face[1]], vertices[face[2]]) for face in faces)
            m = total_volume * density
            file.write(f'{m:.6e}')
            file.write('\n')
            mass = np.append(mass,  m)
    if plate: 
        for _ in range(nb_plates):
            mass = np.append(mass, 1.)
    
    mass = mass.reshape(-1,1)
    with open(par_dir + 'OUTBOX/mass.pkl', 'wb') as file:
        pickle.dump(mass, file)

            


def edge_index_creator(contacts, contacts_wall):
    # contactenate the first two columns of contacts and contacts_wall
    # and create the edge_index
    edge_index = np.concatenate((contacts[:,:2], contacts_wall[:,:2]), axis=0)
    edge_index = torch.tensor(edge_index, dtype=torch.float32)
    return edge_index

def state_variables(dict):
    x = torch.tensor([], dtype=torch.float32)
    for key in dict:
        x = torch.cat((x, torch.tensor(dict[key], dtype=torch.float32)), dim=1)
    return x


def arvd_writer(par_dir):
    samples = sampler(par_dir, fname = 'BODIES.OUT',sw='$bdyty')
    indices = [1,3]
    # remove if file exists
    if os.path.exists(par_dir + 'OUTBOX/BODIES.DAT'):
        os.remove(par_dir + 'OUTBOX/BODIES.DAT')
    arvd = np.array([])

    with open(par_dir + 'OUTBOX/BODIES.DAT', 'w') as file:
        # first sample is the plane
        for sample in samples:
            numbers = extract_numbers(sample[1:])
            numbers_out = [numbers[i] for i in indices]
            # change to scientific notation .6e
            numbers_out = [f'{i:.6e}' for i in numbers_out]
            file.write('    '.join([str(i) for i in numbers_out]))
            file.write('\n')
            arvd = np.append(arvd, numbers_out)
    arvd = arvd.reshape(-1,2)
    with open(par_dir + 'OUTBOX/arvd.pkl', 'wb') as file:
        pickle.dump(arvd, file)



def intercenter_vec(pos, contacts, track_every=1):
    # for each contact find the cand and ant positions
    idx = [int((i-1)//(track_every+1)) for i in contacts[:,0]] if track_every!=0 else contacts[:,0].astype(int)-1
    idx_ant = [int((i-1)//(track_every+1)) for i in contacts[:,1]] if track_every!=0 else contacts[:,1].astype(int)-1
    
    cand = pos[idx,:]
    ant = pos[idx_ant,:]
    # find the intercenter vector
    inter_vec = cand - ant
    return inter_vec

def stress_calculator(contact_forces, inter_vec):
    # contact_forces: (nb_contacts, 3)
    # inter_vec: (nb_contacts, 3)
    # stress: (nb_contacts, 3)
    stress = torch.tensor(contact_forces * inter_vec, dtype=torch.float32)
    return stress

def stress_caculator_voigt(contact_forces, inter_vec, princ_stress_cal=True):
    n_contacts = contact_forces.shape[0]
    stresses = torch.zeros((n_contacts, 6))
    princ_stresses = torch.zeros((n_contacts, 3))
    for i in range(n_contacts):
        force = contact_forces[i]
        vec = inter_vec[i]
        stress_tensor = np.outer(force, vec)
        stress_voigt = np.array([stress_tensor[0,0], stress_tensor[1,1], stress_tensor[2,2], stress_tensor[0,1], stress_tensor[1,2], stress_tensor[0,2]])
        stresses[i] = torch.tensor(stress_voigt, dtype=torch.float32)
        # principle stresses are the first three eigenvalues
        eigvals = np.linalg.eigvals(stress_tensor)
        # the eigenvalues are complex, take the real part
        princ_stresses[i] = torch.tensor(np.real(eigvals), dtype=torch.float32)
    if princ_stress_cal:
        return princ_stresses
    else: 
        return stresses

def stress_calculator_grain(global_forces, edge_index, inter_center, n_nodes,princ_stress_cal=True):
    n_contacts = edge_index.shape[1]
    stresses = torch.zeros((n_nodes, 6))
    princ_stresses = torch.zeros((n_nodes, 3))
    for i in range(n_contacts):
        # weber tensor of stress
        weber_tensor = np.outer(global_forces[i], inter_center[i])
        voigt = torch.tensor([weber_tensor[0,0], weber_tensor[1,1], weber_tensor[2,2], weber_tensor[0,1], weber_tensor[1,2], weber_tensor[0,2]], dtype=torch.float32)

        # transfer this contact stress to the nodes/grians only the diagonals
        stresses[int(edge_index[0,i]),:] += voigt
        stresses[int(edge_index[1,i]),:] -= voigt
        eigvals = np.linalg.eigvals(weber_tensor)
        # the eigenvalues are complex, take the real part
        # and put the princpal stressses in the princ_stresses
        princ_stresses[int(edge_index[0,i])] += torch.tensor(np.real(eigvals), dtype=torch.float32)
        princ_stresses[int(edge_index[1,i])] -= torch.tensor(np.real(eigvals), dtype=torch.float32)
    if princ_stress_cal:
        return princ_stresses
    else:
        return stresses

def read_pickled_file(par_dir, fname, dir='OUTBOX/'):
    if dir: par_dir = par_dir + dir
    with open(par_dir + fname, 'rb') as file:
        data = pickle.load(file)
    return data 


def tetrahedron_volume(a, b, c):
    return abs(np.dot(a, np.cross(b, c))) / 6


def gravitational_force_creator(m, g=9.8):
    f = np.zeros((m.shape[0], 3))
    # first one is rigid plate
    f[5:,2] = -m*g
    return f


class pickle_procesor:
    def __init__(self, par_dir):
        self.par_dir = par_dir
        self.postpro = read_pickled_file(par_dir, 'postpro.pkl', False)
        
    def coordinates(self):
        return self.postpro[4]
    def plate_connection(self):
        inter = self.postpro[1]
        inter_filt = [inter[i] for i in range(len(inter)) if inter[i][0]==b'PRPLx']
        stacked = np.array([inter_filt[i][3] for i in range(len(inter_filt))]).reshape(-1,1)
        return stacked.astype(int)
    def fint(self):
        return self.postpro[5]
    def fext(self):
        return self.postpro[6]
    def edge_data(self):
        return self.postpro[3]
    def edge_index(self):
        return np.array(self.postpro[3][:,1:3]).T -1
    def edge_features(self):    
        return self.postpro[3][:,3:]
    def n_nodes(self):
        return self.postpro[4].shape[0]
    def global_force(self):
        # calculate the global force for each contact point
        edge_features = self.edge_features()
        edge_index = self.edge_index()
        n_contacts = edge_index.shape[1]
        global_force = np.zeros((n_contacts, 3))
        # last three edge_features are intercenter vectors 
        # nine before that are local frame
        frame = edge_features[:,-12:-3].reshape(-1,3,3)
        # first three are contact forces
        contact_forces = edge_features[:,:3]
        # calculate the global force
        for i in range(n_contacts):
            global_force[i] = np.dot(frame[i], contact_forces[i])
        return global_force
    

    
    
    
    
    





