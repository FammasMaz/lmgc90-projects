from utilities.postpro_utilities import *
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch_geometric.data import Data   


# Step:0
# parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--par_dir', type=str, default='./train-track-static/data/sncf_random_test_2/')
parser.add_argument('--par_dir_type', type=str, default='str')
parser.add_argument('--filt', type=bool, default=False)
args = parser.parse_args()

# def dataset_creator(par_dir, filt=True, step='last'):

# # Step:1 
# # creating contact forces files
# # list of observed nodes
#     contact_writer(par_dir)
#     contact_writer_wall(par_dir)
#     dof_writer(par_dir)
#     arvd_writer(par_dir)
#     # explanation of the columns
#     # 1:2 bodies (candidate and antagonist)
#     # 3:5 reaction forces (x,y,z)
#     # 6:8 velocities (x,y,z)
#     # 9 contact area gapTT
#     # 10:12 contact point
#     # 13:15 local tangential vector
#     # 16:18 local normal vector
#     # 19:21 local binormal vector
#     # 22: contact type

#     # Explanation of dof columns:
#     # 1 body 
#     # 2:4 displacement (x,y,z)
#     # 5:7 velocity (x,y,z)
#     # 8:10 angular (x,y,z)

#     # load the contacts
#     contacts = np.loadtxt(par_dir+'/OUTBOX/CONTACT_FORCES.dat')
#     contacts_wall = np.loadtxt(par_dir+'/OUTBOX/CONTACT_FORCES_WALL.dat')
#     dofs = np.loadtxt(par_dir+'/OUTBOX/DOF.DAT')
#     arvd = np.loadtxt(par_dir+'/OUTBOX/BODIES.DAT').reshape(-1,1)
    
#     # filt: if true then find body_idx and only track those
#     if filt:
#         track_every, body_idx = tracker(par_dir)



#     # only take into account the observed nodes, so see if body_idx is in 
#     # first two columns of contacts
#     post_dir = par_dir+'POSTPRO/'

#     if filt:
#         filt_idx = np.where(np.isin(contacts[:,0], body_idx))[0]
#         filt_idx = np.where(np.isin(contacts[filt_idx,1], body_idx))[0]
#         contacts_mod = contacts[filt_idx,:]
#     # same for contacts_wall
#         filt_idx = np.where(np.isin(contacts_wall[:,0], body_idx))[0]
#         contacts_wall_mod = contacts_wall[filt_idx,:]
#         pos = read_bodies_vec(prenom='BODY_',body_idx=body_idx, dir=post_dir, idx_st=2)
#         # external forces acting on the body
#         ext_forces = read_bodies_vec(prenom='Fext_',body_idx=body_idx, dir=post_dir, idx_st=2)
#         res_forces = read_bodies_vec(prenom='REAC_',body_idx=body_idx, dir=post_dir, idx_st=2)
#     # position of the bodies
        

    
#     x_dict = {
#         'pos': pos[-1,1:4],
#         'res_forces': res_forces[-1,1:],
#         'arvd': arvd.T
#     }
#     # forming the graph net
#     x = state_variables(x_dict)
#     edge_index = edge_index_creator(contacts_mod, contacts_wall_mod)
#     f = torch.tensor(ext_forces[-1,1:4], dtype=torch.float32).T

#     # augment position of ground as static 0 at the top of pos
#     pos_mod = np.concatenate((np.zeros((3,1)), pos[-1,1:4]), axis=1).T
#     int_vec, int_vec_wall = intercenter_vec(pos_mod, contacts_mod, contacts_wall_mod, track_every)
#     # edge features are concat of contacts_mod and int_vec
#     edge_features = np.concatenate((contacts_mod[:,2:], int_vec), axis=1)


# Modifying the above
filt_single = lambda x, body_idx: np.where(np.isin(x[:,0], body_idx))[0]
filt_double = lambda x, body_idx: np.where(np.isin(x[:,0], body_idx) & np.isin(x[:,1], body_idx))[0]
def dataset_creator(par_dir, filt=True):
    # post processing directory 
    post_dir = par_dir+'POSTPRO/'

    # load the processed data from dat files
    contacts = np.loadtxt(par_dir+'/OUTBOX/CONTACT_FORCES.dat')
    contacts_wall = np.loadtxt(par_dir+'/OUTBOX/CONTACT_FORCES_WALL.dat')
    dofs = np.loadtxt(par_dir+'/OUTBOX/DOF.DAT')
    arvd = np.loadtxt(par_dir+'/OUTBOX/BODIES.DAT')

    # two cases, one where we filt and one where we dont
    if filt:
        track_every, body_idx = tracker(par_dir)
        # combine the two lines in one
        contacts = contacts[filt_double(contacts, body_idx),:]
        contacts_wall = contacts_wall[filt_double(contacts_wall, body_idx),:]
        dofs = dofs[filt_single(dofs, body_idx),:]
        arvd = arvd[filt_single(arvd, body_idx),:]
        pos = read_bodies_vec(prenom='BODY_',body_idx=body_idx, dir=post_dir, idx_st=2)
        # external forces acting on the body
        ext_forces = read_bodies_vec(prenom='Fext_',body_idx=body_idx, dir=post_dir, idx_st=2)
        res_forces = read_bodies_vec(prenom='REAC_',body_idx=body_idx, dir=post_dir, idx_st=2)
        x_dict = {'pos': pos[-1,1:4].T,
                  'res_forces': res_forces[-1,1:].T,
                  'arvd': arvd[:,1:]
                  }
    else:
        track_every = 0
        pos = dofs[:, 1:4]
        x_dict = {'pos': pos,
                  'arvd': arvd[:,1:]
                  }
        
    # forming the graph net
    x = state_variables(x_dict)
    edge_index = edge_index_creator(contacts, contacts_wall)
    if filt:
        f = torch.tensor(ext_forces[-1,1:4], dtype=torch.float32).T
    else:
        # apply acceleration in z direction equal to gravity size equal to pos, only 3 component 9.8
        f = torch.tensor(np.zeros((pos.shape[0],3)), dtype=torch.float32)
        # apply force to z direction of all bodies except first
        f[1:,2] = -9.8
# augment position of ground if filt was used
    pos_mod = pos_mod = np.concatenate((np.zeros((3,1)), pos[-1,1:4]), axis=1).T if filt else pos
    int_vec = intercenter_vec(pos_mod, contacts,track_every=track_every)
    int_vec_wall = intercenter_vec(pos_mod, contacts_wall, track_every=track_every)
    # edge features are concat of contacts_mod and int_vec
    edge_features = np.concatenate((contacts[:,2:], int_vec), axis=1)
    edge_features_wall = np.concatenate((contacts_wall[:,2:], int_vec_wall), axis=1)
    edge_features = np.concatenate((edge_features, edge_features_wall), axis=0)
    psi = stress_calculator(contacts[:,2:5], int_vec)
    dat = Data(x=x, edge_index=edge_index, edge_attr=torch.tensor(edge_features, dtype=torch.float32), y=x, f=f, psi=psi)

    return dat
def raw_processor(par_dir):
        # write necessary files
    ''' 
    * contact_writer and contact_writer_wall
        Columns: 
            1:2 bodies (candidate and antagonist)
            3:5 reaction forces (x,y,z)
            6:8 velocities (x,y,z)
            9 contact area gapTT
            10:12 contact point
            13:15 local tangential vector
            16:18 local normal vector
            19:21 local binormal vector
            22: contact type
    * dof_writer
        Columns:
            1 body 
            2:4 displacement (x,y,z)
            5:7 velocity (x,y,z)
            8:10 angular (x,y,z)

    * arvd_writer
        Columns:
            1 body arvd
    '''

    contact_writer(par_dir)
    contact_writer_wall(par_dir)
    dof_writer(par_dir)
    arvd_writer(par_dir)

#raw_processor(args.par_dir)
save_dir = './train-track-static/data/pt_data/'
list_par_dir = [args.par_dir+f'sncf_random_test_{i}/' for i in range(2,3)]
if args.par_dir_type == 'str': 
    dat = dataset_creator(args.par_dir, filt=args.filt)
else:
    for par_dir in list_par_dir:
        # extract digit at the last of par_dir
        i = int(par_dir[-2])
        raw_processor(par_dir)
        dat = dataset_creator(par_dir, filt=args.filt)
        torch.save(dat, save_dir+f'freefall_{i}.pt')
        print(f'File saved\n')


