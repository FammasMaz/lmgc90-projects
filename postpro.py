from utilities.postpro import *
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Step:1 
# creating contact forces files
par_dir = './train-track-static/data/seed_687_1/'
# list of observed nodes
track_every, body_idx = tracker(par_dir)
filt = True

contact_writer(par_dir)
contact_writer_wall(par_dir)
dof_writer(par_dir)
arvd_writer(par_dir)
# explanation of the columns
# 1:2 bodies (candidate and antagonist)
# 3:5 reaction forces (x,y,z)
# 6:8 velocities (x,y,z)
# 9 contact area gapTT
# 10:12 contact point
# 13:15 local tangential vector
# 16:18 local normal vector
# 19:21 local binormal vector
# 22: contact type

# Explanation of dof columns:
# 1 body 
# 2:4 displacement (x,y,z)
# 5:7 velocity (x,y,z)
# 8:10 angular (x,y,z)

# load the contacts
contacts = np.loadtxt(par_dir+'/OUTBOX/CONTACT_FORCES.dat')
contacts_wall = np.loadtxt(par_dir+'/OUTBOX/CONTACT_FORCES_WALL.dat')
dofs = np.loadtxt(par_dir+'/OUTBOX/DOF.DAT')
arvd = np.loadtxt(par_dir+'/OUTBOX/BODIES.DAT').reshape(-1,1)
# arvd only for body_idx
arvd = arvd[body_idx-1]


# only take into account the observed nodes, so see if body_idx is in 
# first two columns of contacts
if filt:
    filt_idx = np.where(np.isin(contacts[:,0], body_idx))[0]
    filt_idx = np.where(np.isin(contacts[filt_idx,1], body_idx))[0]
    contacts_mod = contacts[filt_idx,:]
# same for contacts_wall
    filt_idx = np.where(np.isin(contacts_wall[:,0], body_idx))[0]
    contacts_wall_mod = contacts_wall[filt_idx,:]



# position of the bodies
post_dir = par_dir+'POSTPRO/'
pos = read_bodies_vec(prenom='BODY_',body_idx=body_idx, dir=post_dir, idx_st=2)
# external forces acting on the body
ext_forces = read_bodies_vec(prenom='Fext_',body_idx=body_idx, dir=post_dir, idx_st=2)
res_forces = read_bodies_vec(prenom='REAC_',body_idx=body_idx, dir=post_dir, idx_st=2)
x_dict = {
    'pos': pos[-1,1:4],
    'res_forces': res_forces[-1,1:],
    'arvd': arvd.T
}
# forming the graph net
x = state_variables(x_dict)
edge_index = edge_index_creator(contacts_mod, contacts_wall_mod)
f = torch.tensor(ext_forces[-1,1:4], dtype=torch.float32).T

# augment position of ground as static 0 at the top of pos
pos_mod = np.concatenate((np.zeros((3,1)), pos[-1,1:4]), axis=1).T
int_vec, int_vec_wall = intercenter_vec(pos_mod, contacts_mod, contacts_wall_mod, track_every)
# edge features are concat of contacts_mod and int_vec
edge_features = np.concatenate((contacts_mod[:,2:], int_vec), axis=1)

