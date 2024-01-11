# imports
import os, sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath('gen_sample.py')))
sys.path.insert(0, parent_dir)
from datetime import datetime
import numpy as np
from pylmgc90 import pre
from utilities.lmgc90_utilities import * 
from utilities.creators import *
from utilities.generators import *

# create directories
seed=np.random.randint(1,1000,1)[0]
seed=687
cur_time = datetime.now()
par_dir = f'./seed_{seed}_' + cur_time.strftime('%d%m-%H%M%S') + '/'
[os.mkdir(par_dir) if not os.path.exists(par_dir) else None]
create_dirs(par_dir=par_dir)

## geometric params
dim = 3

# ballast params
ptype = 'POLYR' # particle type
gen_type = 'box'
Rmin = 0.4 # minimum radius of particles
Rmax = 2. # maximum radius of particles
max_particles = 2000 # number of particles
track_every = 50
min_vert = 6 # minimum number of vertices
max_vert = 18 # maximum number of vertices
Px = 20*Rmax # width of the particle generation
Py = 15*Rmax # height of the particle generation
Pz = 5*Rmax # depth of the particle generation
layers = [1.0526, 0.7522, 0.7520, 0.6131]
layers = layers[::-1]

# ground params
lx = 130 # width of the domain
ly = 100 # height of the domain
lz = 3 # depth of the domain

## rail params
nb_rails = 2
rail_length = Px*0.8
rail_height = 4*Rmax
rail_width = 2*Rmax
rail_spacing = 8*Rmax
rail_offset = 6*Rmax
rail_color = 'REDxx'


# friction params
pp = 0.3 # particle-particle friction
pw = 0.5 # particle-wall friction
rw = 0.5 # rail-wall friction
rp = 0.5 # rail-particle friction

# entities_to_gen
entities_to_gen = ['wall', 'ballast']


##########################################################################################

####### PREPROCESSING functions #######
## Avatar generation
def creation(entities_to_gen, dict_mat, dict_mod):
    # create materials
    mats = create_materials(dict_mat)
    # create models
    mods = create_models(dict_mod)
    # AVATAR GENERATION
    bodies=pre.avatars()
    ## wall avatar generation
    if 'wall' in entities_to_gen:
      wall_bodies = wall_generator(lx, ly, lz, mats['TDURx'], mods['rigid'])
      # impose driven dofs
      [wall_body.imposeDrivenDof(component=[1,2,3,4,5,6],dofty='vlocy') for wall_body in wall_bodies]
      bodies+=wall_bodies
    if 'ballast' in entities_to_gen:
      ## Ballast generation
      ballast_bodies = ballast_generator(max_particles, Rmin, Rmax, Px, Py, Pz, layers, mats['STONE'], mods['M3DH9'], ptype, min_vert, max_vert, seed=seed)
      bodies+=ballast_bodies
    if 'rail' in entities_to_gen:
      ## rail avatar generation
      rail_bodies = rail_generator(nb_rails, rail_length, rail_height, rail_width, rail_spacing, rail_offset, rail_color, mats['RAILx'], mods['M3DH8'])
      # impose driven dofs
      [rail.imposeDrivenDof(component=3,dofty='vlocy', ct=-20) for rail in rail_bodies]
      bodies+=rail_bodies
    return bodies, mats, mods, wall_bodies, ballast_bodies

## Interaction
def interaction(see_dict, post_dict):
    svs = create_see_tables(see_dict)
    post = create_postpro_commands(post_dict)
    return svs, post

## Output
def write_datbox(dim, bodies, mats, mods, tacts, svs, post, view=True):
   pre.visuAvatars(bodies)
    # write bodies
   pre.writeDatbox(dim,mats,mods,bodies,tacts,svs,post=post, datbox_path=os.path.join(par_dir,'DATBOX'))

####### PREPROCESSING #######
## dictionaries
# materials
dict_mat = {'TDURx': {'materialType':'RIGID', 'density':1000.},
            'PLEXx': {'materialType':'RIGID', 'density':500.},
            'STONE': {'materialType':'ELAS', 'density':2500., 'elas':'standard',
                      'anisotropy':'isotropic', 'young':50e9, 'nu':0.2},
            'RAILx': {'materialType':'ELAS', 'density':2500., 'elas':'standard',
                      'anisotropy':'isotropic', 'young':200e9, 'nu':0.2}}

#models
dict_mod = {'rigid': {'physics':'MECAx', 'element':'Rxx3D', 'dimension':dim},
            'M3DH8': {'physics':'MECAx', 'element':'H8xxx', 'dimension':dim,
                      'external_model':'MatL_', 'kinematic':'small', 'material':'elas_',
                      'anisotropy':'iso__', 'mass_storage':'lump_'},
            'M3DH9': {'physics':'MECAx', 'element':'Rxx3D', 'dimension':dim,
                      'external_model':'MatL_', 'kinematic':'small', 'material':'elas_',
                      'anisotropy':'iso__', 'mass_storage':'lump_'}}

## CREATE AVATARS
bodies, mat, mods, _, ballast_bodies = creation(entities_to_gen, dict_mat, dict_mod)

dict_tact = {'iqsc0': {'law':'IQS_CLB', 'fric':pp},
              'iqsc1': {'law':'IQS_CLB', 'fric':pw},
              'iqsc2': {'law':'GAP_SGR_CLB_g0', 'fric':rw},
              'iqsc3': {'law':'GAP_SGR_CLB_g0', 'fric':rp}}
## INTERACTION
tacts = create_tact_behavs(dict_tact)

dict_see = {'vpp': {'CorpsCandidat':'RBDY3', 'candidat':ptype,'colorCandidat':'BLUEx','behav':tacts['iqsc0'],
                    'CorpsAntagoniste':'RBDY3', 'antagoniste':ptype,'colorAntagoniste':'BLUEx',
                      'alert':0.1},
            'vpw': {'CorpsCandidat':'RBDY3', 'candidat':ptype,'colorCandidat':'BLUEx','behav':tacts['iqsc1'],
                    'CorpsAntagoniste':'RBDY3', 'antagoniste':'PLANx','colorAntagoniste':'WALLx',
                      'alert':0.1},
            'rww': {'CorpsCandidat':'MAILx', 'candidat':'CSxxx','colorCandidat':'REDxx','behav':tacts['iqsc2'],
                    'CorpsAntagoniste':'RBDY3', 'antagoniste':'PLANx','colorAntagoniste':'WALLx',
                      'alert':0.1},
            'rwp': {'CorpsCandidat':'MAILx', 'candidat':'CSxxx','colorCandidat':'REDxx','behav':tacts['iqsc3'],
                    'CorpsAntagoniste':'RBDY3', 'antagoniste':ptype,'colorAntagoniste':'BLUEx',
                      'alert':0.1}}
post_dict = {'CONTACT FORCE DISTRIBUTION': {'step':1, 'val':1},
              'COORDINATION NUMBER': {'step':1},
              'BODY TRACKING': {'step':1, 'rigid_set':ballast_bodies[::track_every]},
              'TORQUE EVOLUTION': {'step':1, 'rigid_set':ballast_bodies[::track_every]},
              'AVERAGE VELOCITY EVOLUTION': {'step':1, 'color':'BLUEx'},
              'KINETIC ENERGY': {'step':1},
              'DISSIPATED ENERGY': {'step':1}
              }

svs, post = interaction(dict_see, post_dict)
write_datbox(dim, bodies, mat, mods, tacts, svs, post)




