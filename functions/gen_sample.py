# imports
import os, sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath('gen_sample.py')))
sys.path.insert(0, parent_dir)
from datetime import datetime
import numpy as np
from pylmgc90 import pre
from utilities.utilities import * 
from utilities.creators import *
from utilities.generators import *

def random_ballast_sample(par_dir, seed=687, visu=False, step=1):

    dim = 3

    # ballast params
    ptype = 'POLYR' # particle type
    gen_type = 'box'
    Rmin = 0.4 # minimum radius of particles
    Rmax = 2. # maximum radius of particles
    max_particles = 4000 # number of particles
    min_vert = 6 # minimum number of vertices
    max_vert = 18 # maximum number of vertices
    Px = np.random.randint(10,40)*Rmax # width of the particle generation
    Py = np.random.randint(10,30)*Rmax # height of the particle generation
    Pz = 5*Rmax # depth of the particle generation
    layers = np.linspace(1,0.5, np.random.randint(3,6))
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
        ballast_bodies, total_paricles = ballast_generator(max_particles, Rmin, Rmax, Px, Py, Pz, layers, mats['STONE'], mods['M3DH9'], ptype, min_vert, max_vert, seed=seed)
        bodies+=ballast_bodies
        track_every = int(total_paricles/50)
    if 'rail' in entities_to_gen:
    ## rail avatar generation
        rail_bodies = rail_generator(nb_rails, rail_length, rail_height, rail_width, rail_spacing, rail_offset, rail_color, mats['RAILx'], mods['M3DH8'])
    # impose driven dofs
        [rail.imposeDrivenDof(component=3,dofty='vlocy', ct=-20) for rail in rail_bodies]
        bodies+=rail_bodies
    dict_tact = {'iqsc0': {'law':'IQS_CLB', 'fric':pp},
                'iqsc1': {'law':'IQS_CLB', 'fric':pw},
                'iqsc2': {'law':'GAP_SGR_CLB_g0', 'fric':rw},
                'iqsc3': {'law':'GAP_SGR_CLB_g0', 'fric':rp}}
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
    post_dict = {'CONTACT FORCE DISTRIBUTION': {'step':step, 'val':1},
                'COORDINATION NUMBER': {'step':step},
                'BODY TRACKING': {'step':step, 'rigid_set':ballast_bodies[::track_every]},
                'TORQUE EVOLUTION': {'step':step, 'rigid_set':ballast_bodies[::track_every]},
                'AVERAGE VELOCITY EVOLUTION': {'step':step, 'color':'BLUEx'},
                'KINETIC ENERGY': {'step':step},
                'DISSIPATED ENERGY': {'step':step}
                }
    svs = create_see_tables(dict_see)
    post = create_postpro_commands(post_dict)
    [pre.visuAvatars(bodies) if visu else None]
    pre.writeDatbox(dim,mats,mods,bodies,tacts,svs,post=post, datbox_path=os.path.join(par_dir,'DATBOX'))
    dict_sample = {'Px':Px, 'Py':Py, 'Pz':Pz, 'Total Particles':total_paricles, 'Track Every': track_every, 'Layers':len(layers)}
    return dict_sample



