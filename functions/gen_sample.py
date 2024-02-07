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
    Px = np.random.randint(10,30)*Rmax # width of the particle generation
    Py = np.random.randint(10,25)*Rmax # height of the particle generation
    Pz = 5*Rmax # depth of the particle generation
    layers = np.linspace(1,0.5, np.random.randint(5,12))
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
        ballast_bodies, total_paricles, par_char = ballast_generator(max_particles, Rmin, Rmax, Px, Py, Pz, layers, mats['STONE'], mods['M3DH9'], ptype, min_vert, max_vert, seed=seed)
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
                        'alert':0.051},
                'vpw': {'CorpsCandidat':'RBDY3', 'candidat':ptype,'colorCandidat':'BLUEx','behav':tacts['iqsc1'],
                        'CorpsAntagoniste':'RBDY3', 'antagoniste':'PLANx','colorAntagoniste':'WALLx',
                        'alert':0.05},
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
    par_char_f = 'particle_characteristics.dat'
    with open(os.path.join(par_dir, par_char_f), 'w') as f:
        for row in zip(*par_char.values()):
            f.write('   '.join(str(x) for x in row)+'\n')
    return dict_sample

def random_ballast_deng(par_dir, seed=687, visu=False, step=1):

    dim = 3

    # ballast params
    ptype = 'POLYR' # particle type
    gen_type = 'box'
    Rmin = 0.025 # minimum radius of particles
    Rmax = 0.060 # maximum radius of particles
    max_particles = 2000 # number of particles
    min_vert = 9 # minimum number of vertices
    max_vert = 20 # maximum number of vertices
    Px = 2.01 # width of the particle generation
    Py = 0.51 # height of the particle generation
    Pz = 0.42 # depth of the particle generation
    layers = np.linspace(1,0.5,np.random.randint(5,8)) # stacks of particles
    layers = layers[::-1]

    # ground params
    lx = 5 # width of the domain
    ly = 3 # height of the domain
    lz = 0.15 # depth of the domain

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
    pw = 0.8 # particle-wall friction
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
        ballast_bodies, total_paricles, par_char = ballast_generator(max_particles, Rmin, Rmax, Px, Py, Pz, layers, mats['TDURx'], mods['rigid'], ptype, min_vert, max_vert, seed=seed)
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
                        'alert':0.01},
                'vpw': {'CorpsCandidat':'RBDY3', 'candidat':ptype,'colorCandidat':'BLUEx','behav':tacts['iqsc1'],
                        'CorpsAntagoniste':'RBDY3', 'antagoniste':'PLANx','colorAntagoniste':'WALLx',
                        'alert':0.01}}
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
    par_char_f = 'particle_characteristics.dat'
    with open(os.path.join(par_dir, par_char_f), 'w') as f:
        for row in zip(*par_char.values()):
            f.write('   '.join(str(x) for x in row)+'\n')
    return dict_sample







def random_ballast_sncf_ir(par_dir, seed=687, visu=False, step=1):
    
    dim = 3

    # ballast params
    ptype = 'POLYR' # particle type
    gen_type = 'cube'
    Rmin = 0.025 # minimum radius of particles
    Rmax = 0.060 # maximum radius of particles
    max_particles = 2000 # number of particles
    min_vert = 9 # minimum number of vertices
    max_vert = 20 # maximum number of vertices
    Px = 2.01 # width of the particle generation
    Py = 0.51 # height of the particle generation
    Pz = 0.42 # depth of the particle generation
    layers = np.linspace(1,0.6,np.random.randint(5,8)) # stacks of particles
    layers = layers[::-1]
    # ground params
    lx = 5 # width of the domain
    ly = 3 # height of the domain
    lz = 0.15 # depth of the domain

    # if distribution is in cubic lattice
    part_dist = 0.1 # particle spacing
    n_layers = int(12*1.8/0.4/len(layers)) # number of layers in a stack of particles

    # friction params
    pp = 0.3 # particle-particle friction
    pw = 0.8 # particle-wall friction


    # entities_to_gen
    entities_to_gen = ['wall', 'ballast']

    # material dict and container
    dict_mat = {'TDURx': {'materialType':'RIGID', 'density':2700.}}
    mats = create_materials(dict_mat)

    # model dict and container
    dict_mod = {'rigid': {'physics':'MECAx', 'element':'Rxx3D', 'dimension':dim}}
    mods = create_models(dict_mod)

    ## Avatar generation
    bodies = pre.avatars()

    ## wall avatar generation if wall in entities_to_gen
    if 'wall' in entities_to_gen:
        wall_bodies = wall_generator(lx, ly, lz, mats['TDURx'], mods['rigid'])
        # impose driven dofs
        [wall_body.imposeDrivenDof(component=[1,2,3,4,5,6],dofty='vlocy') for wall_body in wall_bodies]
        bodies+=wall_bodies

    ## Ballast generation if ballast in entities_to_gen
    if 'ballast' in entities_to_gen:
        ballast_bodies, total_paricles, par_char = ballast_generator(max_particles, Rmin, Rmax, Px, Py, Pz, layers, mats['TDURx'], mods['rigid'], ptype, min_vert, max_vert, seed=seed, part_dist=part_dist, nb_layers=n_layers, gen_type=gen_type)
        bodies+=ballast_bodies
        track_every = int(total_paricles/50)

    dict_tact = {'iqsc0': {'law':'IQS_CLB', 'fric':pp},
                'iqsc1': {'law':'IQS_CLB', 'fric':pw}}
    
    tacts = create_tact_behavs(dict_tact)
    dict_see = {'vpp': {'CorpsCandidat':'RBDY3', 'candidat':ptype,'colorCandidat':'BLUEx','behav':tacts['iqsc0'],
                        'CorpsAntagoniste':'RBDY3', 'antagoniste':ptype,'colorAntagoniste':'BLUEx',
                        'alert':0.001},
                'vpw': {'CorpsCandidat':'RBDY3', 'candidat':ptype,'colorCandidat':'BLUEx','behav':tacts['iqsc1'],
                        'CorpsAntagoniste':'RBDY3', 'antagoniste':'PLANx','colorAntagoniste':'WALLx',
                        'alert':0.001}}
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
    par_char_f = 'particle_characteristics.dat'
    with open(os.path.join(par_dir, par_char_f), 'w') as f:
        for row in zip(*par_char.values()):
            f.write('   '.join(str(x) for x in row)+'\n')
    return dict_sample


def random_ballast_test_sncf(par_dir, seed=687, visu=False, step=1):
    dim = 3

    # ground params
    lx = 6
    ly = 6
    lz = 0.10

    # ballast params
    #ballast_bib, layers, nb_particles, lx, ly, lz, mat, mod, Rmin=0.3, Rmax=0.4
    ballast_bib = 'BIBLIGRAINS/BIBLIGRAINS.DAT'
    #layers = np.linspace(1,0.5, np.random.randint(20,35))
    layers = [1]
    layers = layers[::-1]
    nb_particles = 1500
    Rmin = 1.4044198827808083E-002
    Rmax = 5.7470355839992146E-002
    Px = np.random.uniform(1.96,2.5) # width of the particle generation
    Py = np.random.uniform(0.4,0.6) # height of the particle generation
    Pz = np.random.uniform(0.18,0.25) # depth of the particle generation

    # friction params
    pp = 0.5 # particle-particle friction
    pw = 0.8 # particle-wall friction

    # entities_to_gen
    entities_to_gen = ['wall', 'ballast']

    # material dict and container
    dict_mat = {'TDURx': {'materialType':'RIGID', 'density':2700.}}
    mats = create_materials(dict_mat)

    # model dict and container
    dict_mod = {'rigid': {'physics':'MECAx', 'element':'Rxx3D', 'dimension':dim}}
    mods = create_models(dict_mod)

    ## Avatar generation
    bodies = pre.avatars()

    ## wall avatar generation if wall in entities_to_gen
    if 'wall' in entities_to_gen:
        wall_bodies = wall_generator(lx, ly, lz, mats['TDURx'], mods['rigid'])
        # impose driven dofs
        [wall_body.imposeDrivenDof(component=[1,2,3,4,5,6],dofty='vlocy') for wall_body in wall_bodies]
        bodies+=wall_bodies

    ## Ballast generation if ballast in entities_to_gen
    if 'ballast' in entities_to_gen:
        ballast_bodies, total_paricles, par_char = ballast_generator_custom(ballast_bib, layers, nb_particles, Px, Py, Pz, mats['TDURx'], mods['rigid'], Rmin=Rmin, Rmax=Rmax)
        bodies+=ballast_bodies
        track_every = 50

    # tact dict and container
    dict_tact = {'iqsc0': {'law':'IQS_CLB', 'fric':pp},
                'iqsc1': {'law':'IQS_CLB', 'fric':pw}}
    tacts = create_tact_behavs(dict_tact)

    # see dict and container
    dict_see = {'vpp': {'CorpsCandidat':'RBDY3', 'candidat':'POLYR','colorCandidat':'BLUEx','behav':tacts['iqsc0'],
                        'CorpsAntagoniste':'RBDY3', 'antagoniste':'POLYR','colorAntagoniste':'BLUEx',
                        'alert':0.001},
                'vpw': {'CorpsCandidat':'RBDY3', 'candidat':'POLYR','colorCandidat':'BLUEx','behav':tacts['iqsc1'],
                        'CorpsAntagoniste':'RBDY3', 'antagoniste':'PLANx','colorAntagoniste':'WALLx',
                        'alert':0.001}}
    svs = create_see_tables(dict_see)

    # post dict and container
    post_dict = {'CONTACT FORCE DISTRIBUTION': {'step':step, 'val':1},
                'COORDINATION NUMBER': {'step':step},
                'BODY TRACKING': {'step':step, 'rigid_set':ballast_bodies[::track_every]},
                'TORQUE EVOLUTION': {'step':step, 'rigid_set':ballast_bodies[::track_every]},
                'AVERAGE VELOCITY EVOLUTION': {'step':step, 'color':'BLUEx'},
                'KINETIC ENERGY': {'step':step},
                'DISSIPATED ENERGY': {'step':step}
                }
    post = create_postpro_commands(post_dict)

    # visu
    if visu: pre.visuAvatars(bodies)

    # write datbox
    pre.writeBodies(bodies,chemin=os.path.join(par_dir,'DATBOX'))
    pre.writeBulkBehav(mats,chemin=os.path.join(par_dir,'DATBOX'),dim=3)
    pre.writeTactBehav(tacts,svs,chemin=os.path.join(par_dir,'DATBOX'))
    pre.writeDrvDof(bodies,chemin=os.path.join(par_dir,'DATBOX'))
    pre.writeDofIni(bodies,chemin=os.path.join(par_dir,'DATBOX'))
    pre.writeVlocRlocIni(chemin=os.path.join(par_dir,'DATBOX'))
    pre.writePostpro(commands=post, parts=bodies, path=os.path.join(par_dir,'DATBOX'))


    dict_sample = {'Px':Px, 'Py':Py, 'Pz':Pz, 'Total Particles':total_paricles, 'Track Every': track_every, 'Layers':len(layers)}
    par_char_f = 'particle_characteristics.dat'
    with open(os.path.join(par_dir, par_char_f), 'w') as f:
        for row in zip(*par_char.values()):
            f.write('   '.join(str(x) for x in row)+'\n')
    return dict_sample


def random_compacted_sncf(par_dir, seed=687, visu=False, step=1, args=None):
    dim = 3

    # ground params
    lx = 3
    ly = 2
    lz = 0.05

    # a trapezoid for shaping the ballast
    txb = 0.2
    txt = 0.4
    ty = 0.8
    tz = 0.4

    ## ballast params
    ballast_bib = 'BIBLIGRAINS/BIBLIGRAINS.DAT'
    #layers = np.linspace(1,0.5, np.random.randint(20,35))
    if args.layers==1:layers = [1] 
    else:
        nb_layers  = np.random.randint(args.nb_layers_min,args.nb_layers_max) 
        layers = np.linspace(1,args.layers, nb_layers)
    layers = layers[::-1]
    nb_particles = 1500
    Rmin = 1.4044198827808083E-002
    Rmax = 5.7470355839992146E-002
    Px = np.random.uniform(1.96,2.5) # width of the particle generation
    Py = np.random.uniform(0.4,0.6) # height of the particle generation
    Pz = np.random.uniform(0.18,0.25) # depth of the particle generation

    # friction params
    pp = 0.5 # particle-particle friction
    pw = 0.8 # particle-wall friction
    pt = 0.5 # particle-trapezoid friction


    # material dict and container
    dict_mat = {'TDURx': {'materialType':'RIGID', 'density':2700.}}
    mats = create_materials(dict_mat)

    # model dict and container
    dict_mod = {'rigid': {'physics':'MECAx', 'element':'Rxx3D', 'dimension':dim}}
    mods = create_models(dict_mod)

    ## Avatar generation
    bodies = pre.avatars()

    ## wall avatar generation if wall in entities_to_gen
    if args.wall:
        wall_bodies = wall_generator(lx, ly, lz, mats['TDURx'], mods['rigid'])
        # impose driven dofs
        [wall_body.imposeDrivenDof(component=[1,2,3,4,5,6],dofty='vlocy') for wall_body in wall_bodies]
        bodies+=wall_bodies

    if args.trap:
        trapezoid = trapezoid_generator(txb,txt, ty, tz, mats['TDURx'], mods['rigid'])
        trapezoid.translate(dy=lz)
        bodies+=trapezoid

    if args.ballast:
        ballast_bodies, total_paricles, par_char = ballast_generator_custom(ballast_bib, layers, nb_particles, Px, Py, Pz, mats['TDURx'], mods['rigid'], Rmin=Rmin, Rmax=Rmax)
        bodies+=ballast_bodies
        track_every = 50

    # tact dict and container
    dict_tact = {'iqsc0': {'law':'IQS_CLB', 'fric':pp},
                 'iqsc1': {'law':'IQS_CLB', 'fric':pw},
                'iqsc2': {'law':'IQS_CLB', 'fric':pt}}
    tacts = create_tact_behavs(dict_tact)

    # see dict and container
    dict_pp = {'vpp': {'CorpsCandidat':'RBDY3', 'candidat':'POLYR','colorCandidat':'BLUEx','behav':tacts['iqsc0'],
                        'CorpsAntagoniste':'RBDY3', 'antagoniste':'POLYR','colorAntagoniste':'BLUEx',
                        'alert':0.001}}
    dict_pw = {'vpw': {'CorpsCandidat':'RBDY3', 'candidat':'POLYR','colorCandidat':'BLUEx','behav':tacts['iqsc1'],
                        'CorpsAntagoniste':'RBDY3', 'antagoniste':'PLANx','colorAntagoniste':'WALLx',
                        'alert':0.001}}
    dict_pt = {'vpt': {'CorpsCandidat':'RBDY3', 'candidat':'POLYR','colorCandidat':'BLUEx','behav':tacts['iqsc2'],
                        'CorpsAntagoniste':'RBDY3', 'antagoniste':'RBDY3','colorAntagoniste':'BLUEx',
                        'alert':0.001}}
    dict_see = {}
    if args.wall: dict_see.update(dict_pp)
    if args.trap: dict_see.update(dict_pt)
    if args.ballast: dict_see.update(dict_pw)
    svs = create_see_tables(dict_see)

    # post dict and container
    post_dict = {'CONTACT FORCE DISTRIBUTION': {'step':step, 'val':1},
                'COORDINATION NUMBER': {'step':step},
                'BODY TRACKING': {'step':step, 'rigid_set':ballast_bodies[::track_every]},
                'TORQUE EVOLUTION': {'step':step, 'rigid_set':ballast_bodies[::track_every]},
                'AVERAGE VELOCITY EVOLUTION': {'step':step, 'color':'BLUEx'},
                'KINETIC ENERGY': {'step':step},
                'DISSIPATED ENERGY': {'step':step}
                }
    post = create_postpro_commands(post_dict)

    # visu
    if visu: pre.visuAvatars(bodies)

    # write datbox
    pre.writeDatbox(dim,mats,mods,bodies,tacts,svs,post=post, datbox_path=os.path.join(par_dir,'DATBOX'))
    params = {'nb_layers': nb_layers, 'ratio': args.layers}
    return params

    





