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
    layers = np.linspace(1,0.5, np.random.randint(20,30))
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
                #'BODY TRACKING': {'step':step, 'rigid_set':ballast_bodies[::track_every]},
                #'TORQUE EVOLUTION': {'step':step, 'rigid_set':ballast_bodies[::track_every]},
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
    # ground params
    lx = 6
    ly = 6
    lz = 0.10

    # a trapezoid for shaping the ballast
    txb = 0.2
    txt = 0.4
    ty = 0.8
    tz = 0.4

    ## ballast params
    ballast_bib = 'BIBLIGRAINS/BIBLIGRAINS.DAT'
    #layers = np.linspace(1,0.5, np.random.randint(20,35))
    if args.layers==1:layers = [1]; nb_layers = 1
    else:
        nb_layers  = np.random.randint(args.nb_layers_min,args.nb_layers_max) 
        layers = np.linspace(1,args.layers, nb_layers)
    layers = layers[::-1]
    nb_particles = 8000
    Rmin = 1.4044198827808083E-002
    Rmax = 5.7470355839992146E-002
    Px = np.random.uniform(1.96,2.5) # width of the particle generation
    Py = np.random.uniform(0.4,0.6) # height of the particle generation
#    Pz = np.random.uniform(0.18,0.25) # depth of the particle generation
    Pz = np.random.uniform(3,4) # depth of the particle generation


    # friction params
    pp = 0.3 # particle-particle friction
    pw = 0.2 # particle-wall friction
    pt = 0.5 # particle-trapezoid friction

    # velocity evolution

    t0=1.
    t1 =2.
    vx =0.
    pre.writeEvolution(f=imposedVx, instants=np.linspace(0., 2*t1, 1000) ,path=par_dir+'DATBOX/', name='vx.dat')
    pre.writeEvolution(f=imposedVxneg, instants=np.linspace(0., 2*t1, 1000) ,path=par_dir+'DATBOX/', name='vx2.dat')

    pre.writeEvolution(f=imposedVz, instants=np.linspace(0., 2*t1, 1000) ,path=par_dir+'DATBOX/', name='vz.dat')
    


    # material dict and container
    dict_mat = {'TDURx': {'materialType':'RIGID', 'density':2700.},
                'TDURT': {'materialType':'RIGID', 'density':8000.}}
    mats = create_materials(dict_mat)

    # model dict and container
    dict_mod = {'rigid': {'physics':'MECAx', 'element':'Rxx3D', 'dimension':dim}}
    mods = create_models(dict_mod)

    ## Avatar generation
    bodies = pre.avatars()

    ## wall avatar generation if wall in entities_to_gen
    if args.wall:
        wall_bodies = wall_generator(lx, ly, lz, mats['TDURx'], mods['rigid'])
        #wall_mesh = pre.readMesh('floor.msh', dim=3)
        #wall_bodies = pre.volumicMeshToRigid3D(wall_mesh, material=mats['TDURx'], model=mods['rigid'], color='WALLx')
        # impose driven dofs
        [wall_body.imposeDrivenDof(component=[1,2,3,4,5,6],dofty='vlocy') for wall_body in wall_bodies]
        #wall_bodies.imposeDrivenDof(component=[1,2,3,4,5,6],dofty='vlocy')
        bodies+=wall_bodies

        ## create boundaries
        b1 = pre.rigidPlan(lx/2, 0.3,0.05,center=[0.,0.,0.], material=mats['TDURx'], model=mods['rigid'], color='WALLx')
        #b1.rotate(description='axis', alpha=np.deg2rad(90), axis=[0,0,1])   
        b1.rotate(description='axis', alpha=np.deg2rad(90), axis=[1,0,0])
        b1.translate(dz = 0.1 + (0.3/2.), dy = -ly/2.)
        b1.imposeDrivenDof(component=[1,2,3,4,5,6],dofty='vlocy')
        bodies+=b1   
        b2 = pre.rigidPlan(lx/2, 0.3,0.05,center=[0.,0.,0.], material=mats['TDURx'], model=mods['rigid'], color='WALLx')
        b2.rotate(description='axis', alpha=np.deg2rad(90), axis=[1,0,0])
        b2.translate(dz = 0.1 + (0.3/2.), dy = ly/2.)
        b2.imposeDrivenDof(component=[1,2,3,4,5,6],dofty='vlocy')
        bodies+=b2
        b3 = pre.rigidPlan(0.3,ly/2.,0.05,center=[0.,0.,0.], material=mats['TDURx'], model=mods['rigid'], color='WALLx')
        b3.rotate(description='axis', alpha=np.deg2rad(90), axis=[0,1,0])
        b3.translate(dz = 0.1 + (0.3/2.), dx = lx/2.)
        b3.imposeDrivenDof(component=[1,2,3,4,5,6],dofty='vlocy')
        bodies+=b3
        b4 = pre.rigidPlan(0.3,ly/2.,0.05,center=[0.,0.,0.], material=mats['TDURx'], model=mods['rigid'], color='WALLx')
        b4.rotate(description='axis', alpha=np.deg2rad(90), axis=[0,1,0])
        b4.translate(dz = 0.1 + (0.3/2.), dx = -lx/2.)
        b4.imposeDrivenDof(component=[1,2,3,4,5,6],dofty='vlocy')
        bodies+=b4


        







    if args.trap:
        #trapezoid = trapezoid_generator(txb,txt, ty, tz, mats['TDURx'], mods['rigid'])
        #mesh from file
        meshed_trap = pre.readMesh('tri.msh', dim=3)
        trapezoid = pre.volumicMeshToRigid3D(meshed_trap, material=mats['TDURT'], model=mods['rigid'], color='BLUEx')
        trapezoid.imposeDrivenDof(component=[1,2,4,5,6],dofty='vlocy')
        trapezoid.imposeDrivenDof(component=2,dofty='vlocy',description='evolution',evolutionFile='vx2.dat')
        
        trapezoid.translate(dz = 0.05, dy = -1.)
        bodies+=trapezoid
        # create a second one and mirrored
        trapezoid2 = pre.volumicMeshToRigid3D(meshed_trap, material=mats['TDURT'], model=mods['rigid'], color='BLUEx')
        trapezoid2.imposeDrivenDof(component=[1,2,4,5,6],dofty='vlocy')
        trapezoid2.imposeDrivenDof(component=2,dofty='vlocy',description='evolution',evolutionFile='vx.dat')
        trapezoid2.translate(dz = 0.05, dy = -1.)
        trapezoid2.rotate(description='axis', alpha=3.14, axis=[0,0,1])
        bodies+=trapezoid2

        # top compactor
        meshed_top = pre.readMesh('compact_top.msh', dim=3)
        top_compactor = pre.volumicMeshToRigid3D(meshed_top, material=mats['TDURT'], model=mods['rigid'], color='BLUEx')
        top_compactor.imposeDrivenDof(component=[2,3,4,5,6],dofty='vlocy')
        top_compactor.imposeDrivenDof(component=3,dofty='vlocy',description='evolution',evolutionFile='vz.dat')

        top_compactor.translate(dz = 3)
        bodies+=top_compactor
        



    if args.ballast:
        ballast_bodies, total_paricles, par_char = ballast_generator_custom(ballast_bib, layers, nb_particles, Px, Py, Pz, mats['TDURx'], mods['rigid'], Rmin=Rmin, Rmax=Rmax)
        bodies+=ballast_bodies
        track_every = 50

    # tact dict and container
    dict_tact = {'iqsc0': {'law':'IQS_CLB', 'fric':pp},
                 'iqsc1': {'law':'IQS_CLB', 'fric':pw}}
    tacts = create_tact_behavs(dict_tact)

    # see dict and container
    dict_pp = {'vpp': {'CorpsCandidat':'RBDY3', 'candidat':'POLYR','colorCandidat':'BLUEx','behav':tacts['iqsc0'],
                        'CorpsAntagoniste':'RBDY3', 'antagoniste':'POLYR','colorAntagoniste':'BLUEx',
                        'alert':0.0001}}
    dict_pw = {'vpw': {'CorpsCandidat':'RBDY3', 'candidat':'POLYR','colorCandidat':'BLUEx','behav':tacts['iqsc1'],
                        'CorpsAntagoniste':'RBDY3', 'antagoniste':'PLANx','colorAntagoniste':'WALLx',
                        'alert':0.0001}}
    # dict_ma = {'vma': {'CorpsCandidat':'RBDY3', 'candidat':'POLYR','colorCandidat':'REDx','behav':tacts['iqsc3'],
    #                     'CorpsAntagoniste':'RBDY3', 'antagoniste':'PLANx','colorAntagoniste':'WALLx',
    #                     'alert':0.001}}
    # dict_mb = { 'vma': {'CorpsCandidat':'RBDY3', 'candidat':'POLYR','colorCandidat':'BLUEx','behav':tacts['iqsc4'],
    #                     'CorpsAntagoniste':'RBDY3', 'antagoniste':'POLYR','colorAntagoniste':'REDxx',
    #                     'alert':0.001}}
    dict_see = {}
    if args.wall: dict_see.update(dict_pp)
    if args.ballast: dict_see.update(dict_pw)
    #if args.trap: dict_see.update(dict_ma); #dict_see.update(dict_mb)
    if args.trap & ~args.ballast: dict_see.update(dict_pw)

    svs = create_see_tables(dict_see)

    # post dict and container
    post_dict = {'CONTACT FORCE DISTRIBUTION': {'step':step, 'val':1},
                'COORDINATION NUMBER': {'step':step},
                #'BODY TRACKING': {'step':step, 'rigid_set':ballast_bodies[::track_every]},
                #'TORQUE EVOLUTION': {'step':step, 'rigid_set':ballast_bodies[::track_every]},
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

    if args.ballast: 
        dict_sample = {'Px':Px, 'Py':Py, 'Pz':Pz, 'Total Particles':total_paricles, 'Track Every': track_every, 'Layers':len(layers)}
        par_char_f = 'particle_characteristics.dat'
        with open(os.path.join(par_dir, par_char_f), 'w') as f:
            for row in zip(*par_char.values()):
                f.write('   '.join(str(x) for x in row)+'\n')
        return dict_sample, params

    else: return None, params

def closet_ballast(par_dir, seed=687, visu=False, step=1, args=None):
    dim=3
    # box dimensions
    thick = 0.01
    lengthb = 1.
    widthb = 1.
    heightb = 1.

    # ballast params
    ballast_bib = 'BIBLIGRAINS/BIBLIGRAINS.DAT'
    dgrid = 0.1
    nbx = int(lengthb/dgrid)
    nby = int(widthb/dgrid)
    nlayer = int(20*1.8/0.4)



    
    # material dict and container
    dict_mat = {'TDURx': {'materialType':'RIGID', 'density':2700.}}
    mats = create_materials(dict_mat)

    # model dict and container
    dict_mod = {'rigid': {'physics':'MECAx', 'element':'Rxx3D', 'dimension':dim}}
    mods = create_models(dict_mod)

    ## Avatar generation

    bodies = pre.avatars()
    if args.ballast:

        bodies, z = ballast_generator_closet(ballast_bib, nbx, nby, nlayer, dgrid, lengthb, widthb, mats['TDURx'], mods['rigid'])
    
    if args.closet:
        dict = {
        'left':{
            'axe1': lengthb/2 + 0.1,
            'axe2': heightb/2,
            'axe3': thick/2,
            'x': 0.,
            'y': -widthb/2 - thick/2,
            'z': heightb/2,
            'rotate':{'theta': -0.5*np.pi},
            'imposeDrivenDof': {'component':[1,2,3,4,5,6]}
        },
        'right':{
            'axe1': lengthb/2 + 0.1,
            'axe2': heightb/2,
            'axe3': thick/2,
            'x': 0.,
            'y': widthb/2 + thick/2,
            'z': heightb/2,
            'rotate':{'theta': 0.5*np.pi},
            'imposeDrivenDof': {'component':[1,2,3,4,5,6]}
        },
        'front':{
            'axe1': heightb/2 ,
            'axe2': widthb/2 + 0.1,
            'axe3': thick/2,
            'x': lengthb/2 + thick/2,
            'y': 0.,
            'z': heightb/2,
            'rotate':{'description':'axis', 'alpha': -0.5*np.pi, 'axis':[0.,1.,0.]},
            'imposeDrivenDof': {'component':[1,2,3,4,5,6]}
        },
        'back':{
            'axe1': heightb/2 ,
            'axe2': widthb/2 + 0.1,
            'axe3': thick/2,
            'x': -lengthb/2 - thick/2,
            'y': 0.,
            'z': heightb/2,
            'rotate':{'description':'axis', 'alpha': 0.5*np.pi, 'axis':[0.,1.,0.]},
            'imposeDrivenDof': {'component':[1,2,3,4,5,6]}
        },
        'bottom':{
            'axe1': lengthb/2+0.1,
            'axe2': widthb/2+0.1,
            'axe3': thick/2,
            'x': 0.,
            'y': 0.,
            'z': -thick/2.,
            'imposeDrivenDof': {'component':[1,2,3,4,5,6]}
        },
        'top':{
            'axe1': lengthb/2+0.1,
            'axe2': widthb/2+0.1,
            'axe3': thick/2,
            'x': 0.,
            'y': 0.,
            'z': max(z)+dgrid/2.,
            'rotate':{'theta': np.pi},
            'imposeDrivenDof': {'component':[1,2,4,5,6]}
        }
        }
    bodies_plates = plate_definition(dict, mat=mats['TDURx'], mod=mods['rigid'])
    if args.visu:pre.visuAvatars(bodies)
    bodies+=bodies_plates

    # tact dict and container
    dict_tact = {'iqsc0': {'law':'IQS_CLB', 'fric':0.0},
                 'iqsc1': {'law':'IQS_CLB', 'fric':0.1},
                 'iqsc3': {'law':'IQS_CLB', 'fric':0.8}}
    tacts = create_tact_behavs(dict_tact)

    # see dict and container
    dict_pp = {'vpp': {'CorpsCandidat':'RBDY3', 'candidat':'POLYR','colorCandidat':'BLUEx','behav':tacts['iqsc1'],
                        'CorpsAntagoniste':'RBDY3', 'antagoniste':'POLYR','colorAntagoniste':'BLUEx',
                        'alert':0.001}}

    dict_pw = {'vpw': {'CorpsCandidat':'RBDY3', 'candidat':'POLYR','colorCandidat':'BLUEx','behav':tacts['iqsc1'],
                        'CorpsAntagoniste':'RBDY3', 'antagoniste':'PLANx','colorAntagoniste':'VERTx',
                        'alert':0.001}}
    
    dict_see = {}
    if args.ballast: dict_see.update(dict_pp)
    if args.closet: dict_see.update(dict_pw)

    svs = create_see_tables(dict_see)
    post_dict = {'CONTACT FORCE DISTRIBUTION': {'step':step, 'val':1},
                'COORDINATION NUMBER': {'step':step},
                'AVERAGE VELOCITY EVOLUTION': {'step':step, 'color':'BLUEx'},
                'KINETIC ENERGY': {'step':step},
                'DISSIPATED ENERGY': {'step':step}
                }
    post = create_postpro_commands(post_dict)
    pre.writeDatbox(dim,mats,mods,bodies,tacts,svs,post=post, datbox_path=os.path.join(par_dir,'DATBOX'))
    return None








t0=2.
t1 =3.
vx =-0.3

def imposedVx(t):
    # 0 until t0
    if t <= t0:
        return 0.
    # linear growing between [t0, t1]
    elif t > t0 and t <= t1:
        return vx
    # leak = 0
    elif t > t1 and t <= t1 + 0.1:
        return 0.
    # constant value afterward
    else:
        return -vx*2.
    
def imposedVxneg(t):
    # 0 until t0
    if t <= t0:
        return 0.
    # linear growing between [t0, t1]
    elif t > t0 and t <= t1:
        return -vx
    elif t > t1 and t <= t1 + 0.1:
        return 0.
    # constant value afterward
    else:
        return vx*2.
    
vz=2.2
t2 = 2.5
def imposedVz(t):
    # 0 until t0
    if t <= 1.0:
        return 0.
    # linear growing between [t0, t1]
    elif t > 1. and t <= t2:
        return -vz
    elif t > t2 and t <= t2 + 0.1:
        return 0.
    else:
        return vz/100.

        