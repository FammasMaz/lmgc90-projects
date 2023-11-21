import os, sys
import numpy as np
import math
import shutil

from pylmgc90 import pre

if not os.path.isdir('./DATBOX'):
  os.mkdir('./DATBOX')
if os.path.isdir('./DISPLAY'):
  shutil.rmtree('./DISPLAY')
if os.path.isdir('./POSTPRO'):
  shutil.rmtree('./POSTPRO')
if os.path.isdir('./OUTBOX'):
  shutil.rmtree('./OUTBOX')


# 3D firstfall
dim = 3

## geometric params
ptype = 'POLYR' # particle type
Rmin = 2 # minimum radius of particles
Rmax = 5 # maximum radius of particles

Px = 30*Rmax # width of the particle generation
Py = 5*Rmax # height of the particle generation
Pz = 10*Rmax # depth of the particle generation

lx = 50*Rmax # width of the domain
ly = 30*Rmax # height of the domain
lz = 30*Rmax # depth of the domain

nb_particles = 500 # number of particles
pp = 0.3 # particle-particle friction
pw = 0.5 # particle-wall friction


## containers definitions:
bodies = pre.avatars() # container for bodies
mats = pre.materials() # container for materials
mods = pre.models() # container for models
svs = pre.see_tables() # container for see tables
tacts = pre.tact_behavs() # container for tact behaviors

## create materials
# print(pre.config.lmgc90dicts.bulkBehavOptions['RIGID'])
tdur = pre.material(name='TDURx', materialType='RIGID', density=1000.)
plex = pre.material(name='PLEXx', materialType='RIGID', density=5.)
mats.addMaterial(tdur,plex)

## create a model of rigid
mod = pre.model(name='rigid',physics='MECAx',element='Rxx3D',dimension=dim)
mods.addModel(mod)

## creation of the walls
back = pre.rigidPlan(axe1=lx/2.,axe2=ly/2.,axe3=Rmax/2.,center=[0.,0.,-Rmax],
                     material=tdur,model=mod,color='WALLx')
left = pre.rigidPlan(axe1=ly/2.,axe2=lz/2.,axe3=Rmax/2.,center=[0.,0.,-Rmax],
                     material=tdur,model=mod,color='WALLx')
down = pre.rigidPlan(axe1=lx/2.,axe2=lz/2.,axe3=Rmax/2.,center=[0.,0.,-Rmax],
                   material=tdur,model=mod,color='WALLx')

left.rotate(description='axis',alpha=math.pi/2.,axis=[0.,1.,0.],center=left.nodes[1].coor)
left.translate(dx=-lx/2.,dz=ly/2.)
down.rotate(description='axis',alpha=math.pi/2.,axis=[1.,0.,0.],center=down.nodes[1].coor)
down.translate(dy=-ly/2.,dz=lz/2.)
bodies.addAvatar(back);bodies.addAvatar(left);bodies.addAvatar(down)

## creation of spheres grains and stuff
radii = pre.granulo_Random(nb_particles,Rmin,Rmax)
[nb_rem,coors] = pre.depositInBox3D(radii,Px,Py,Pz)

for i in range(nb_rem):
    if ptype == 'POLYR':
        body = pre.rigidPolyhedron(radius=radii[i], center=coors[3*i : 3*(i+1)], nb_vertices=int(np.random.uniform(5,9)), model=mod,
                          material=plex, color='BLUEx')
    else:
        body = pre.rigidSphere(r=radii[i],center=coors[3*i:3*(i+1)],model=mod,
                            material=plex,color='BLUEx')
    body.translate(dx=-8*Rmax,dz=40)
    body.imposeInitValue(component=2,value=-20.0) # velocity imposed on the particle in y direction
    body.imposeInitValue(component=1,value=-20.0) # velocity imposed on the particle in x direction
    bodies.addAvatar(body)

pre.visuAvatars(bodies)

## impose 0 velcities on walls
left.imposeDrivenDof(component=[1,2,3,4,5,6],dofty='vlocy')
down.imposeDrivenDof(component=[1,2,3,4,5,6],dofty='vlocy')
back.imposeDrivenDof(component=[1,2,3,4,5,6],dofty='vlocy')

## define interatactions4
ppi = pre.tact_behav(name='iqsc0',law='IQS_CLB',fric=pp)
pwi = pre.tact_behav(name='iqsc1',law='IQS_CLB',fric=pw)

tacts+=ppi;tacts+=pwi

## see table
vpp = pre.see_table(CorpsCandidat='RBDY3', candidat=ptype,colorCandidat='BLUEx',behav=ppi,
                    CorpsAntagoniste='RBDY3', antagoniste=ptype,colorAntagoniste='BLUEx',
                      alert=0.1)
vpw = pre.see_table(CorpsCandidat='RBDY3', candidat=ptype,colorCandidat='BLUEx',behav=pwi,
                    CorpsAntagoniste='RBDY3', antagoniste='PLANx',colorAntagoniste='WALLx',
                      alert=0.1)

svs+=vpp;svs+=vpw

post = pre.postpro_commands()
pre.writeDatbox(dim,mats,mods,bodies,tacts,svs,post=post)




