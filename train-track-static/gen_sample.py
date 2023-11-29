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
Rmax = 3 # maximum radius of particles

Px = 45*Rmax # width of the particle generation
Py = 25*Rmax # height of the particle generation
Pz = 5*Rmax # depth of the particle generation

lx = 65*Rmax # width of the domain
ly = 55*Rmax # height of the domain
lz = 45*Rmax # depth of the domain

nb_particles = 200 # number of particles
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
plex = pre.material(name='PLEXx', materialType='RIGID', density=500.)
mats.addMaterial(tdur,plex)


## create a model of rigid
mod = pre.model(name='rigid',physics='MECAx',element='Rxx3D',dimension=dim)
mods.addModel(mod)

## creation of the walls

down = pre.rigidPlan(axe1=lx/2.,axe2=lz/2.,axe3=Rmax/2.,center=[0.,0.,0.],
                   material=tdur,model=mod,color='WALLx')

#down.rotate(description='axis',alpha=math.pi/2.,axis=[1.,0.,0.],center=down.nodes[1].coor)
#down.translate(dy=-ly/2.,dz=lz/2.)
bodies.addAvatar(down)

## creation of spheres grains and stuff
radii = pre.granulo_Random(nb_particles,Rmin,Rmax)

layers = [1, 0.75, 0.5]
layersx = [1, 0.99, 0.98]
for j in range(len(layers)):
  [nb_rem,coors] = pre.depositInBox3D(radii,Px,Py*layers[j],Pz)
  for i in range(nb_rem):
      if ptype == 'POLYR':
          body = pre.rigidPolyhedron(radius=radii[i], center=coors[3*i : 3*(i+1)], nb_vertices=6, generation_type='random',model=mod,
                            material=plex, color='BLUEx')
      else:
          body = pre.rigidSphere(r=radii[i],center=coors[3*i:3*(i+1)],model=mod,
                            material=plex,color='BLUEx')
          bodies.addAvatar(body)

      body.translate(dz=Pz*(j)/2. + Rmax/2.)
      # body.rotate(description='Euler',phi=2.*math.pi*np.random.rand(1),psi=2.*math.pi*np.random.rand(1),
      #             center=body.nodes[1].coor)
      #body.imposeInitValue(component=2,value=-20.0) # velocity imposed on the particle in y direction
      #body.imposeInitValue(component=1,value=-20.0) # velocity imposed on the particle in x direction
      bodies.addAvatar(body)

rail = pre.rigidPlan(axe1=Px/2.1,axe2=Rmax,axe3=Rmax/2.,center=[0.,0.,0.],
                    material=tdur,model=mod,color='REDxx')
rail.rotate(description='axis',alpha=math.pi/2.,axis=[1.,0.,0.],center=rail.nodes[1].coor)
rail.translate(dz=Pz*(j),dy = -3*Rmax)
bodies.addAvatar(rail)
rail2 = pre.rigidPlan(axe1=Px/2.1,axe2=Rmax,axe3=Rmax/2.,center=[0.,0.,0.],
                    material=tdur,model=mod,color='REDxx')
rail2.rotate(description='axis',alpha=math.pi/2.,axis=[1.,0.,0.],center=rail2.nodes[1].coor)
rail2.translate(dz=Pz*(j),dy = 3*Rmax)

bodies.addAvatar(rail2)

## impose 0 velcities on walls
down.imposeDrivenDof(component=[1,2,3,4,5,6],dofty='vlocy')




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
vpw2 = pre.see_table(CorpsCandidat='RBDY3', candidat='PLANx',colorCandidat='REDxx',behav=pwi,
                    CorpsAntagoniste='RBDY3', antagoniste='PLANx',colorAntagoniste='WALLx',
                      alert=0.1)
vpw3 = pre.see_table(CorpsCandidat='RBDY3', candidat=ptype,colorCandidat='BLUEx',behav=pwi,
                    CorpsAntagoniste='RBDY3', antagoniste='DISKx',colorAntagoniste='REDxx',
                      alert=0.1)

svs+=vpp;svs+=vpw ;svs+=vpw2;svs+=vpw3

post = pre.postpro_commands()
body_track = pre.postpro_command(name='CONTACT FORCE DISTRIBUTION', step = 1, val=1)
body_coord = pre.postpro_command(name='COORDINATION NUMBER', step = 1) 
post.addCommand(body_track)
post.addCommand(body_coord)
pre.visuAvatars(bodies)
pre.writeDatbox(dim,mats,mods,bodies,tacts,svs,post=post)



