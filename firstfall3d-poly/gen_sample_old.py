# import modules
from __future__ import print_function
import os, sys
import numpy as np
import math
import matplotlib.pyplot as plt
from pylmgc90 import pre

## defining variables
pp = 0.1 # particle-particle friction
pw = 0.5 # particle-wall friction

Rmin = 1.0 # minimum radius of particles
Rmax = 2.5 # maximum radius of particles

Px = 30*Rmax # width of the particle generation
Py = 5*Rmax # height of the particle generation
Pz = 10*Rmax # depth of the particle generation


lx = 50*Rmax # width of the domain
ly = 30*Rmax # height of the domain
lz = 30*Rmax # depth of the domain

n = 200 # number of particles

dim = 3 # dimension of the problem

## container defintion
bodies = pre.avatars() # container for bodies
mat = pre.materials() # container for materials
svs = pre.see_tables() # container for see tables
tacts = pre.tact_behavs() # container for tact behaviors

## material definition
tdur = pre.material(name='TDURx', materialType='RIGID', density=1000.)
plex = pre.material(name='PLEXx', materialType='RIGID', density=5.)
mat.addMaterial(tdur,plex)

## model definition
mod = pre.model(name='rigid', physics='MECAx', element='Rxx3D',dimension=dim)

## particle generation
radii = pre.granulo_Random(n, Rmin, Rmax) # random radii
[nb_rem, coor] = pre.depositInBox3D(radii, Px, Py, Pz) # deposit particles in a box

# balls avatar generation
for i in range(nb_rem):
    #body = pre.rigidDisk(r=radii[i], center=coor[2*i : 2*(i+1)],model=mod,material=plex,color='BLUEx')
    body = pre.rigidPolyhedron(radius=radii[i], center=coor[3*i : 3*(i+1)], nb_vertices=5, model=mod,
                             material=plex, color='BLUEx')
    body.translate(dy=40,dz=Rmax, dx=0.5*lz + 2.*Rmax)
    bodies+=body
## wall avatar generation
back=pre.roughWall3D(lx=lx, ly=ly, model=mod, material=tdur, r=Rmax*0.5, color='WALLx', center=[0.5*lx, 0.5*ly, -Rmax])
down=pre.roughWall3D(lx=lx, ly=lz, model=mod, material=tdur, r=Rmax*0.5, color='WALLx', center=[0.5*lx, 0.5*ly, -Rmax])
left=pre.roughWall3D(lx=ly, ly=ly, model=mod, material=tdur, r=Rmax*0.5, color='WALLx', center=[0.5*lx, 0.5*ly, -Rmax])
down.rotate(theta=math.pi/2.,center=down.nodes[1].coor);down.translate(dy=-0.5*ly);down.translate(dz=0.5*lz)
left.rotate(theta=math.pi/2.,center=left.nodes[1].coor);left.rotate(psi=math.pi/2.,center=left.nodes[1].coor);left.translate(dz=0.5*lz, dx=-0.5*lx)
bodies += down; bodies += back; bodies += left
pre.visuAvatars(bodies)

## boundary conditions
down.imposeDrivenDof(component=[1,2,3,4,5,6],dofty='vlocy')
left.imposeDrivenDof(component=[1,2,3,4,5,6],dofty='vlocy')
back.imposeDrivenDof(component=[1,2,3,4,5,6],dofty='vlocy')


## defining interactions

ppi = pre.tact_behav(name='iqsc0', law='IQS_CLB', fric=pp)
ppw = pre.tact_behav(name='iqsc1', law='IQS_CLB', fric=pw)
tacts+=ppi; tacts+=ppw

## visibility declaration
# between disk and disk
sv = pre.see_table(CorpsCandidat='RBDY3', candidat='POLYR',colorCandidat='BLUEx',behav=ppi,
                   CorpsAntagoniste='RBDY3', antagoniste='POLYR',colorAntagoniste='BLUEx',
                   alert=0.1*Rmin)
# between disk and wall
svw = pre.see_table(CorpsCandidat='RBDY3', candidat='POLYR',colorCandidat='BLUEx',behav=ppw,
                    CorpsAntagoniste='RBDY3', antagoniste='PLANx',colorAntagoniste='WALLx',
                    alert=0.1*Rmin)
svs+=sv; svs+=svw


pre.visuAvatars(bodies)

if not os.path.isdir('./DATBOX/'):
    os.mkdir('./DATBOX/')

## writing data files
pre.writeBulkBehav(mat,chemin='./DATBOX/',dim=dim)
pre.writeBodies(bodies,chemin='./DATBOX/')
pre.writeDofIni(bodies,chemin='./DATBOX/')
pre.writeDrvDof(bodies,chemin='./DATBOX/')
pre.writeTactBehav(tacts,svs,chemin='./DATBOX/')
pre.writeVlocRlocIni(chemin='./DATBOX/')



