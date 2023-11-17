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

Rmin = 0.5 # minimum radius of particles
Rmax = 2.0 # maximum radius of particles

Px = 30*Rmax # width of the particle generation
Py = 5*Rmax # height of the particle generation

lx = 50*Rmax # width of the domain
ly = 30*Rmax # height of the domain

n = 1000 # number of particles

dim = 2 # dimension of the problem

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
mod = pre.model(name='rigid', physics='MECAx', element='Rxx2D',dimension=dim)

## particle generation
radii = pre.granulo_Random(n, Rmin, Rmax) # random radii
[nb_rem, coor] = pre.depositInBox2D(radii, Px, Py) # deposit particles in a box

# balls avatar generation
for i in range(0, nb_rem):
    #body = pre.rigidDisk(r=radii[i], center=coor[2*i : 2*(i+1)],model=mod,material=plex,color='BLUEx')
    body = pre.rigidDisk(r=radii[i], center=coor[2*i : 2*(i+1)], model=mod,
                             material=plex, color='BLUEx')
    body.translate(dy=40)
    bodies+=body

## wall avatar generation
down = pre.rigidJonc(axe1=0.5*lx+Rmax,axe2=Rmax,center=[0.5*lx, -Rmax],model=mod,
                     material=tdur,color='WALLx')
up = pre.rigidJonc(axe1=0.5*lx+Rmax,axe2=Rmax,center=[0.5*lx, ly+Rmax],model=mod,
                     material=tdur,color='WALLx')
left = pre.rigidJonc(axe1=0.5*ly+Rmax,axe2=Rmax,center=[-Rmax,0.5*ly],model=mod,
                        material=tdur,color='WALLx')
right = pre.rigidJonc(axe1=0.5*ly+Rmax,axe2=Rmax,center=[lx+Rmax,0.5*ly],model=mod,
                        material=tdur,color='WALLx')

bodies += down; bodies += up; bodies += left; bodies += right

left.rotate(psi=math.pi/2.,center=left.nodes[1].coor)
right.rotate(psi=math.pi/2.,center=right.nodes[1].coor)

## boundary conditions
down.imposeDrivenDof(component=[1,2,3],dofty='vlocy')
up.imposeDrivenDof(component=[1,2,3],dofty='vlocy')
left.imposeDrivenDof(component=[1,2,3],dofty='vlocy')
right.imposeDrivenDof(component=[1,2,3],dofty='vlocy')

## defining interactions

ppi = pre.tact_behav(name='iqsc0', law='IQS_CLB', fric=pp)
ppw = pre.tact_behav(name='iqsc1', law='IQS_CLB', fric=pw)
tacts+=ppi; tacts+=ppw

## visibility declaration
# between disk and disk
sv = pre.see_table(CorpsCandidat='RBDY2', candidat='DISKx',colorCandidat='BLUEx',behav=ppi,
                   CorpsAntagoniste='RBDY2', antagoniste='DISKx',colorAntagoniste='BLUEx',
                   alert=0.1*Rmin)
# between disk and wall
svw = pre.see_table(CorpsCandidat='RBDY2', candidat='DISKx',colorCandidat='BLUEx',behav=ppw,
                    CorpsAntagoniste='RBDY2', antagoniste='JONCx',colorAntagoniste='WALLx',
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



