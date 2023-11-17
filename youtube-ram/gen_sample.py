###################################################################################
# THIS IS GENERATION INPUT FILES SCRIPT TO BE USED FOR SIMULATION OF GRANULAR MATERIALS
###################################################################################
# DATE 24 JANUARY, 2021
# BY RAM CHAND (PHD, PHYSICS)
# EMAIL: RAM.CHAND2K11@YAHOO.COM
#==================================================================================

#******************************************************
####################IMPORT MODULES ####################
#******************************************************
from __future__ import print_function
import os, sys
import numpy, math
from pylmgc90.pre import *

# define seed for randomization
if '--norand' in sys.argv:
    seed = 1
else:
    seed = None
#******************************************************
##################DEFINE VARIABLES #####################
#******************************************************

# friction between particles and wall
pp = 0.3 # particle-particle
pw = 0.5 # particle-wall

# size of particles
Rmin = 0.5
Rmax = 2

# simulation region
lx = 50*Rmax
ly = 30*Rmax

#region for particle generation
Px = 50*Rmax
Py = 10*Rmax

# number of particles
nb_particles = 1000

# space dimension
dim = 2

#******************************************************
###################DEFINE CONTAINERS ##################
#******************************************************
# avatar container
bodies = avatars()
# materials container
mat = materials()
# visibilty table container : this class defines objects to store the see table used by contact detection
svs = see_tables()
# contact behavior container (interaction model with its parameters)
tacts = tact_behavs()

#------------------------------------------
# define materials
tdur = material(name='TDURx',materialType='RIGID',density=1000.)
plex = material(name='PLEXx',materialType='RIGID',density=100.)
mat.addMaterial(tdur,plex) # add to container

#define model
mod = model(name='rigid', physics='MECAx', element='Rxx2D', dimension=dim)

#*******************
#Avatar definition
#*******************
#generate radii of all particles
radii = granulo_Random(nb_particles, Rmin, Rmax, seed)
# deposit particles in particle region
[nb_remaining_particles, coor]=depositInBox2D(radii, Px, Py)

#******************
#Avatar generation
#******************
# particle AVATAR generation
for i in range(0,nb_remaining_particles,1):
    body=rigidDisk(r=radii[i], center=coor[2*i : 2*(i + 1)], model=mod, material=plex, color='BLEUx')
    body.translate(dy=40)
    bodies += body
#----------------------------

#wall AVATAR generation
down = rigidJonc(axe1=0.5*lx+Rmax, axe2=Rmax, center=[0.5*lx, -Rmax], model=mod, material=tdur, color='WALLx')
up   = rigidJonc(axe1=0.5*lx+Rmax, axe2=Rmax, center=[0.5*lx, ly+Rmax], model=mod, material=tdur, color='WALLx')
left = rigidJonc(axe1=0.5*ly+Rmax, axe2=Rmax, center=[-Rmax, 0.5*ly], model=mod, material=tdur, color='WALLx')
right= rigidJonc(axe1=0.5*ly+Rmax, axe2=Rmax, center=[lx+Rmax, 0.5*ly], model=mod, material=tdur, color='WALLx')

# add walls to avatar container
bodies += down; bodies += up; bodies += left; bodies += right

# some operation on wall avatars
left.rotate(psi=-math.pi/2., center=left.nodes[1].coor)
right.rotate(psi=math.pi/2., center=right.nodes[1].coor)

#define boundary conditions on avatars (apply v=0 on all walls)
down.imposeDrivenDof(component=[1, 2, 3], dofty='vlocy')
up.imposeDrivenDof(component=[1, 2, 3], dofty='vlocy')
left.imposeDrivenDof(component=[1, 2, 3], dofty='vlocy')
right.imposeDrivenDof(component=[1, 2, 3], dofty='vlocy')
#******************************************************
#################DEFINE INTERACTIONS ##################
#******************************************************

#  - particles vs particles and particles vs walls
ppi=tact_behav(name='iqsc0',law='IQS_CLB',fric=pp)
tacts+=ppi # add to the container

pwi=tact_behav(name='iqsc1',law='IQS_CLB',fric=pw)
tacts+=pwi # add to the container

# * VISIBILITY TABLES DECLARATION *
#   - between particles of type (disk bleu) vs (disk bleu)
dd = see_table(CorpsCandidat='RBDY2',candidat='DISKx', colorCandidat='BLEUx',behav=ppi, CorpsAntagoniste='RBDY2', antagoniste='DISKx',colorAntagoniste='BLEUx',alert=0.1*Rmin)
svs+=dd # ADD TO CONTAINER

#  between particles of type (disk bleu) vs (WALLx)
dw = see_table(CorpsCandidat='RBDY2',candidat='DISKx', colorCandidat='BLEUx',behav=pwi, CorpsAntagoniste='RBDY2', antagoniste='JONCx',colorAntagoniste='WALLx',alert=0.1*Rmin)
svs+=dw # ADD TO CONTAINER

#******************************************************
####################  WRITE FILES #####################
#******************************************************

if not os.path.isdir('./DATBOX'):
    os.mkdir('./DATBOX')
    
writeBodies(bodies,chemin='DATBOX/')
writeBulkBehav(mat,chemin='DATBOX/',dim=dim)
writeTactBehav(tacts,svs,chemin='DATBOX/')
writeDrvDof(bodies,chemin='DATBOX/')
writeDofIni(bodies,chemin='DATBOX/')
writeVlocRlocIni(chemin='DATBOX/')

#**************************************************
####################VISUALIZE #####################
#**************************************************
try:
    visuAvatars(bodies)
except:
    pass

#*******************************************************************
################ END OF SAMPLE GENERATION PART #####################
#*******************************************************************

