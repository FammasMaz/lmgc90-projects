###################################################################################
# THIS IS SIMULATION RUN FILE FOR ALL GENERATION INPUT FILES FOR GRANULAR MATERIALS
###################################################################################
# DATE 24 JANUARY, 2021
# BY RAM CHAND (PHD, PHYSICS)
# EMAIL: RAM.CHAND2K11@YAHOO.COM
#==================================================================================

# RUN MPI
#export OMP_SCHEDULE=STATIC
#export OMP_NUM_THREADS=4
#export OPENBLAS_NUM_THREADS=1
#***********************************
#IMPORTING CHIPY MODULE ETC
#***********************************
from __future__ import print_function
import os

import math
from pylmgc90 import chipy
#*********************************************
#INITIALIZING AND CHECKING/CREATING DIRECTORIES
#*********************************************

chipy.Initialize()                  #initializing
#chipy.utilities_DisableLogMes()      # Log message management
chipy.checkDirectories()             # Check if all subdirectories are presents
chipy.SetDimension(2)                # Set dimension in chipy

#***********************************
#DEFINING VARIABLES ETC
#***********************************
dt = 1.e-3                          #time step
theta = 0.5                         #theta integrator paramter
nb_steps = 6000                      #number of steps to be run for simulation
freq_detect = 1                     # Contact detection frequency
freq_write = 100                     #write frequency
                                    #*************************
freq_display = 100                  # display parameter
ref_radius = 0.1                  #***************************

#SOLVER DEFINITION
# Non Linear Gauss Seidel (NLGS) parameters
stype  = 'Stored_Delassus_Loops         '
norm   = 'QM/16'
tol    = 1e-4
relax  = 1.0
gs_it1 = 50                    # Minimum number of iterations
gs_it2 = 10                   # Maximum number of iterations  = gs_it2 * gs_it1

#**************************************
#MODEL READING AND COMPUTATION ON MODEL
#**************************************
#####################
### Model Reading ###
#####################
chipy.utilities_logMes('READ BODIES')
chipy.ReadBodies()

chipy.utilities_logMes('READ BEHAVIOURS')
chipy.ReadBehaviours()

chipy.utilities_logMes('LOAD BEHAVIOURS')
chipy.LoadBehaviours()

chipy.utilities_logMes('READ INI DOF')
chipy.ReadIniDof()

chipy.utilities_logMes('LOAD TACTORS')
chipy.LoadTactors()

chipy.utilities_logMes('READ INI Vloc Rloc')
chipy.ReadIniVlocRloc()

chipy.utilities_logMes('READ DRIVEN DOF')
chipy.ReadDrivenDof()
############################
### End of Model Reading ###
############################
#####################
### Model Writing ###
#####################
chipy.utilities_logMes('WRITE BODIES')
chipy.overall_WriteBodies()
chipy.RBDY2_WriteBodies()

chipy.utilities_logMes('WRITE BEHAVIOURS')
chipy.bulk_behav_WriteBehaviours()
chipy.tact_behav_WriteBehaviours()

chipy.utilities_logMes('WRITE DRIVEN DOF')
chipy.overall_WriteDrivenDof()
chipy.RBDY2_WriteDrivenDof()
############################
### End of Model Writing ###
############################

#########################################
### Computation parameters definition ###
#########################################
chipy.utilities_logMes('INIT TIME STEPPING')
chipy.TimeEvolution_SetTimeStep(dt)
chipy.Integrator_InitTheta(theta)

### Init postpro ###
chipy.OpenDisplayFiles()

### COMPUTE MASS ###
chipy.ComputeMass()

#***********************
#SIMULATION STARTS HERE
#***********************
for k in range(1, nb_steps + 1, 1):
   #
   chipy.utilities_logMes('itere : '+str(k))
   #
   chipy.utilities_logMes('INCREMENT STEP')
   chipy.TimeEvolution_IncrementStep()
   chipy.RBDY2_IncrementStep()

   chipy.utilities_logMes('DISPLAY TIMES')
   chipy.TimeEvolution_DisplayStep()

   chipy.utilities_logMes('COMPUTE Fext')
   chipy.RBDY2_ComputeFext()

   chipy.utilities_logMes('COMPUTE Fint')
   chipy.RBDY2_ComputeBulk()
   
   chipy.utilities_logMes('COMPUTE Free Vlocy')
   chipy.RBDY2_ComputeFreeVelocity()
   #
   chipy.utilities_logMes('SELECT PROX TACTORS')
   chipy.overall_SelectProxTactors()
   chipy.DKJCx_SelectProxTactors()
   chipy.DKDKx_SelectProxTactors()
   #
   chipy.RecupRloc()
   chipy.nlgs_ExSolver(stype,norm,tol,relax,gs_it1,gs_it2)
   chipy.StockRloc()
   #
   chipy.utilities_logMes('COMPUTE DOF')
   chipy.RBDY2_ComputeDof()
   #
   chipy.utilities_logMes('UPDATE DOF')
   chipy.TimeEvolution_UpdateStep()
   chipy.RBDY2_UpdateDof()
   #
   chipy.utilities_logMes('WRITE LAST DOF')
   chipy.TimeEvolution_WriteLastDof()
   chipy.RBDY2_WriteLastDof()
   #
   chipy.utilities_logMes('WRITE LAST Vloc Rloc')
   chipy.TimeEvolution_WriteLastVlocRloc()
   chipy.DKDKx_WriteLastVlocRloc()
   chipy.DKJCx_WriteLastVlocRloc()
   #
   ### post2D ###
   chipy.WriteDisplayFiles(freq_display,ref_radius)
   ### wrtieout handling ###
   chipy.overall_CleanWriteOutFlags()

#*****************
#END OF SIMULATION
#*****************

chipy.CloseDisplayFiles()
chipy.Finalize()

