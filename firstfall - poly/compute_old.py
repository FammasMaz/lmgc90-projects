import math, os, sys
from pylmgc90 import chipy
import numpy as np

chipy.Initialize()
chipy.checkDirectories() # check if directories exist e.g. DATABOX

chipy.SetDimension(2) # 2D problem

## parameters for computation

ni = 6000 # number of iterations
dt = 1.e-3 # time step
theta = 0.5 

freq_write = 50 # frequency of writing results
ref_radius = 0.1

freq_visu = 50 # frequency of visualization
freq_detect = 1 # frequency of detection of contact

## solver definition
stype = 'Stored_Delassus_Loops' # solver type
tol = 1.e-4 # tolerance
relax = 1.0 # relaxation parameter
norm = 'QM/16'
gs_it1 = 50 # min number of Gauss-Seidel iterations
gs_it2 = 10 # max number of Gauss-Seidel iterations (gs_it1*gs_it2)

## model reading
chipy.utilities_logMes('READ BODIES')
chipy.ReadBodies() # read bodies from file

chipy.utilities_logMes('READ BEHAVIORS')
chipy.ReadBehaviours() # read behaviors from file

chipy.utilities_logMes('LOAD BEHAVIORS')
chipy.LoadBehaviours() # load behaviors

chipy.utilities_logMes('READ INI DOF')
chipy.ReadIniDof() # read initial degrees of freedom and velocities

chipy.utilities_logMes('LOAD TACTORS')
chipy.LoadTactors() # load tactors

chipy.utilities_logMes('READ INI Vloc Rloc')
chipy.ReadIniVlocRloc() # read initial velocities and rotations

chipy.utilities_logMes('READ DRIVEN DOF')
chipy.ReadDrivenDof() # read driven degrees of freedom

## model writing
chipy.utilities_logMes('WRITE BODIES')
chipy.overall_WriteBodies() # write bodies for overall
chipy.RBDY2_WriteBodies() # write bodies for RBDY2

chipy.utilities_logMes('WRITE BEHAVIORS')
chipy.bulk_behav_WriteBehaviours() # write behaviors for bulk
chipy.tact_behav_WriteBehaviours() # write behaviors for tacts

chipy.utilities_logMes('WRITE DRIVEN DOF')
chipy.overall_WriteDrivenDof() # write driven degrees of freedom for overall
chipy.RBDY2_WriteDrivenDof() # write driven degrees of freedom for RBDY2

## compute paramaters definition
chipy.utilities_logMes('INIT TIME STEPPING')
chipy.TimeEvolution_SetTimeStep(dt) # set time step
chipy.Integrator_InitTheta(theta) # set theta for theta-method

## init postpro
chipy.OpenDisplayFiles()
chipy.ComputeMass()

## simulation start
for k in range(1,ni+1):
    chipy.utilities_logMes('iter:'+str(k))
    chipy.utilities_logMes('INCREMENT STEP')
    chipy.TimeEvolution_IncrementStep() # increment step
    chipy.RBDY2_IncrementStep() # increment step for RBDY2

    chipy.utilities_logMes('DISPLAY TIMES')
    chipy.TimeEvolution_DisplayStep() # display times

    chipy.utilities_logMes('COMPUTE Fext')
    chipy.RBDY2_ComputeFext() # compute external forces for RBDY2

    chipy.utilities_logMes('COMPUTE Fint')
    chipy.RBDY2_ComputeBulk() # compute internal forces for RBDY2

    chipy.utilities_logMes('COMPUTE Free Vloc')
    chipy.RBDY2_ComputeFreeVelocity() # compute free velocity for RBDY2

    chipy.utilities_logMes('COMPUTE PROX TACTORS')
    chipy.overall_SelectProxTactors() # set prox tactors for overall
    chipy.DKJCx_SelectProxTactors() # select prox tactors for DKJCx
    chipy.DKPLx_SelectProxTactors() # select prox tactors for DKPLx
    chipy.RecupRloc()
    chipy.nlgs_ExSolver(stype,norm,tol,relax,gs_it1,gs_it2)
    chipy.StockRloc()

    chipy.utilities_logMes('COMPUTE DOF')
    chipy.RBDY2_ComputeDof() # compute degrees of freedom for RBDY2

    chipy.utilities_logMes('UPDATE DOF')
    chipy.TimeEvolution_UpdateStep() # update degrees of freedom for time evolution
    chipy.RBDY2_UpdateDof() # update degrees of freedom for RBDY2

    chipy.utilities_logMes('WRITE LAST DOF')
    chipy.TimeEvolution_WriteLastDof() # write last degrees of freedom for time evolution
    chipy.RBDY2_WriteLastDof() # write last degrees of freedom for RBDY2

    chipy.utilities_logMes('WRITE LAST Vloc Rloc')
    chipy.TimeEvolution_WriteLastVlocRloc() # write last velocities and rotations)
    chipy.DKJCx_WriteLastVlocRloc() # write last velocities and rotations for DKJCx
    chipy.DKPLx_WriteLastVlocRloc() # write last velocities and rotations for DKPLx

    # post 2d
    chipy.WriteDisplayFiles(freq_visu,ref_radius)
    # writeout handling
    chipy.overall_CleanWriteOutFlags()

chipy.CloseDisplayFiles()
chipy.Finalize()


    






