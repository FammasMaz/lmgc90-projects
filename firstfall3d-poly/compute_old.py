import math, os, sys
from pylmgc90 import chipy
import numpy as np

chipy.Initialize()
chipy.checkDirectories() # check if directories exist e.g. DATABOX

chipy.SetDimension(3) # 2D problem

## parameters for computation

ni = 1000 # number of iterations
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
# interaction parameters
Rloc_tol = 5.e-2


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
chipy.WriteBodies()

chipy.utilities_logMes('WRITE BEHAVIORS')
chipy.bulk_behav_WriteBehaviours() # write behaviors for bulk
chipy.tact_behav_WriteBehaviours() # write behaviors for tacts

chipy.utilities_logMes('WRITE DRIVEN DOF')
chipy.WriteDrivenDof() # write driven degrees of freedom for RBDY2

## compute paramaters definition
chipy.utilities_logMes('INIT TIME STEPPING')
chipy.TimeEvolution_SetTimeStep(dt) # set time step
chipy.Integrator_InitTheta(theta) # set theta for theta-method

## init postpro
chipy.OpenDisplayFiles()
chipy.ComputeMass()


chipy.PRPRx_ShrinkPolyrFaces(1e-3)
chipy.PRPRx_UseCpF2fExplicitDetection(1e-3)
chipy.PRPRx_LowSizeArrayPolyr(10)
chipy.nlgs_3D_DiagonalResolution()
## simulation start
for k in range(1,ni+1):
    chipy.utilities_logMes('iter:'+str(k))
    chipy.IncrementStep()

    chipy.utilities_logMes('COMPUTE Fext')
    chipy.ComputeFext() # compute external forces for RBDY2

    chipy.utilities_logMes('COMPUTE Fint')
    chipy.ComputeBulk() # compute internal forces for RBDY2

    chipy.utilities_logMes('COMPUTE Free Vloc')
    chipy.ComputeFreeVelocity() # compute free velocity for RBDY2

    chipy.utilities_logMes('SELECT PROX TACTORS')
    chipy.SelectProxTactors()

    chipy.utilities_logMes('RESOLUTION' )
    chipy.RecupRloc()

    chipy.ExSolver(stype, norm, tol, relax, gs_it1, gs_it2)
    chipy.UpdateTactBehav()
    chipy.StockRloc()

    chipy.utilities_logMes('COMPUTE DOF')
    chipy.ComputeDof() # compute degrees of freedom for RBDY2

    chipy.utilities_logMes('UPDATE DOF')
    chipy.UpdateStep() # update degrees of freedom for time evolution

    chipy.utilities_logMes('WRITE OUT')
    chipy.WriteOut()
    # post 2d
    chipy.utilities_logMes('VISU & POSTPRO')
    chipy.WriteDisplayFiles(freq_visu)
    # writeout handling
    chipy.WritePostproFiles()

chipy.CloseDisplayFiles()
chipy.ClosePostproFiles()
chipy.Finalize()


    






