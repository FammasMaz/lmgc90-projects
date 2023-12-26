from pylmgc90 import chipy
from pylmgc90.chipy import computation 
import numpy as np

# Initializing
chipy.Initialize()
chipy.checkDirectories()
model = 'POLYR'
dim = 3
mhyp = 0 # modeling hypothesis ( 1 = plain strain, 2 = plain stress, 3 = axi-symmetry)
deformable = False
# solver and params
dt = 1.e-3
nb_steps = 10000
theta = 0.5
freq_write = 50 # frequency of writing results
freq_disp = 50 # frequency of visualization
ref_radius = 0.1 # radius for visualization
Rloc_tol = 5.e-2 # interaction parameter
# nlgs
tol = 1.e-4
relax = 1.0
norm = 'QM/16'
gs_it1 = 50 # min number of Gauss-Seidel iterations
gs_it2 = 10 # max number of Gauss-Seidel iterations (gs_it1*gs_it2)
solver_type = 'Stored_Delassus_Loops'
lx = 75. # x periodicity

## read and loading data

chipy.SetDimension(dim,mhyp)
chipy.utilities_logMes('INIT TIME STEPPING')
chipy.TimeEvolution_SetTimeStep(dt)
chipy.Integrator_InitTheta(theta)
chipy.utilities_logMes('READ DATABOX')
chipy.ReadDatbox(deformable)

## Open display & postpro
chipy.utilities_logMes('DISPLAY & WRITE')
chipy.OpenDisplayFiles()
chipy.OpenPostproFiles()

## simulation
if model !='SPHER': chipy.PRPRx_UseCpCundallDetection(300) # use Cundall detection

chipy.utilities_logMes('COMPUTE  MASS')
chipy.ComputeMass()
chipy.ComputeBulk()
chipy.utilities_logMes('ASSEMB KT')
chipy.AssembleMechanicalLHS()
for k in range(nb_steps):
    chipy.utilities_logMes('INCREMENT STEP')
    chipy.IncrementStep()

    chipy.utilities_logMes('COMPUTE  Fext')
    chipy.ComputeFext()

    chipy.utilities_logMes('COMPUTE  Fint')
    chipy.ComputeBulk()

    chipy.utilities_logMes('COMPUTE Free Vlocy')
    chipy.ComputeFreeVelocity()

    chipy.utilities_logMes('COMPUTE PROX TACTORS')
    chipy.SelectProxTactors()

    chipy.utilities_logMes('RESOLUTION')
    chipy.RecupRloc(Rloc_tol)

    chipy.ExSolver(solver_type, norm, tol, relax, gs_it1, gs_it2)
    chipy.UpdateTactBehav()

    chipy.StockRloc()

    chipy.utilities_logMes('COMPUTE DOF, FIELDS, etc...')
    chipy.ComputeDof()

    chipy.utilities_logMes('UPDATE DOF, FIELDS')
    chipy.UpdateStep()
    chipy.utilities_logMes('WRITE OUT')
    chipy.WriteOut(freq_write)

    chipy.utilities_logMes('VISU & POSTPRO')
    chipy.WriteDisplayFiles(freq_disp)
    chipy.WritePostproFiles()
    chipy.checkInteractiveCommand()



## close display and postpro
chipy.CloseDisplayFiles()
chipy.ClosePostproFiles()

## Finalizing
chipy.Finalize()




from pylmgc90.chipy import computation

# space dimension
dim = 2

# modeling hypothesis: 1 = plain strain
mhyp = 1

# time evolution parameters
dt = 1e-3
nb_steps = 10000

# theta integrator parameter
theta = 0.5

# interaction parameters
Rloc_tol = 5.e-2

# nlgs parameters
tol = 1e-4
relax = 1.0
norm = 'Quad '
gs_it1 = 50
gs_it2 = 10
stype='Stored_Delassus_Loops         '

# write parameter
freq_write   = 10

# display parameters
freq_display = 10

computation.initialize(dim, dt, theta, mhyp, deformable=False)

for k in range(1, nb_steps+1):
    if k%50==0:
        print( f"computing step {k}")
    computation.one_step(stype, norm, tol, relax, gs_it1, gs_it2,
                         freq_write, freq_display                )

computation.finalize()