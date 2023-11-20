from pylmgc90 import chipy

# Initializing
chipy.Initialize()
chipy.checkDirectories()

dim = 3
mhyp = 0 # modeling hypothesis ( 1 = plain strain, 2 = plain stress, 3 = axi-symmetry)
deformable = 0 
# solver and params
dt = 1.e-3
nb_steps = 9000
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
chipy.PRPRx_UseCpF2fExplicitDetection(1e-3)
chipy.nlgs_3D_DiagonalResolution()
chipy.utilities_logMes('COMPUTE  MASS')
chipy.ComputeMass()

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




