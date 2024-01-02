from pylmgc90 import chipy
from pylmgc90.chipy import computation 
import numpy as np
def computer(deformable=False):
    # Initializing
    chipy.Initialize()
    chipy.checkDirectories()
    model = 'POLYR'
    dim = 3
    mhyp = 0 # modeling hypothesis ( 1 = plain strain, 2 = plain stress, 3 = axi-symmetry)
    deformable = deformable
    # solver and params
    dt = 5.e-4
    nb_steps = 5000
    theta = 0.5
    freq_write = 50 # frequency of writing results
    freq_disp = 50 # frequency of visualization
    ref_radius = 0.1 # radius for visualization
    Rloc_tol = 5.e-2 # interaction parameter
    # nlgs
    tol = 1.666e-3
    relax = 1.0
    norm = 'Quad'
    gs_it1 = 100 # min number of Gauss-Seidel iterations
    gs_it2 = 10 # max number of Gauss-Seidel iterations (gs_it1*gs_it2)
    # solver_type = 'Stored_Delassus_Loops'
    solver_type = 'Exchange_Local_Global'

    ## read and loading data
    chipy.SetDimension(dim,mhyp)
    chipy.TimeEvolution_SetTimeStep(dt)
    chipy.Integrator_InitTheta(theta)
    chipy.ReadDatbox(deformable)

    ## Open display & postpro
    chipy.OpenDisplayFiles()
    chipy.OpenPostproFiles()

    ## simulation
    chipy.POLYR_TopologyAngle(10)
    chipy.PRPRx_ShrinkPolyrFaces(1.e-2)
    if model !='SPHER': chipy.PRPRx_UseCpCundallDetection(100) # use Cundall detection
    chipy.PRPRx_LowSizeArrayPolyr(10)

    chipy.ComputeMass()
    chipy.ComputeBulk()
    #chipy.AssembleMechanicalLHS()
    for k in range(nb_steps):
        chipy.IncrementStep()

        chipy.ComputeFext()

        chipy.ComputeBulk()

        chipy.ComputeFreeVelocity()

        chipy.SelectProxTactors()

        chipy.RecupRloc(Rloc_tol)

        chipy.ExSolver(solver_type, norm, tol, relax, gs_it1, gs_it2)
        chipy.UpdateTactBehav()

        chipy.StockRloc()

        chipy.ComputeDof()

        chipy.UpdateStep()
        chipy.WriteOut(freq_write)
        chipy.WriteOutVlocRloc(freq_write)

        print(f'\nSTEP {k}\n')
        chipy.WriteDisplayFiles(freq_disp)
        chipy.WritePostproFiles()
        # chipy.checkInteractiveCommand()

    ## close display and postpro
    chipy.CloseDisplayFiles()
    chipy.ClosePostproFiles()

    ## Finalizing
    chipy.Finalize()
