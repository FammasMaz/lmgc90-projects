from pylmgc90 import chipy
from pylmgc90 import post
import numpy as np
from tqdm.auto import tqdm
import pickle

def computer(deformable=False):
    # Initializing
    chipy.Initialize()  # initializing the library
    chipy.checkDirectories() # checking/creating mandatory subfolders
    chipy.utilities_DisableLogMes() # disable log

    model = 'POLYR'
    dim = 3
    mhyp = 0 # modeling hypothesis ( 1 = plain strain, 2 = plain stress, 3 = axi-symmetry)
    deformable = deformable
    # solver and params
    dt = 1.e-3
    nb_steps = 2000
    theta = 0.5
    freq_write = 1000 # frequency of writing results
    freq_disp = 1000 # frequency of visualization
    Rloc_tol = 5.e-2 # interaction parameter
    # nlgs
    tol = 1.666e-2
    relax = 1.0
    norm = 'Quad'
    gs_it1 = 100 # min number of Gauss-Seidel iterations
    gs_it2 = 10 # max number of Gauss-Seidel iterations (gs_it1*gs_it2)
    # solver_type = 'Stored_Delassus_Loops'
    solver_type = 'Stored_Delassus_Loops'

    ## read and loading data
    chipy.SetDimension(dim,mhyp)
    chipy.TimeEvolution_SetTimeStep(dt)
    chipy.Integrator_InitTheta(theta)
    chipy.ReadDatbox(deformable)

    ## Open display & postpro
    chipy.OpenDisplayFiles()
    chipy.OpenPostproFiles()

    ## simulation
    #chipy.POLYR_TopologyAngle(10)
    if model !='SPHER': chipy.PRPRx_UseCpCundallDetection(100) # use Cundall detection
    chipy.PRPRx_LowSizeArrayPolyr(10)

    chipy.ComputeMass()
    chipy.ComputeBulk()

    failed = False
    for k in tqdm(range(0,nb_steps)):
        try:    
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

            chipy.WriteDisplayFiles(freq_disp)
            chipy.WritePostproFiles()
            # chipy.checkInteractiveCommand()
        except:
            print(f'Error at step {k}')
            failed = True
            return failed
        
    # Post Processing
    time = chipy.TimeEvolution_GetTime()
    f2f = chipy.PRPRx_GetF2f2Inters()
    inter = chipy.getInteractions()
    icdans = np.array([inter[i][1] for i in range(len(inter)) if inter[i][0]==b'PRPRx'])
    ic_vec = np.zeros((len(icdans),3))
    # for each interaction get the intercenter vector
    for i in range(len(icdans)):
        ic_vec[i,:] = chipy.PRPRx_GetInteractionVector("Coor_", int(icdans[i]))
    # making the post pro files manually
    # 15. rl (3,1)
    # 16. vl (3,1)
    # 17. gaptt(1)
    # 18. coord (3,1)
    # 19. local (3,3)
    indices = [3,7,15,16,17,18,13]
    # use only the ones where first value is PRPRx
    inter_filt = [inter[i] for i in range(len(inter)) if inter[i][0]==b'PRPRx']
    stacked = np.array([inter_filt[i][1] for i in range(len(inter_filt))]).reshape(-1,1)
    for index in indices:
        column_data = np.array([row[index] for row in inter_filt])
        if index in [3,7,17]:
            column_data = column_data.reshape(-1, 1)
        if index == 13:
            column_data = np.array([0 if row[13] == b'stick' else 1 for row in inter_filt]).reshape(-1, 1)
        stacked = np.hstack((stacked, column_data))

    column_19 = np.array([row[19] for row in inter_filt]).reshape(-1, 9)
    stacked = np.hstack((stacked, column_19))
    stacked = np.hstack((stacked, ic_vec))
    # save the post pro files
    with open('postpro.pkl','wb') as f:
        pickle.dump((f2f, inter, ic_vec, stacked),f)
    # as dat file
    np.savetxt('postpro.dat', stacked, fmt='%s', delimiter='   ')



              

    

    ## close display and postpro
    chipy.CloseDisplayFiles()
    chipy.ClosePostproFiles()



    # also save in a text file

    ## Finalizing

    #f = chipy.PRPRx_GetInteractionVector("Coor_", )

    chipy.Finalize()
    return failed, f

def computer_juan():
        # Initializing
    chipy.Initialize()

    # checking/creating mandatory subfolders
    chipy.checkDirectories()

    # logMes
    chipy.utilities_DisableLogMes()

    #
    # defining some variables
    #

    # space dimension
    dim = 3

    # modeling hypothesis ( 1 = plain strain, 2 = plain stress, 3 = axi-symmetry)
    mhyp = 0

    # time evolution parameters
    dt = 5.e-4
    nb_steps = 5000

    # theta integrator parameter
    theta = 0.5

    # deformable  yes=1, no=0
    deformable = 0

    # interaction parameters
    Rloc_tol = 5.e-2

    # nlgs parameters
    tol = 1.666e-4
    relax = 1.0
    norm = 'Quad '
    gs_it1 = 100
    gs_it2 = 10
    #solver_type='Stored_Delassus_Loops         '
    solver_type   = 'Exchange_Local_Global         '

    # write parameter
    freq_write   = 50

    # display parameters
    freq_display = 50
    ref_radius = 1.e-2

    chipy.POLYR_TopologyAngle(10)
    chipy.PRPRx_ShrinkPolyrFaces(1.e-2)
    chipy.PRPRx_UseCpCundallDetection(100)
    chipy.PRPRx_LowSizeArrayPolyr(10)

    #
    # read and load
    #

    # Set space dimension
    chipy.SetDimension(dim,mhyp)
    #
    chipy.utilities_logMes('INIT TIME STEPPING')
    chipy.TimeEvolution_SetTimeStep(dt)
    chipy.Integrator_InitTheta(theta)
    #
    chipy.utilities_logMes('READ BEHAVIOURS')
    chipy.ReadBehaviours()
    if deformable: chipy.ReadModels()
    #
    chipy.utilities_logMes('READ BODIES')
    chipy.ReadBodies()
    #
    chipy.utilities_logMes('LOAD BEHAVIOURS')
    chipy.LoadBehaviours()
    if deformable: chipy.LoadModels()
    #
    chipy.utilities_logMes('READ INI DOF')
    chipy.ReadIniDof()
    #
    if deformable:
        chipy.utilities_logMes('READ INI GPV')
        chipy.ReadIniGPV()
    #
    chipy.utilities_logMes('READ DRIVEN DOF')
    chipy.ReadDrivenDof()
    #
    chipy.utilities_logMes('LOAD TACTORS')
    chipy.LoadTactors()
    #
    chipy.utilities_logMes('READ INI Vloc Rloc')
    chipy.ReadIniVlocRloc()

    #
    # paranoid writes
    #
    chipy.utilities_logMes('WRITE BODIES')
    chipy.WriteBodies()
    chipy.utilities_logMes('WRITE BEHAVIOURS')
    chipy.WriteBehaviours()
    chipy.utilities_logMes('WRITE DRIVEN DOF')
    chipy.WriteDrivenDof()

    #
    # open display & postpro
    #

    chipy.utilities_logMes('DISPLAY & WRITE')
    chipy.OpenDisplayFiles()
    chipy.OpenPostproFiles()

    #
    # simulation part ...
    #

    # ... calls a simulation time loop
    # since constant compute elementary mass once
    chipy.utilities_logMes('COMPUTE MASS')
    chipy.ComputeMass()

    for k in range(0,nb_steps):
    #
        chipy.utilities_logMes('INCREMENT STEP')
        chipy.IncrementStep()

        chipy.utilities_logMes('COMPUTE Fext')
        chipy.ComputeFext()
        chipy.utilities_logMes('COMPUTE Fint')
        chipy.ComputeBulk()
        chipy.utilities_logMes('COMPUTE Free Vlocy')
        chipy.ComputeFreeVelocity()

        chipy.utilities_logMes('SELECT PROX TACTORS')
        chipy.SelectProxTactors()

        chipy.utilities_logMes('RESOLUTION' )
        chipy.RecupRloc(Rloc_tol)

        chipy.ExSolver(solver_type, norm, tol, relax, gs_it1, gs_it2)
        chipy.UpdateTactBehav()

        chipy.StockRloc()

        chipy.utilities_logMes('COMPUTE DOF, FIELDS, etc.')
        chipy.ComputeDof()

        chipy.utilities_logMes('UPDATE DOF, FIELDS')
        chipy.UpdateStep()

        chipy.utilities_logMes('WRITE OUT DOF')
        chipy.WriteOutDof(freq_write)
        chipy.utilities_logMes('WRITE OUT Rloc')
        chipy.WriteOutVlocRloc(freq_write)

        chipy.utilities_logMes('VISU & POSTPRO')
        chipy.WriteDisplayFiles(freq_display)
        chipy.WritePostproFiles()
        print(f'\nSTEP {k}\n') if k % 10 == 0 else None


    #
    # close display & postpro
    #
    chipy.CloseDisplayFiles()
    chipy.ClosePostproFiles()

    # this is the end
    chipy.Finalize()
