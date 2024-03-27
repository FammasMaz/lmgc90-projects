from pylmgc90 import chipy
from pylmgc90 import post
import numpy as np
from tqdm.auto import tqdm
import pickle
from tabulate import tabulate

def computer(deformable=0, freq_disp=1, dt=5.e-3,time=3.5, info_dict = False, solver='SDL'):
    # Initializing
    chipy.Initialize()  # initializing the library
    chipy.checkDirectories() # checking/creating mandatory subfolders
    chipy.utilities_DisableLogMes() # disable log

    model = 'POLYR'
    dim = 3
    mhyp = 0 # modeling hypothesis ( 1 = plain strain, 2 = plain stress, 3 = axi-symmetry)
    deformable = deformable
    # solver and params
    nb_steps = int(time/dt)
    print(f'Number of steps: {nb_steps}')
    theta = 0.5
    freq_write = nb_steps//4 # frequency of writing results
    freq_disp = int(nb_steps/freq_disp) # frequency of 
    Rloc_tol = 5.e-2 # interaction parameter
    # nlgs
    tol = 1.666e-3
    relax = 1.0
    norm = 'Quad'
    gs_it1 = 20 # min number of Gauss-Seidel iterations
    gs_it2 = 50 # max number of Gauss-Seidel iterations (gs_it1*gs_it2)
    solver_type = 'Stored_Delassus_Loops' if solver == 'SDL' else 'Exchange_Local_Global'

    chipy.nlgs_3D_DiagonalResolution()
    
    # print the params
    table = [
        ["Parameter", "Value"],
        ["Time step", dt],
        ["Number of steps", nb_steps],
        ["Total time", time],
        ["Every x file written", freq_write],
        ["Every x file displayed", freq_disp],
        ["Num of files written", nb_steps//freq_write],
        ["Solver Type", solver_type],
    ]
    # add the key and values from info dict
    if info_dict:
        for key, value in info_dict.items():
            table.append([key, value])

    print(tabulate(table, headers="firstrow", tablefmt="grid"))



    ## read and loading data
    chipy.SetDimension(dim,mhyp)
    chipy.TimeEvolution_SetTimeStep(dt)
    chipy.Integrator_InitTheta(theta)
    chipy.ReadDatbox(deformable)

    ## Open display & postpro
    chipy.OpenDisplayFiles()
    chipy.OpenPostproFiles()

    ## simulation
    # topology angle

    if model !='SPHER': chipy.PRPRx_UseCpCundallDetection(100) # use Cundall detection

    chipy.ComputeMass()
    chipy.ComputeBulk()

    failed = False
    ## min ranges
    chipy.RBDY3_SetZminBoundary(-0.8)

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
            chipy.WriteDisplayFiles(freq_disp)
            chipy.WritePostproFiles()
            # chipy.checkInteractiveCommand()
        except:
            print(f'Error at step {k}')
            failed = True
            return failed
        
    # Post Processing
    chipy.WriteLastDof()
    chipy.WriteLastVlocRloc()
    x = chipy.POLYR_GetNbPOLYR()
    y = chipy.PLANx_GetNbPLANx()
    z = chipy.RBDY3_GetNbRBDY3()
    print(f'Number of POLYR: {x}, Number of PLANx: {y}, Number of RBDY3: {z}')
    coor = np.array([chipy.RBDY3_GetBodyVector("Coor_", i+1)[:3] for i in range(z)])
    fint = np.array([chipy.RBDY3_GetBodyVector("Fint_", i+1)[:3] for i in range(z)])
    fext = np.array([chipy.RBDY3_GetBodyVector("Fext_", i+1)[:3] for i in range(z)])
    time = chipy.TimeEvolution_GetTime()
    f2f = chipy.PRPRx_GetF2f2Inters()
    inter = chipy.getInteractions()

    # try again with directly getting data
    # filter in the inter for b'PRPRx'
    inter_prpr = inter[inter['inter']==b'PRPRx']
    rl = inter_prpr['rl']
    vl = inter_prpr['vl']
    gaptt = inter_prpr['gapTT'].reshape(-1,1)
    coord = inter_prpr['coor']
    type = inter_prpr['status'] == b'stick'
    # convert to 1 or 0
    type = type.astype(int).reshape(-1,1)
    uc = inter_prpr['uc'].reshape(-1,9)
    # get the intercenter vector
    cdbdy = inter_prpr['icdbdy'].reshape(-1,1)
    anbdy = inter_prpr['ianbdy'].reshape(-1,1)
    icdans = inter_prpr['icdan'].reshape(-1,1)
    # for each body in in cdbdy and anbdy get the coor and compute the difference
    ic_vec = np.zeros((len(cdbdy),3))
    for i in range(len(cdbdy)):
        ic_vec[i,:] = coor[cdbdy[i]-1] - coor[anbdy[i]-1]
    
    # stack all the data
    stacked = np.hstack((icdans,cdbdy,anbdy,rl,vl,gaptt,coord,type,uc,ic_vec))
    # save the post pro files
    with open('postpro.pkl','wb') as f:
        pickle.dump((f2f, inter, ic_vec, stacked, coor, fint, fext),f)
        

    # as dat file
    np.savetxt('postpro.dat', stacked, fmt='%s', delimiter='   ')

    chipy.WriteBodies()

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
