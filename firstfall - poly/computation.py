#!/usr/bin/env python
# %load computation.py
from pylmgc90 import chipy

def initialize(dim, dt, theta, h5_file=None, logmes=True):
    """
    Initialize an LMGC90 simulation involving only rigid bodies.

    :param dim: (integer) dimension of the simulation (2 or 3)
    :param dt: (real) time step of the simulation
    :param theta: (real) value of the theta integrator ( value in [0.,1.])
    :param h5_file: (string optional) HDF5 file in which to save the computation.
                    If not set, only text files in the OUTBOX directory will be available.
    :param logmes: (boolean optional) set to False to desactivate LMGC90 log messaging.
    """

    chipy.utilities_setStopMode(False)

    chipy.Initialize()
    
    chipy.checkDirectories()
    
    
    if not logmes:
        chipy.utilities_DisableLogMes()
    
    chipy.SetDimension(dim,1)
    #
    chipy.utilities_logMes('INIT TIME STEPPING')
    chipy.TimeEvolution_SetTimeStep(dt)
    chipy.Integrator_InitTheta(theta)
    #
    chipy.utilities_logMes('READ BEHAVIOURS')
    chipy.ReadBehaviours()
    #
    chipy.utilities_logMes('READ BODIES')
    chipy.ReadBodies()
    #
    chipy.utilities_logMes('LOAD BEHAVIOURS')
    chipy.LoadBehaviours()
    #
    chipy.utilities_logMes('READ DRIVEN DOF')
    chipy.ReadDrivenDof()
    #
    chipy.utilities_logMes('LOAD TACTORS')
    chipy.LoadTactors()
    #
    chipy.utilities_logMes('READ INI')
    chipy.ReadIni()

    # paranoid writes
    chipy.utilities_logMes('WRITE BODIES')
    chipy.WriteBodies()
    chipy.utilities_logMes('WRITE BEHAVIOURS')
    chipy.WriteBehaviours()
    chipy.utilities_logMes('WRITE DRIVEN DOF')
    chipy.WriteDrivenDof()

    # open display & postpro
    chipy.utilities_logMes('DISPLAY & WRITE')
    chipy.OpenDisplayFiles()
    chipy.OpenPostproFiles()
    chipy.InitHDF5(h5_file)
    
    # if HDF5 is available
    if h5_file is not None:
        chipy.InitHDF5(h5_file)

    # since constant compute elementary mass once
    chipy.utilities_logMes('COMPUTE MASS')
    chipy.ComputeMass()


def one_step(stype, norm, tol, relax, gs_it1, gs_it2, f_write, f_display,
             ref_radius, rloc_tol=None                                  ):
    """
    Compute one step of a computation with rigids bodies.

    :param stype: type of contact solver to use can only be:
                  * 'Stored_Delassus_Loops         '
                  * 'Exchange Local Global         '
    :param norm: type of norm to use in contact solver to check convergence, can be:
                 * 'Quad '
                 * 'Maxm '
                 * 'QM/16'
    :param tol: (real) desired tolerance to decided if contact solver has converged.
    :param relax: (real) relaxation
    :param gs_it1: (integer) maximum number of converge check of the contact solver before stopping (outer loop).
    :param gs_it2: (integer) number of contact solver iteration to run before checking convergence (inner loop).
    :param f_write: (integer) frequency at which to save into file(s).
    :param f_display: (integer) frequency at which to save display files (if 0 no file generated).
    :param ref_radius: (real) reference length to use to display interactions in paraview files.
    :param rloc_tol: (real optional) if a geometric tolerance is to be used to recup
                     previous time step interactions reaction value.
    """

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
    chipy.RecupRloc(rloc_tol)

    chipy.ExSolver(stype, norm, tol, relax, gs_it1, gs_it2)
    chipy.UpdateTactBehav()

    chipy.StockRloc()

    chipy.utilities_logMes('COMPUTE DOF, FIELDS, etc.')
    chipy.ComputeDof()

    chipy.utilities_logMes('UPDATE DOF, FIELDS')
    chipy.UpdateStep()

    chipy.utilities_logMes('WRITE OUT')
    chipy.WriteOut(f_write)

    if f_display > 0:
        chipy.utilities_logMes('VISU & POSTPRO')
        chipy.WriteDisplayFiles(f_display,ref_radius)

    chipy.WritePostproFiles()

    chipy.checkInteractiveCommand()


def finalize(cleanup=True):
    """
    Finalize an LMGC90 computation.

    :param cleanup: boolean stating if the memory of LMGC90 must be purged
                    (True by default)
    """

    chipy.CloseDisplayFiles()
    chipy.ClosePostproFiles()
    
    # this is the end
    if cleanup:
        chipy.Finalize()



