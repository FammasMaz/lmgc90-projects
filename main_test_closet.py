from functions.gen_sample import closet_ballast
from functions.compute import computer_juan, computer
from utilities.lmgc90_utilities import create_dirs
import json
import os, shutil
import os
import sys
from contextlib import contextmanager
import argparse

# Boolean Interpretor
def str2bool(v):
    if v.lower() in ('yes', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'f', 'n', '0'):
        return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--visu', type=str2bool, default=False, help='Visualize the sample')
parser.add_argument('--closet', type=str2bool, default=True, help='Add wall')
parser.add_argument('--ballast', type=str2bool, default=False, help='Add ballast')
parser.add_argument("--i", "--iteration", dest="iteration", default=801, type=int, help="Iteration number")
parser.add_argument("--v", "--verbose", dest="verbose", default=True, type=str2bool, help="Verbose")
parser.add_argument("--f", "--freq_display", dest="freq_display", default=1000, type=int, help="Frequency of display")
parser.add_argument("--c", "--compute", dest="compute", default=True, type=str2bool, help="Compute")
parser.add_argument("--N", "--n_layers", dest="n_layers", default=10, type=int, help="Number of layers")

# thats it
args = parser.parse_args()


@contextmanager
def stdout_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different
i = args.iteration -1
while i < 850:
    par_dir = f'./train-track-static/data/closet_{i+1}/'
    print(f'Iteration {i+1}')
    print(f'Creating directory {par_dir}...')
    if i!=-1:
        shutil.rmtree(par_dir) if os.path.exists(par_dir) else None
        os.mkdir(par_dir) if not os.path.exists(par_dir) else None
        create_dirs(par_dir=par_dir)
        if not args.verbose:
            with stdout_redirected():
                test = closet_ballast(par_dir=par_dir, args=args)
                
        else:
            test = closet_ballast(par_dir=par_dir, seed=687, visu=args.visu, step=1000, args=args)
        os.chdir(par_dir)
    # Write to file
    else: os.chdir(par_dir)
    #if failed:
    #    print(f'Failed at iteration {i}: trying again...')
    #    continue
    #shutil.rmtree('DISPLAY') if i % 10 != 0 else None
    if args.compute:failed=computer(freq_disp=args.freq_display)
    else: break
    i += 1  # Increment i after the try-except block
    os.chdir('../../../')
