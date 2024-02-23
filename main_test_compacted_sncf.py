from functions.gen_sample import random_compacted_sncf
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
parser.add_argument('--wall', type=str2bool, default=True, help='Add wall')
parser.add_argument('--trap', type=str2bool, default=False, help='Add trapezoid')
parser.add_argument('--ballast', type=str2bool, default=False, help='Add ballast')
parser.add_argument('--layers', type=float, default=1, help='Add ballast') # size ratio of the top layer
parser.add_argument('--nb_layers_min', type=int, default=1, help='Add ballast')
parser.add_argument('--nb_layers_max', type=int, default=5, help='Add ballast')
parser.add_argument('--freq_display', type=int, default=1000, help='frequency of display')
parser.add_argument('--verbose', type=str2bool, default=False, help='verbose')
parser.add_argument("--i", "--iteration", dest="iteration", default=801, type=int, help="Iteration number")

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
while i < 801:
    par_dir = f'./train-track-static/data/sncf_compacted_{i+1}/'
    print(f'Iteration {i+1}')
    print(f'Creating directory {par_dir}...')
    if i!=-1:
        shutil.rmtree(par_dir) if os.path.exists(par_dir) else None
        os.mkdir(par_dir) if not os.path.exists(par_dir) else None
        create_dirs(par_dir=par_dir)
        if not args.verbose:
            with stdout_redirected():
                dict_sample, simul_params = random_compacted_sncf(par_dir=par_dir, seed=687, visu=args.visu, step=1000, args=args)
        else:
            dict_sample, simul_params = random_compacted_sncf(par_dir=par_dir, seed=687, visu=args.visu, step=1000, args=args)
        # add 
        '''SNAPSHOT SAMPLE
        STEP 2000'''
        # to par_dir/DATBOX/POSTPRO.DAT. First remove the last line
        # with open(par_dir+'DATBOX/POSTPRO.DAT', 'r') as file:
        #     lines = file.readlines()
        # with open(par_dir+'DATBOX/POSTPRO.DAT', 'w') as file:
        #     file.writelines(lines[:-1])
        #     file.write('SNAPSHOT SAMPLE\nSTEP 100\nEND')
        # to par_dir/DATBOX/POSTPRO.DAT end
        print(f'Number of layers: {simul_params["nb_layers"]}')
        print(f'Ratio of top to lower layer: {simul_params["ratio"]}')
        os.chdir(par_dir)
        dict_str = json.dumps(dict_sample, indent=4)
    # Write to file
        with open('dict.txt', 'w') as file:
            file.write(dict_str)
    # Write to file
    else: os.chdir(par_dir)
    print('Finished generating sample. Starting computation...')
    failed=computer(freq_disp=args.freq_display)  # Call your function
    #if failed:
    #    print(f'Failed at iteration {i}: trying again...')
    #    continue
    #shutil.rmtree('DISPLAY') if i % 10 != 0 else None
    i += 1  # Increment i after the try-except block
    os.chdir('../../../')