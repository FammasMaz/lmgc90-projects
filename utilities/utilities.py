'''Create utility functions here'''
import os, shutil, math
from pylmgc90 import pre
def create_dirs(rmtrees=True, par_dir=None):
    list_dir = ['DATBOX', 'DISPLAY', 'POSTPRO', 'OUTBOX'] #to be removed
    list_dir = [os.path.join(par_dir, dir) if par_dir is not None else dir for dir in list_dir]
    [shutil.rmtree(dir) if rmtrees and os.path.exists(dir) else None for dir in list_dir]
    dir = os.path.join(par_dir, 'DATBOX') if par_dir is not None else './DATBOX'
    # copy compute.py to new dir if par_dir is not None
    os.mkdir(dir)




