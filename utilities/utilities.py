'''Create utility functions here'''
import os, shutil, math
from pylmgc90 import pre
def create_dirs(rmtrees=True, par_dir=None):
    [[shutil.rmtree(dir) for dir in ['./DATBOX', './DISPLAY', './POSTPRO', './OUTBOX'] if rmtrees and os.path.isdir(dir)] if par_dir is None else None]
    dir = os.path.join(par_dir, 'DATBOX') if par_dir is not None else './DATBOX'
    os.mkdir(dir)



