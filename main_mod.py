from functions.gen_sample import random_ballast_deng
from functions.compute import computer
from utilities.utilities import create_dirs
import json
import os, shutil
par_dir = f'./train-track-static/seed_mod_{687}_{1}/'
shutil.rmtree(par_dir) if os.path.exists(par_dir) else None
os.mkdir(par_dir) if not os.path.exists(par_dir) else None
create_dirs(par_dir=par_dir)
dict_sample = random_ballast_deng(par_dir=par_dir, seed=687, visu=True, step=10)
os.chdir(par_dir)
dict_str = json.dumps(dict_sample, indent=4)

    # Write to file
with open('dict.txt', 'w') as file:
    file.write(dict_str)
computer()
os.chdir('../../')