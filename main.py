import os, json
from utilities.utilities import create_dirs
from functions.gen_sample import random_ballast_sample
from functions.compute import computer

for i in range(54,76):
    par_dir = f'./train-track-static/seed_{687}_{i}/'
    [os.mkdir(par_dir) if not os.path.exists(par_dir) else None]
    create_dirs(par_dir=par_dir)
    #visu = True if i==1 else False
    dict_sample = random_ballast_sample(par_dir=par_dir, seed=687, visu=False, step=10)
    print(dict_sample)
    os.chdir(par_dir)
    dict_str = json.dumps(dict_sample, indent=4)

    # Write to file
    with open('dict.txt', 'w') as file:
        file.write(dict_str)

    computer()
    os.chdir('../../')


