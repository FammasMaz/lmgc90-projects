import os, json
import shutil
from utilities.utilities import create_dirs
from functions.gen_sample import random_ballast_sample
from functions.compute import computer
import os
import shutil
import json
import sys
import tempfile

i = 6
max_retries = 500  # Maximum number of retries for a single iteration

while i < 200:
    par_dir = f'./train-track-static/seed_radius_{687}_{i+1}/'
    shutil.rmtree(par_dir) if os.path.exists(par_dir) else None
    os.mkdir(par_dir) if not os.path.exists(par_dir) else None
    create_dirs(par_dir=par_dir)
    dict_sample = random_ballast_sample(par_dir=par_dir, seed=687, visu=False, step=10)
    print(dict_sample)
    os.chdir(par_dir)
    dict_str = json.dumps(dict_sample, indent=4)

    # Write to file
    with open('dict.txt', 'w') as file:
        file.write(dict_str)

    attempt = 0
    while attempt < max_retries:
        # Redirect output to a temporary file
        with tempfile.TemporaryFile(mode='w+') as temp:
            old_stdout = sys.stdout
            sys.stdout = temp

            computer()  # Call your function

            # Reset standard output
            sys.stdout = old_stdout

            # Check the temporary file for the "STOP 1" message
            temp.seek(0)
            stop_detected = any("Nslide" in line for line in temp)

        if stop_detected:
            print(f"STOP 1 detected, retrying iteration {i}. Attempt {attempt + 1} of {max_retries}.")
            attempt += 1
        else:
            shutil.rmtree('OUTBOX')
            shutil.rmtree('DATBOX')
            shutil.rmtree('DISPLAY') if i % 10 != 0 else None
            break  # Success, exit the while loop

    if attempt == max_retries:
        print(f"Max retries reached for iteration {i}. Moving to next iteration.")

    i += 1  # Increment i after the try-except block
    os.chdir('../../')
