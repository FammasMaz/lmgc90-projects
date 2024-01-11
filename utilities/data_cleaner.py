'''Clean the directory given as the argument'''

import argparse
import os, shutil
from pathlib import Path


parser = argparse.ArgumentParser(description='Clean the directory given as the argument')
parser.add_argument('--dir', type=str, help='directory to clean')
args = parser.parse_args()

def cleaner(src_dir):
    # create a folder with the folder name in ../cleaned_data
    path = Path(src_dir)
    folder_name = path.parts[-1]
    # create the folder. If it exists remove it first   
    shutil.rmtree('./train-track-static/data/cleaned_data/'+folder_name) if os.path.exists('./train-track-static/data/cleaned_data/'+folder_name) else None
    os.makedirs('./train-track-static/data/cleaned_data/'+folder_name)
    # make outbox, postpro and datbox folders
    os.mkdir('./train-track-static/data/cleaned_data/'+folder_name+'/OUTBOX')


    # OUTBOX Cleaning
    #find the highest Vloc_Rloc.OUT file
    highest_num = -1

    for filename in os.listdir(src_dir+'OUTBOX/'):
        if filename.startswith("Vloc_Rloc.OUT"):
            file_num = int(filename.split('.')[2])
            if file_num > highest_num:
                highest_num = file_num
    # copy the highest Vloc_Rloc.OUT file to ../cleaned_data/folder_name/OUTBOX
    shutil.copyfile(src_dir+'OUTBOX/Vloc_Rloc.OUT.'+str(highest_num), './train-track-static/data/cleaned_data/'+folder_name+'/OUTBOX/Vloc_Rloc.OUT')
    # copy DOF.OUT
    shutil.copyfile(src_dir+'OUTBOX/DOF.OUT.'+str(highest_num//2), './train-track-static/data/cleaned_data/'+folder_name+'/OUTBOX/DOF.OUT')
    # Copy the complete postpro
    shutil.copytree(src_dir+'POSTPRO/', './train-track-static/data/cleaned_data/'+folder_name+'/POSTPRO/')
    # copy dict.txt
    shutil.copyfile(src_dir+'dict.txt', './train-track-static/data/cleaned_data/'+folder_name+'/dict.txt')
    # copy particle_characteristics.DAT
    shutil.copyfile(src_dir+'particle_characteristics.DAT', './train-track-static/data/cleaned_data/'+folder_name+'/particle_characteristics.DAT')
    print('Cleaning done for '+folder_name)

cleaner(args.dir)
