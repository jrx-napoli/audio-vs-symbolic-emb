import os
import sys
import shutil
import subprocess
import re
import os
import random
from config import *
from unidecode import unidecode #TODO: Add to requirements


def contains_midi_files(directory):
    '''Check if the directory (recursively) contains any MIDI files (.mid or .midi).'''
    if not os.path.exists(directory) or not os.path.isdir(directory):
        print(f'Error: Directory \"{directory}\" does not exist or is not a directory."')
        return False

    midi_extensions = {'mid', 'midi'}

    for root, _, files in os.walk(directory):
        for file in files:
            file_ext = file.lower().split('.')[-1]
            if file_ext in midi_extensions:
                return True

    return False

def run_command(command):
    '''Execute a command and handle errors.'''
    print(f'Executing: "{command}"')
    try:
        subprocess.run(command, shell=True, check=True)
        print(f'Successfully executed: "{command}"')
    except subprocess.CalledProcessError as e:
        print(f'Error executing command: "{command}"\n{e}')
        sys.exit(1)

def remove_folder(folder):
    '''Cross-platform folder deletion.'''
    if os.path.exists(folder):
        try:
            shutil.rmtree(folder)
            print(f'Successfully deleted: "{folder}"')
        except Exception as e:
            print(f'Failed to delete "{folder}": {e}')
            sys.exit(1)

def extract_mid_features(mid_dir, feat_dir=None, global_flag=True):
    '''Extract performance signal features from MIDI files.'''
    global_flag = ' --get_global' if global_flag else ''

    # Step 0: Convert to absolute path
    mid_dir = os.path.abspath(mid_dir)
    if feat_dir is not None:
        feat_dir = os.path.abspath(feat_dir)

    # Step 1: Delete temp folder
    remove_folder('temp/mtf')


    # Step 2: Convert MIDI files to MTF format
    run_command(f'python midi2mtf.py "{mid_dir}" ./temp/mtf --m3_compatible')


    # Step 3: Run extract_clamp3.py
    if feat_dir is None:
        run_command(f'python extract_clamp3.py ./temp/mtf ../cache/mid_features{global_flag}')
    else:
        feat_dir = os.path.abspath(feat_dir)
        run_command(f'python extract_clamp3.py ./temp/mtf "{feat_dir}"{global_flag}')
