import os
import sys
import shutil
import subprocess
import os
from .config import *

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


