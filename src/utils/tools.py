import os
import shutil
import sys

from utils.logger import get_logger

logger = get_logger(__name__)


def contains_midi_files(directory):
    """Check if the directory (recursively) contains any MIDI files (.mid or .midi)."""
    if not os.path.exists(directory) or not os.path.isdir(directory):
        logger.error(
            f'Error: Directory "{directory}" does not exist or is not a directory."'
        )
        return False

    midi_extensions = {"mid", "midi"}

    for _, _, files in os.walk(directory):
        for file in files:
            file_ext = file.lower().split(".")[-1]
            if file_ext in midi_extensions:
                return True

    return False


def remove_folder(folder):
    """Cross-platform folder deletion."""
    if os.path.exists(folder):
        try:
            shutil.rmtree(folder)
            logger.info(f'Successfully deleted: "{folder}"')
        except Exception as e:
            logger.error(f'Failed to delete "{folder}": {e}')
            sys.exit(1)
