import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np

from utils.logger import get_logger
from utils.tools import remove_folder

from .extract_clamp3 import extract_features
from .midi2mtf import midi_2_mtf

logger = get_logger(__name__)


class SymbolicProcessor:
    """Class processing symbolic embeddings"""
    def process_directory(
        self, input_dir: Path, output_dir: Path
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Process directory with midi files"""
        global_flag = False

        # Step 1: Create a temporary directory
        os.makedirs("temp", exist_ok=True)

        # Step 2: Determine modalities automatically
        if not (input_dir):
            logger.error(f'Error: Could not determine input modality for "{input_dir}"')
            sys.exit(1)

        logger.info("Detected input modality.")

        # Step 3: Extract features based on detected modality

        results = extract_mid_features(input_dir, output_dir, global_flag)

        # Step 4: Clean up
        remove_folder("temp")

        return results


def extract_mid_features(
    mid_dir, feat_dir, global_flag=True
) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract performance signal features from MIDI files."""
    global_flag = " --get_global" if global_flag else ""

    # Step 1: Delete temp folder
    remove_folder("temp/mtf")

    # Step 2: Convert MIDI files to MTF format
    midi_2_mtf(mid_dir, "./temp/mtf")

    # Step 3: Run extract_clamp3.py
    if feat_dir is None:
        return extract_features("./temp/mtf", "../cache/mid_features")
    else:
        feat_dir = os.path.abspath(feat_dir)
        return extract_features("./temp/mtf", feat_dir)
