import os
import sys
from pathlib import Path
from typing import Dict
import tempfile
import traceback

import numpy as np
import torch

from utils.logger import get_logger
from utils.tools import remove_folder

from .extract_clamp3 import extract_features
from .midi2mtf import load_midi

logger = get_logger(__name__)


class SymbolicProcessor:
    """Class processing symbolic embeddings"""
    def process_file(
        self, input_file: Path
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Process a single midi file"""
        if not input_file.exists():
            logger.error(f'Error: Input file does not exist: "{input_file}"')
            sys.exit(1)

        logger.info("Processing MIDI file.")

        # Create a temporary directory for the MTF file
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Convert MIDI to MTF format
                logger.info("Converting MIDI to MTF format...")
                try:
                    mtf_content = load_midi(str(input_file), m3_compatible=True)
                    if not mtf_content:
                        logger.error(f"Failed to convert MIDI to MTF format: {input_file}")
                        return {}
                except Exception as e:
                    logger.error(f"Error in MIDI to MTF conversion: {str(e)}\n{traceback.format_exc()}")
                    return {}
                
                mtf_file = Path(temp_dir) / "temp.mtf"
                
                # Write MTF content to temporary file
                try:
                    with open(mtf_file, "w", encoding="utf-8") as f:
                        f.write(mtf_content)
                except Exception as e:
                    logger.error(f"Failed to write MTF file: {str(e)}\n{traceback.format_exc()}")
                    return {}

                # Extract features
                logger.info("Extracting features...")
                try:
                    # Check if CUDA is available
                    if torch.cuda.is_available():
                        logger.info("Using CUDA for feature extraction")
                    else:
                        logger.info("CUDA not available, using CPU")
                        
                    results = extract_features(temp_dir)
                    if not results:
                        logger.error("Feature extraction returned empty results")
                        return {}
                except Exception as e:
                    logger.error(f"Failed to extract features: {str(e)}\n{traceback.format_exc()}")
                    return {}

                # Since we know we only processed one file (temp.mtf), get its embeddings
                if 'temp' in results:
                    embeddings = results['temp']
                    return embeddings
                else:
                    logger.error(f"No features found in results for {input_file}")
                    return {}
            except Exception as e:
                logger.error(f"Failed to process MIDI file: {str(e)}\n{traceback.format_exc()}")
                return {}
