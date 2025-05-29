from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import openl3
import soundfile as sf

from config import *
from utils.logger import get_logger

logger = get_logger(__name__)


class AudioProcessor:
    def __init__(self, output_dir: Path):
        """
        Initialize the audio processor.

        Args:
            output_dir: Directory to save processed audio features
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize OpenL3 model
        self.model = openl3.models.load_audio_embedding_model(
            input_repr=INPUT_REPR,
            content_type=CONTENT_TYPE,
            embedding_size=EMBEDDING_SIZE,
        )

    def load_audio(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """
        Load audio file.

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (audio data, sample rate)
        """
        try:
            audio, sr = sf.read(str(file_path))
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)  # Convert to mono if stereo
            return audio, sr
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {str(e)}")
            raise

    def extract_features(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Extract audio features using OpenL3.

        Args:
            audio: Audio data
            sr: Sample rate

        Returns:
            Dictionary of audio features containing time-series embeddings
        """
        try:
            # Extract OpenL3 embeddings
            emb, ts = openl3.get_audio_embedding(
                audio,
                sr,
                model=self.model,
                center=True,
                hop_size=HOP_SIZE,
                batch_size=32,
            )

            return {"embeddings": emb, "timestamps": ts}  # Time-series embeddings

        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise

    def process_file(self, file_path: Path) -> Dict[str, np.ndarray]:
        """
        Process a single audio file.

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary of processed features
        """
        try:
            # Load audio
            audio, sr = self.load_audio(file_path)

            # Extract features
            features = self.extract_features(audio, sr)

            # Save features
            output_path = self.output_dir / f"{file_path.stem}_features.npz"
            np.savez(output_path, **features)

            return features

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise
