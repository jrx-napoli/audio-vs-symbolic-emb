import os
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Tuple
import logging
import tqdm
import openl3
import tensorflow as tf

logger = logging.getLogger(__name__)

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
            input_repr="mel256",
            content_type="music",
            embedding_size=512
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
            Dictionary of audio features
        """
        try:
            # Extract OpenL3 embeddings
            emb, ts = openl3.get_audio_embedding(
                audio,
                sr,
                model=self.model,
                center=True,
                hop_size=0.1,
                batch_size=32
            )
            
            # Average over time to get a single embedding
            embedding = np.mean(emb, axis=0)
            
            return {
                'embedding': embedding,
                'timestamps': ts
            }
            
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
    
    def process_directory(self, input_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Process all audio files in a directory.
        
        Args:
            input_dir: Directory containing audio files
            
        Returns:
            Dictionary mapping filenames to their features
        """
        results = {}
        
        # Get all audio files
        audio_files = list(input_dir.glob('*.wav')) + list(input_dir.glob('*.mp3'))
        
        for file_path in audio_files:
            try:
                features = self.process_file(file_path)
                results[file_path.stem] = features
                logger.info(f"Processed {file_path.name}")
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {str(e)}")
                continue
        
        return results 