import os
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Tuple
import logging
import tqdm

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, sample_rate: int = 44100, duration: float = 10.0):
        """
        Initialize the audio processor.
        
        Args:
            sample_rate: Target sample rate for audio processing
            duration: Duration in seconds to process from each audio file
        """
        self.sample_rate = sample_rate
        self.duration = duration
    
    def load_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            return audio, sr
            
        except Exception as e:
            logger.error(f"Error loading audio file {audio_path}: {str(e)}")
            raise
    
    def extract_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract audio features for embedding generation.
        
        Args:
            audio: Audio data array
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Extract mel spectrogram with fixed size
        target_length = 1024  # Fixed length for all spectrograms
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=128,
            hop_length=512
        )
        mel_spec_db = librosa.power_to_db(mel_spec)
        
        # Resize spectrogram to fixed length
        if mel_spec_db.shape[1] > target_length:
            mel_spec_db = mel_spec_db[:, :target_length]
        elif mel_spec_db.shape[1] < target_length:
            # Pad with zeros
            pad_width = target_length - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)))
        
        features['mel_spectrogram'] = mel_spec_db
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=13
        )
        features['mfcc'] = mfccs
        
        # Extract chroma features
        try:
            chroma = librosa.feature.chroma_stft(
                y=audio,
                sr=self.sample_rate
            )
            features['chroma'] = chroma
        except Exception as e:
            logger.warning(f"Could not extract chroma features: {str(e)}")
            # Use zeros as fallback
            features['chroma'] = np.zeros((12, len(audio) // 512))
        
        return features
    
    def process_directory(self, input_dir: Path, output_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Process all audio files in a directory.
        
        Args:
            input_dir: Directory containing audio files
            output_dir: Directory to save processed features
            
        Returns:
            Dictionary mapping filenames to their features
        """
        results = {}
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each audio file
        for audio_file in tqdm.tqdm(list(input_dir.glob('*.wav'))):
            try:
                # Load and preprocess audio
                audio, sr = self.load_audio(audio_file)
                
                # Extract features
                features = self.extract_features(audio)
                
                # Save features
                filename = audio_file.stem
                np.save(output_dir / f"{filename}_features.npy", features)
                
                results[filename] = features
                
            except Exception as e:
                logger.error(f"Error processing {audio_file}: {str(e)}")
                continue
        
        return results 