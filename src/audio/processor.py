import librosa
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

class AudioProcessor:
    def __init__(self, config: dict):
        """
        Initialize audio processor with configuration.
        
        Args:
            config: Configuration dictionary containing audio processing parameters
        """
        self.config = config['audio']
        self.sample_rate = self.config['sample_rate']
        self.duration = self.config['duration']
        self.hop_length = self.config['hop_length']
        self.n_fft = self.config['n_fft']
        self.n_mels = self.config['n_mels']
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and resample if necessary.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio signal, sample rate)
        """
        y, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
        return y, sr
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel spectrogram from audio signal.
        
        Args:
            audio: Audio signal
            
        Returns:
            Mel spectrogram
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        return librosa.power_to_db(mel_spec, ref=np.max)
    
    def extract_features(self, file_path: str) -> np.ndarray:
        """
        Extract features from audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Feature array
        """
        audio, _ = self.load_audio(file_path)
        mel_spec = self.extract_mel_spectrogram(audio)
        return mel_spec 