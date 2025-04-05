import pretty_midi
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

class SymbolicProcessor:
    def __init__(self, config: dict):
        """
        Initialize symbolic music processor with configuration.
        
        Args:
            config: Configuration dictionary containing symbolic music processing parameters
        """
        self.config = config['symbolic']
        self.max_sequence_length = self.config['max_sequence_length']
        self.quantization_level = self.config['quantization_level']
    
    def load_midi(self, file_path: str) -> pretty_midi.PrettyMIDI:
        """
        Load MIDI file.
        
        Args:
            file_path: Path to MIDI file
            
        Returns:
            PrettyMIDI object
        """
        return pretty_midi.PrettyMIDI(file_path)
    
    def quantize_notes(self, midi: pretty_midi.PrettyMIDI) -> List[Dict[str, Any]]:
        """
        Quantize notes to a fixed grid.
        
        Args:
            midi: PrettyMIDI object
            
        Returns:
            List of quantized notes
        """
        quantized_notes = []
        for instrument in midi.instruments:
            for note in instrument.notes:
                quantized_notes.append({
                    'pitch': note.pitch,
                    'start': int(note.start * self.quantization_level),
                    'end': int(note.end * self.quantization_level),
                    'velocity': note.velocity
                })
        return quantized_notes
    
    def extract_features(self, file_path: str) -> np.ndarray:
        """
        Extract features from MIDI file.
        
        Args:
            file_path: Path to MIDI file
            
        Returns:
            Feature array
        """
        midi = self.load_midi(file_path)
        notes = self.quantize_notes(midi)
        
        # Create piano roll representation
        max_time = max(note['end'] for note in notes)
        piano_roll = np.zeros((128, min(max_time, self.max_sequence_length)))
        
        for note in notes:
            if note['start'] < self.max_sequence_length:
                end = min(note['end'], self.max_sequence_length)
                piano_roll[note['pitch'], note['start']:end] = note['velocity']
        
        return piano_roll 