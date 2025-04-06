import os
from pathlib import Path
import numpy as np
import pretty_midi
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class SymbolicProcessor:
    def __init__(self, max_notes: int = 1000):
        """
        Initialize the symbolic music processor.
        
        Args:
            max_notes: Maximum number of notes to process from each MIDI file
        """
        self.max_notes = max_notes
    
    def load_midi(self, midi_path: Path) -> pretty_midi.PrettyMIDI:
        """
        Load and preprocess a MIDI file.
        
        Args:
            midi_path: Path to the MIDI file
            
        Returns:
            PrettyMIDI object
        """
        try:
            midi_data = pretty_midi.PrettyMIDI(str(midi_path))
            return midi_data
        except Exception as e:
            logger.error(f"Error loading MIDI file {midi_path}: {str(e)}")
            raise
    
    def extract_features(self, midi_data: pretty_midi.PrettyMIDI) -> Dict[str, np.ndarray]:
        """
        Extract features from MIDI data.
        
        Args:
            midi_data: PrettyMIDI object
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Get piano roll with fixed size
        target_length = 1024  # Fixed length for all piano rolls
        piano_roll = midi_data.get_piano_roll(fs=100)  # 100 Hz resolution
        
        # Resize piano roll to fixed length
        if piano_roll.shape[1] > target_length:
            piano_roll = piano_roll[:, :target_length]
        elif piano_roll.shape[1] < target_length:
            # Pad with zeros
            pad_width = target_length - piano_roll.shape[1]
            piano_roll = np.pad(piano_roll, ((0, 0), (0, pad_width)))
        
        features['piano_roll'] = piano_roll
        
        # Extract note features
        notes = []
        for instrument in midi_data.instruments:
            for note in instrument.notes[:self.max_notes]:
                notes.append([
                    note.pitch,
                    note.start,
                    note.end,
                    note.velocity
                ])
        
        if notes:
            note_features = np.array(notes)
            features['note_features'] = note_features
        
        # Extract tempo changes
        tempo_changes = []
        try:
            tempo_data = midi_data.get_tempo_changes()
            if isinstance(tempo_data, tuple) and len(tempo_data) == 2:
                times, tempos = tempo_data
                for time, tempo in zip(times, tempos):
                    tempo_changes.append([time, tempo])
            elif len(tempo_data) > 0:
                for time, tempo in tempo_data:
                    tempo_changes.append([time, tempo])
        except Exception as e:
            logger.warning(f"Could not extract tempo changes: {str(e)}")
        
        if tempo_changes:
            tempo_features = np.array(tempo_changes)
            features['tempo_features'] = tempo_features
        
        return features
    
    def process_directory(self, input_dir: Path, output_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Process all MIDI files in a directory.
        
        Args:
            input_dir: Directory containing MIDI files
            output_dir: Directory to save processed features
            
        Returns:
            Dictionary mapping filenames to their features
        """
        results = {}
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each MIDI file
        for midi_file in tqdm(list(input_dir.glob('*.mid'))):
            try:
                # Load MIDI file
                midi_data = self.load_midi(midi_file)
                
                # Extract features
                features = self.extract_features(midi_data)
                
                # Save features
                filename = midi_file.stem
                np.save(output_dir / f"{filename}_features.npy", features)
                
                results[filename] = features
                
            except Exception as e:
                logger.error(f"Error processing {midi_file}: {str(e)}")
                continue
        
        return results 