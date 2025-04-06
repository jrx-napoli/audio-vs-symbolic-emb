import pretty_midi
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

def create_piano_note(frequency, duration, amplitude, sample_rate=44100):
    """
    Create a more realistic piano note using harmonic synthesis and envelope shaping.
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Fundamental frequency with harmonics tuned for piano timbre
    harmonics = [1, 2, 3, 4, 5, 6, 7, 8]
    # These amplitudes create a warmer, more natural piano sound
    harmonic_amplitudes = [1.0, 0.7, 0.33, 0.2, 0.14, 0.11, 0.09, 0.07]
    
    # Generate the note with all harmonics
    note = np.zeros_like(t)
    for harmonic, amp in zip(harmonics, harmonic_amplitudes):
        note += amp * np.sin(2 * np.pi * frequency * harmonic * t)
    
    # Apply piano-like envelope
    attack_time = 0.02  # 20ms attack
    decay_time = 0.05   # 50ms initial decay
    sustain_level = 0.7
    release_time = 0.4  # 400ms release
    
    attack_samples = int(attack_time * sample_rate)
    decay_samples = int(decay_time * sample_rate)
    release_samples = int(release_time * sample_rate)
    
    # Create envelope
    envelope = np.ones_like(t)
    # Attack
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    # Decay to sustain
    decay_curve = np.linspace(1, sustain_level, decay_samples)
    envelope[attack_samples:attack_samples + decay_samples] = decay_curve
    # Release
    release_start = len(envelope) - release_samples
    release_curve = np.linspace(sustain_level, 0, release_samples)
    envelope[release_start:] *= release_curve
    
    # Apply envelope and amplitude
    note = note * envelope * amplitude
    
    # Add subtle resonance
    resonance = 0.2 * np.roll(note, int(0.01 * sample_rate))
    note += resonance
    
    return note

def midi_to_audio(midi_path: str, output_path: str, sample_rate: int = 44100) -> None:
    """
    Convert a MIDI file to audio file with realistic piano sound.
    """
    # Load MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    
    # Initialize output audio
    duration = int(midi_data.get_end_time() * sample_rate) + sample_rate  # Add 1 second padding
    audio = np.zeros(duration)
    
    # Process each instrument
    for instrument in midi_data.instruments:
        # Keep the original piano program or default to grand piano
        if instrument.program not in range(0, 8):
            instrument.program = 0
        
        # Process each note
        for note in instrument.notes:
            # Calculate frequency from MIDI note number
            frequency = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
            
            # Calculate note duration
            note_duration = note.end - note.start
            
            # Scale velocity for more natural dynamics
            velocity = note.velocity
            if velocity > 100:
                velocity = 100 + (velocity - 100) * 0.5
            elif velocity < 30:
                velocity = 30 + (velocity - 30) * 0.7
            
            # Normalize velocity to amplitude (0.0 to 1.0)
            amplitude = (velocity / 127.0) * 0.8  # Reduce max amplitude to 0.8 to prevent clipping
            
            # Generate note
            note_audio = create_piano_note(frequency, note_duration + 0.5, amplitude, sample_rate)
            
            # Add note to the output at the correct time position
            start_idx = int(note.start * sample_rate)
            end_idx = start_idx + len(note_audio)
            
            # Ensure we don't exceed audio buffer
            if end_idx > len(audio):
                note_audio = note_audio[:len(audio) - start_idx]
                end_idx = len(audio)
            
            audio[start_idx:end_idx] += note_audio[:end_idx-start_idx]
    
    # Normalize final audio
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / (max_val * 1.1)  # Leave some headroom
    
    # Apply gentle fade in/out to prevent clicks
    fade_samples = int(0.01 * sample_rate)  # 10ms fade
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    audio[:fade_samples] *= fade_in
    audio[-fade_samples:] *= fade_out
    
    # Save as WAV file
    sf.write(output_path, audio, sample_rate)

def convert_directory(input_dir: str, output_dir: str, sample_rate: int = 44100) -> None:
    """
    Convert all MIDI files in a directory to audio files.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all MIDI files
    midi_files = list(input_path.glob('*.mid')) + list(input_path.glob('*.midi'))
    
    if not midi_files:
        print(f"No MIDI files found in {input_dir}")
        return
        
    # Convert each file
    for midi_file in tqdm(midi_files, desc="Converting MIDI files"):
        output_file = output_path / f"{midi_file.stem}.wav"
        try:
            midi_to_audio(str(midi_file), str(output_file), sample_rate)
            print(f"Successfully converted {midi_file.name} to {output_file.name}")
        except Exception as e:
            print(f"Error converting {midi_file.name}: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert MIDI files to audio files')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing MIDI files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save audio files')
    parser.add_argument('--sample_rate', type=int, default=44100, help='Sample rate for output audio')
    
    args = parser.parse_args()
    convert_directory(args.input_dir, args.output_dir, args.sample_rate)