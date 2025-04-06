# Audio vs Symbolic Embeddings Analysis

This project analyzes the similarities between embeddings of symbolic music (e.g., sheet music, MIDI) and audio format music embeddings. The goal is to investigate whether feature extraction methods for symbolic music (like CLaMP) contain information similar to that obtained from audio embeddings (e.g., OpenL3, CLAP), with special focus on style and compositional techniques.

## Project Structure

```
├── data/                    # Data storage
│   ├── raw/                # Raw audio and symbolic music files
│   ├── processed/          # Processed data ready for analysis
│   └── embeddings/         # Generated embeddings
├── notebooks/              # Jupyter notebooks for analysis
├── src/                    # Source code
│   ├── audio/             # Audio processing and embedding extraction
│   ├── symbolic/          # Symbolic music processing and embedding extraction
│   ├── analysis/          # Analysis tools and similarity metrics
│   └── utils/             # Utility functions
├── configs/                # Configuration files
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### MIDI to Audio Conversion

The project includes a tool for converting MIDI files to audio format using a custom piano synthesizer. 

#### Usage:
```bash
python src/utils/midi_to_audio.py --input_dir data/raw/midi --output_dir data/raw/audio
```

Parameters:
- `--input_dir`: Directory containing MIDI files to convert
- `--output_dir`: Directory where WAV files will be saved
- `--sample_rate`: (Optional) Sample rate for output audio (default: 44100 Hz)

The conversion process:
1. Loads MIDI files from the input directory
2. Processes each note with custom piano synthesis
3. Applies proper envelope shaping and dynamics
4. Saves the result as high-quality WAV files

## License

[To be determined] 