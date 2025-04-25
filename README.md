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
python src/utils/midi_to_audio.py --input_dir data/raw/GiantMIDI-PIano/midi --output_dir data/raw/GiantMIDI-PIano/audio
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

### Generate Embeddings

The project generates and saves embeddings for audio and symbolic presentation.

#### Usage:
```bash
python src/generate_embeddings.py --dataset GiantMIDI-PIano 
```

Parameters:
- `--dataset`: Name of the dataset to process (e.g., 'GiantMIDI-PIano')

### Running Experiments

The project includes a comprehensive experiment runner for comparing audio and symbolic embeddings.

#### Usage:
```bash
python src/run_experiment.py --dataset GiantMIDI-PIano --output_dir results
```

Parameters:
- `--dataset`: Name of the dataset to process (e.g., 'GiantMIDI-PIano')
- `--output_dir`: Directory to save experiment results (default: 'results')


The experiment pipeline:
1. **Loading embeddings**
   - Loads previously generated embeddings

2. **Similarity Analysis**
   - Computes cosine similarity between audio and symbolic embeddings
   - Generates visualizations (PCA, t-SNE)
   - Performs statistical analysis

3. **Results**
   - Saves similarity scores
   - Generates visualizations and statistics
   - Creates comprehensive analysis reports

Output files:
- `similarities.npy`: Similarity scores between embeddings
- `statistics.txt`: Statistical analysis results
- `embedding_visualization.png`: PCA and t-SNE plots
- `similarity_distribution.png`: Distribution of similarity scores

## License

[To be determined] 
