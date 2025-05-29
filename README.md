# Audio vs Symbolic Embeddings Analysis

This project analyzes the similarities between embeddings of symbolic music (e.g., sheet music, MIDI) and audio format music embeddings. The goal is to investigate whether feature extraction methods for symbolic music (like CLaMP) contain information similar to that obtained from audio embeddings (e.g., OpenL3, CLAP), with special focus on style and compositional techniques.

## Project Structure

```
├── data/                    # Data storage
│   ├── raw/                # Raw audio and symbolic music files
│   │   └── dataset_name/   # Dataset-specific directory
│   │       ├── info.csv    # Dataset metadata
│   │       ├── audio/      # Audio files
│   │       └── midi/       # MIDI files
│   ├── processed/          # Processed data ready for analysis
│   └── embeddings/         # Generated embeddings
├── logs/                   # Error logs from transforming midi files
├── src/                    # Source code
│   ├── audio/             # Audio processing and embedding extraction
│   ├── symbolic/          # Symbolic music processing and embedding extraction
│   ├── analysis/          # Analysis tools and similarity metrics
│   └── utils/             # Utility functions
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Dataset requirements

The project supports any dataset that follows this structure:

1. Create a directory for your dataset under `data/raw/` (e.g., `data/raw/groove/`)
2. Place your audio and MIDI files in this directory (in any structure)
3. Create an `info.csv` file in your dataset directory with the following required columns:
   - `id`: Unique identifier for each recording
   - `audio_filename`: Path to the audio file relative to the dataset directory
   - `midi_filename`: Path to the MIDI file relative to the dataset directory

All other columns present in the .csv file will be stored with embeddings if present.

Example `info.csv`:
```csv
id,audio_filename,midi_filename,drummer,style,bpm,beat_type,time_signature,duration,split
recording1,audio/rec1.wav,midi/rec1.mid,artist1,jazz,120,swing,4-4,180.5,train
recording2,audio/rec2.wav,midi/rec2.mid,artist2,rock,140,straight,4-4,240.0,test
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

### Generating embeddings

The project generates and saves embeddings for audio and symbolic presentation.

#### Usage:
```bash
python src/generate_embeddings.py --dataset your_dataset_name
```

Parameters:
- `--dataset`: Name of your dataset directory under `data/raw/`

The script will:
1. Process each audio and MIDI file pair listed in `info.csv`
2. Generate embeddings for both formats
3. Save the results in HDF5 format under `data/embeddings/your_dataset_name/embeddings.h5`

The HDF5 file structure:
```
embeddings.h5
├── audio_embeddings/
│   ├── recording_id/
│   │   ├── sequence      # Original time-series embeddings
│   │   ├── average      # Time-averaged embeddings
│   │   └── [metadata]   # All metadata fields from info.csv
│   └── ...
└── symbolic_embeddings/
    ├── recording_id/
    │   ├── sequence      # Original time-series embeddings
    │   ├── average      # Time-averaged embeddings
    │   └── [metadata]   # All metadata fields from info.csv
    └── ...
```

### Running experiments

The project includes a comprehensive experiment runner for comparing audio and symbolic embeddings.

#### Usage:
```bash
python src/run_experiment.py --dataset your_dataset_name --output_dir results
```

Parameters:
- `--dataset`: Name of your dataset directory under `data/raw/`
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
