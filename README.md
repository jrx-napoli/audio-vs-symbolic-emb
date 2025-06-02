# Audio vs Symbolic Embeddings Analysis

This project analyzes the similarities between embeddings of symbolic music (e.g., sheet music, MIDI) and audio format music embeddings. The goal is to investigate whether feature extraction methods for symbolic music (like CLaMP) contain information similar to that obtained from audio embeddings (e.g., OpenL3, CLAP), with special focus on style and compositional techniques.

## Project Structure

```
.
├── data/
│   ├── raw/                # Raw audio and symbolic music files
│   │   └── dataset_name/   # Dataset-specific directory
│   │       ├── info.csv    # Dataset metadata
│   │       ├── audio/      # Audio files
│   │       └── midi/       # MIDI files
│   └── embeddings/
│       └── groove/
│           └── embeddings.h5    # Pre-computed embeddings
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── embedding_dataset.py # Dataset class for loading embeddings
│   ├── models/
│   │   ├── __init__.py
│   │   └── classifiers.py      # Model architectures
│   ├── training/
│   │   ├── __init__.py
│   │   └── experiment.py       # Training experiment class
│   ├── utils/
│   │   ├── __init__.py
│   │   └── metrics.py         # Evaluation metrics
│   ├── __init__.py
│   └── main.py                # CLI entry point
├── results/                   # Training results and model checkpoints
└── requirements.txt           # Python dependencies
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

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/audio-vs-symbolic-emb.git
cd audio-vs-symbolic-emb
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Generating embeddings

The project generates and saves embeddings for audio and symbolic presentation.

#### Usage:

```bash
python src/generate_embeddings.py --dataset your_dataset_name
```

The script will:
1. Process each audio and MIDI file pair listed in `info.csv`
2. Generate embeddings for both formats
3. Save the results in HDF5 format under `data/embeddings/your_dataset_name/embeddings.h5`


## Comparing Embeddings

The project includes a comprehensive experiment runner for comparing audio and symbolic embeddings.

```bash
python src/compare_embeddings.py --h5_path data/embeddings/groove/embeddings.h5 --output_dir results --label style  
```

#### Command Line Arguments

- `--h5_path`: Path to the HDF5 file containing embeddings (required)
- `--output_dir`: Directory to save results.
- `--label`: (Optional) What labels should be compared for both audio and symbolic embeddings. If not given the script will compare aduio embeddings agains symbolic embeddings. 

## Training a Classification Model

The main entry point for training is `src/main.py`. You can run experiments using the following command:

```bash
python -m src.main \
    --h5_path data/embeddings/groove/embeddings.h5 \
    --medium audio_embeddings \
    --label style \
    --batch_size 32 \
    --learning_rate 0.001 \
    --num_epochs 50 \
    --embedding_type average 
```

#### Command Line Arguments

- `--h5_path`: Path to the HDF5 file containing embeddings (required)
- `--medium`: Type of embeddings to use (required, choices: 'audio_embeddings' or 'symbolic_embeddings')
- `--label`: Target label for classification (required, e.g., 'style', 'drummerId')
- `--batch_size`: Batch size for training (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--num_epochs`: Number of training epochs (default: 50)
- `--save_dir`: Directory to save model and results (optional, defaults to results/{medium}_{label}_{timestamp})
- `--embedding_type`: Embedding type that should be compared ('average'/'sequence' - default: 'average)
### Results

Training results are saved in the `results` directory (or specified `save_dir`), including:
- `best_model.pth`: The best model checkpoint based on validation accuracy
- `training_history.json`: Training metrics and configuration

### Custom Models

To use a custom model architecture:

1. Create a new model class in `src/models/classifiers.py` or a new file in the `models` directory
2. Inherit from `torch.nn.Module`
3. Implement the required interface:
   ```python
   class MyCustomModel(nn.Module):
       def __init__(self, input_dim: int, num_classes: int, **kwargs):
           super().__init__()
           # Your model architecture here
           
       def forward(self, x):
           # Forward pass implementation
           return output
   ```
4. Use your model by modifying the `model_fn` parameter in `src/main.py`

## Development

### Adding New Features

1. **New Dataset Types**: Add new dataset classes in `src/data/`
2. **New Models**: Add new model architectures in `src/models/`
3. **New Training Procedures**: Extend `ClassificationExperiment` in `src/training/`
4. **New Utilities**: Add helper functions in `src/utils/`

### Code Style

- Follow PEP 8 guidelines
- Use type hints for function arguments and return values
- Document classes and functions with docstrings
- Keep modules focused and single-purpose

## License

[To be determined] 
