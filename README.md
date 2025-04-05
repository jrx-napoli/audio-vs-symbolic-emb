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

[To be added as the project develops]

## License

[To be determined] 