import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.comparator import EmbeddingComparator
from audio.processor import AudioProcessor
from symbolic.processor import SymbolicProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    def __init__(self, dataset_name: str, output_dir: str):
        """
        Initialize the experiment runner.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'GiantMIDI-PIano')
            output_dir: Directory to save experiment results
        """
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup paths
        self.raw_dir = Path('data/raw') / dataset_name
        self.processed_dir = Path('data/processed') / dataset_name
        self.embeddings_dir = Path('data/embeddings') / dataset_name

        # Create necessary directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        # Initialize processors
        self.audio_processor = AudioProcessor(self.processed_dir / 'audio')
        self.symbolic_processor = SymbolicProcessor()
        self.comparator = EmbeddingComparator(self.output_dir)

        # Initialize results storage
        self.results = {
            'audio_embeddings': {},
            'symbolic_embeddings': {},
            'similarities': {},
            'metadata': {}
        }

    def process_data(self) -> None:
        """Process raw data into a standardized format."""
        logger.info("Processing data...")

        # Process audio files
        audio_features = self.audio_processor.process_directory(
            self.raw_dir / 'audio'
        )
        self.results['audio_embeddings'] = audio_features

        # Process MIDI files
        symbolic_features = self.symbolic_processor.process_directory(
            self.raw_dir / 'midi',
            self.processed_dir / 'symbolic'
        )
        self.results['symbolic_embeddings'] = symbolic_features

    def generate_embeddings(self) -> None:
        """Generate embeddings for both audio and symbolic formats."""
        logger.info("Generating embeddings...")

        # For now, we'll use the processed features as embeddings
        # In the future, we can add more sophisticated embedding generation
        audio_embeddings = {}
        symbolic_embeddings = {}

        # Process audio embeddings
        for k, v in self.results['audio_embeddings'].items():
            try:
                if 'mel_spectrogram' in v:
                    audio_embeddings[k] = v['mel_spectrogram'].flatten()
                else:
                    logger.warning(f"No mel spectrogram found for {k}")
            except Exception as e:
                logger.error(f"Error processing audio embedding for {k}: {str(e)}")

        # Process symbolic embeddings
        for k, v in self.results['symbolic_embeddings'].items():
            try:
                if 'piano_roll' in v:
                    symbolic_embeddings[k] = v['piano_roll'].flatten()
                else:
                    logger.warning(f"No piano roll found for {k}")
            except Exception as e:
                logger.error(f"Error processing symbolic embedding for {k}: {str(e)}")

        # Validate embeddings
        if not audio_embeddings:
            raise ValueError("No valid audio embeddings generated")
        if not symbolic_embeddings:
            raise ValueError("No valid symbolic embeddings generated")

        self.results['audio_embeddings'] = audio_embeddings
        self.results['symbolic_embeddings'] = symbolic_embeddings

        logger.info(
            f"Generated {len(audio_embeddings)} audio embeddings and {len(symbolic_embeddings)} symbolic embeddings")

    def compute_similarities(self) -> None:
        """Compute similarity metrics between embeddings."""
        logger.info("Computing similarities...")

        self.results['similarities'] = self.comparator.compute_similarities(
            self.results['audio_embeddings'],
            self.results['symbolic_embeddings']
        )

    def analyze_results(self) -> None:
        """Analyze and visualize the results."""
        logger.info("Analyzing results...")

        # Visualize embeddings
        self.comparator.visualize_embeddings(
            self.results['audio_embeddings'],
            self.results['symbolic_embeddings']
        )

        # Analyze correlations
        stats = self.comparator.analyze_correlations(self.results['similarities'])

        # Save analysis
        self.comparator.save_analysis(self.results['similarities'], stats)

    def save_results(self) -> None:
        """Save all experiment results."""
        logger.info("Saving results...")

        # Save embeddings
        np.save(self.embeddings_dir / 'audio_embeddings.npy', self.results['audio_embeddings'])
        np.save(self.embeddings_dir / 'symbolic_embeddings.npy', self.results['symbolic_embeddings'])

        # Save similarities
        np.save(self.output_dir / 'similarities.npy', self.results['similarities'])

        # Save metadata
        pd.DataFrame(self.results['metadata']).to_csv(self.output_dir / 'metadata.csv')

        logger.info(f"Results saved to {self.output_dir}")

    def run(self) -> None:
        """Run the complete experiment pipeline."""
        logger.info(f"Starting experiment for dataset: {self.dataset_name}")

        try:
            self.process_data()
            self.generate_embeddings()
            self.compute_similarities()
            self.analyze_results()
            self.save_results()

            logger.info("Experiment completed successfully!")

        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Run audio vs symbolic embeddings analysis')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Name of the dataset to process (e.g., GiantMIDI-PIano)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save experiment results')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocessing even if results exist')

    args = parser.parse_args()

    runner = ExperimentRunner(args.dataset, args.output_dir)
    runner.run()


if __name__ == '__main__':
    main()
