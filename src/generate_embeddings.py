import argparse
import logging
from pathlib import Path

import numpy as np

from audio.processor import AudioProcessor
from symbolic.processor import SymbolicProcessor
from tools import remove_folder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GenerateEmbeddings:
    def __init__(self, dataset_name: str):
        """
        Initialize the experiment runner.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'GiantMIDI-PIano')
            output_dir: Directory to save experiment results
        """
        self.dataset_name = dataset_name

        # Setup paths
        self.raw_dir = Path('data/raw') / dataset_name
        self.processed_dir = Path('data/processed') / dataset_name
        self.embeddings_dir = Path('data/embeddings') / dataset_name

        remove_folder(Path('data/processed') / dataset_name)
        remove_folder(Path('data/embeddings') / dataset_name)

        # Create necessary directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)


        # Initialize processors
        self.audio_processor = AudioProcessor(self.processed_dir / 'audio')
        self.symbolic_processor = SymbolicProcessor()

        # Initialize results storage
        self.results = {
            'audio_embeddings': {},
            'symbolic_embeddings': {}
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

    def transform_embeddings(self) -> None:
        """Generate embeddings for both audio and symbolic formats."""
        logger.info("Generating embeddings...")

        # For now, we'll use the processed features as embeddings
        # In the future, we can add more sophisticated embedding generation
        audio_embeddings = {}
        symbolic_embeddings = {}

        # Process audio embeddings
        for k, v in self.results['audio_embeddings'].items():
            try:
                if 'embeddings' in v:
                    # Average OpenL3 embeddings over time
                    audio_embeddings[k] = np.mean(v['embeddings'], axis=0)
                else:
                    logger.warning(f"No embeddings found for {k}")
            except Exception as e:
                logger.error(f"Error processing audio embedding for {k}: {str(e)}")

        # Process symbolic embeddings
        for k, v in self.results['symbolic_embeddings'].items():
            try:
                if 'embeddings' in v:
                    symbolic_embeddings[k] = np.mean(v['embeddings'], axis=0)
                else:
                    logger.warning(f"No embeddings found for {k}")
            except Exception as e:
                logger.error(f"Error processing symbolic embedding for {k}: {str(e)}")

        # Validate embeddings
        if not audio_embeddings:
            raise ValueError("No valid audio embeddings generated")
        if not symbolic_embeddings:
            raise ValueError("No valid symbolic embeddings generated")

        self.results['audio_embeddings'] = audio_embeddings
        self.results['symbolic_embeddings'] = symbolic_embeddings

        logger.info(f"Transformed {len(audio_embeddings)} audio embeddings and {len(symbolic_embeddings)} symbolic embeddings")

    def save_embeddings(self) -> None:
        """Save all experiment embeddings."""
        logger.info("Saving embeddings...")

        # Save embeddings
        np.savez(
            self.embeddings_dir / 'embeddings.npz',
            audio_embeddings=self.results['audio_embeddings'],
            symbolic_embeddings=self.results['symbolic_embeddings']
)


        logger.info(f"Embeddings saved to {self.embeddings_dir}")

    def run(self) -> None:
        """Run the complete experiment pipeline."""
        logger.info(f"Starting experiment for dataset: {self.dataset_name}")

        try:
            self.process_data()
            self.transform_embeddings()
            self.save_embeddings()

            logger.info("Experiment completed successfully!")

        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            raise
def main():
    parser = argparse.ArgumentParser(description='Run audio vs symbolic embeddings analysis')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Name of the dataset to process (e.g., GiantMIDI-PIano)')

    args = parser.parse_args()

    runner = GenerateEmbeddings(args.dataset)
    runner.run()


if __name__ == '__main__':
    main()