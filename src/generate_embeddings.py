import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import h5py

from audio.processor import AudioProcessor
from symbolic.processor import SymbolicProcessor
from utils.logger import get_logger
from utils.tools import remove_folder

logger = get_logger(__name__)


class GenerateEmbeddings:
    """Class generating audio and symbolic embeddings"""
    def __init__(self, dataset_name: str):
        """
        Initialize the experiment runner.

        Args:
            dataset_name: Name of the dataset (e.g., 'groove')
        """
        self.results = None
        self.dataset_name = dataset_name

        # Setup paths
        self.raw_dir = Path("data/raw") / dataset_name
        self.processed_dir = Path("data/processed") / dataset_name
        self.embeddings_dir = Path("data/embeddings") / dataset_name

        remove_folder(Path("data/processed") / dataset_name)
        remove_folder(Path("data/embeddings") / dataset_name)

        # Create necessary directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        # Initialize processors
        self.audio_processor = AudioProcessor(self.processed_dir / "audio")
        self.symbolic_processor = SymbolicProcessor()

        # Load metadata
        self.metadata = pd.read_csv(self.raw_dir / "info.csv")

    def process_data(self) -> None:
        """Process raw data into a standardized format."""
        logger.info("Processing data...")

        audio_features = {}
        symbolic_features = {}

        counter = 0

        # Process files one by one according to metadata
        for _, row in self.metadata.iterrows():

            counter += 1
            if counter == 5:
                break

            try:
                # Process audio file
                audio_path = self.raw_dir / row['audio_filename']
                if audio_path.exists():
                    audio_features[row['id']] = self.audio_processor.process_file(audio_path)
                    logger.info(f"Processed audio file: {audio_path}")
                else:
                    logger.warning(f"Audio file not found: {audio_path}")

                # Process MIDI file
                midi_path = self.raw_dir / row['midi_filename']
                if midi_path.exists():
                    symbolic_features[row['id']] = self.symbolic_processor.process_file(midi_path)
                    logger.info(f"Processed MIDI file: {midi_path}")
                else:
                    logger.warning(f"MIDI file not found: {midi_path}")

            except Exception as e:
                logger.error(f"Error processing files for {row['id']}: {str(e)}")
                continue

        self.results = {
            "audio_embeddings": audio_features,
            "symbolic_embeddings": symbolic_features,
        }

    def transform_embeddings(self) -> None:
        """Generate embeddings for both audio and symbolic formats."""
        logger.info("Generating embeddings...")

        audio_embeddings = {}
        symbolic_embeddings = {}
        metadata_records = []

        # Process audio embeddings
        for k, v in self.results["audio_embeddings"].items():
            try:
                if "embeddings" in v:
                    # Store both original sequence and averaged embeddings
                    audio_embeddings[k] = {
                        "sequence": v["embeddings"],  # Original time-series embeddings
                        "average": np.mean(v["embeddings"], axis=0)  # Time-averaged embeddings
                    }
                    
                    # Get corresponding metadata
                    metadata_row = self.metadata[self.metadata['id'] == k].iloc[0]
                    # Convert metadata row to dictionary and add to records
                    metadata_records.append(metadata_row.to_dict())
                else:
                    logger.warning(f"No embeddings found for {k}")
            except Exception as e:
                logger.error(f"Error processing audio embedding for {k}: {str(e)}")

        # Process symbolic embeddings
        for k, v in self.results["symbolic_embeddings"].items():
            try:
                if "embeddings" in v:
                    symbolic_embeddings[k] = {
                        "sequence": v["embeddings"],  # Original time-series embeddings
                        "average": np.mean(v["embeddings"], axis=0)  # Time-averaged embeddings
                    }
                else:
                    logger.warning(f"No embeddings found for {k}")
            except Exception as e:
                logger.error(f"Error processing symbolic embedding for {k}: {str(e)}")

        # Validate embeddings
        if not audio_embeddings:
            raise ValueError("No valid audio embeddings generated")
        # if not symbolic_embeddings:
        #     raise ValueError("No valid symbolic embeddings generated")

        self.results = {
            "audio_embeddings": audio_embeddings,
            "symbolic_embeddings": symbolic_embeddings,
            "metadata": metadata_records
        }

        logger.info(
            f"Transformed {len(audio_embeddings)} audio embeddings and {len(symbolic_embeddings)} symbolic embeddings"
        )

    def save_embeddings(self) -> None:
        """Save all experiment embeddings in HDF5 format."""
        logger.info("Saving embeddings...")

        # Create HDF5 file
        with h5py.File(self.embeddings_dir / "embeddings.h5", 'w') as f:

            # Save audio embeddings
            audio_group = f.create_group('audio_embeddings')

            for k, v in self.results["audio_embeddings"].items():

                # Create recording group directly under audio_embeddings
                rec_group = audio_group.create_group(k)
                
                # Save embeddings
                rec_group.create_dataset('sequence', data=v['sequence'])
                rec_group.create_dataset('average', data=v['average'])
                
                # Add metadata for this recording
                metadata = self.metadata[self.metadata['id'] == k].iloc[0]
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float)):
                        rec_group.create_dataset(key, data=str(value))
                    else:
                        rec_group.create_dataset(key, data=str(value))

            # Save symbolic embeddings
            symbolic_group = f.create_group('symbolic_embeddings')

            for k, v in self.results["symbolic_embeddings"].items():

                # Create recording group directly under symbolic_embeddings
                rec_group = symbolic_group.create_group(k)
                
                # Save embeddings
                rec_group.create_dataset('sequence', data=v['sequence'])
                rec_group.create_dataset('average', data=v['average'])
                
                # Add metadata for this recording
                metadata = self.metadata[self.metadata['id'] == k].iloc[0]
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float)):
                        rec_group.create_dataset(key, data=str(value))
                    else:
                        rec_group.create_dataset(key, data=str(value))

        logger.info(f"Embeddings saved to {self.embeddings_dir}")

    def run(self) -> None:
        """Run the complete experiment pipeline."""
        logger.info(f"Starting to generate embeddings for dataset: {self.dataset_name}")

        try:
            self.process_data()
            self.transform_embeddings()
            self.save_embeddings()

            logger.info("Generating embeddings completed successfully!")

        except Exception as e:
            logger.error(f"Generating embeddings: {str(e)}")
            raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run audio vs symbolic embeddings analysis"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset to process (e.g., groove)",
    )

    args = parser.parse_args()

    runner = GenerateEmbeddings(args.dataset)
    runner.run()


if __name__ == "__main__":
    main()
