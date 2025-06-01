import argparse
from pathlib import Path

import numpy as np
import h5py

from analysis.comparator import EmbeddingComparator
from utils.logger import get_logger

logger = get_logger(__name__)


class ComparisonRunner:
    """Class running comparison experiments for generated embeddings"""
    def __init__(self, h5_path: str, output_dir: str):
        """
        Initialize the experiment runner.

        Args:
            h5_path: Path to the HDF5 file containing embeddings
            output_dir: Directory to save experiment results
        """
        self.h5_path = Path(h5_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize comparator
        self.comparator = EmbeddingComparator(self.output_dir)

        # Initialize results storage
        self.results = {
            "audio_embeddings": {},
            "symbolic_embeddings": {},
            "similarities": {},
        }

    def load_embeddings(self) -> None:
        """Load average embeddings from HDF5 file"""
        logger.info("Loading embeddings...")

        try:
            with h5py.File(self.h5_path, 'r') as f:
                # Load audio embeddings
                audio_group = f['audio_embeddings']
                for key in audio_group.keys():
                    self.results["audio_embeddings"][key] = np.array(audio_group[key]['average'])
                
                # Load symbolic embeddings
                symbolic_group = f['symbolic_embeddings']
                for key in symbolic_group.keys():
                    self.results["symbolic_embeddings"][key] = np.array(symbolic_group[key]['average'])
                
            logger.info(f"Loaded {len(self.results['audio_embeddings'])} audio embeddings and "
                       f"{len(self.results['symbolic_embeddings'])} symbolic embeddings")
            
        except Exception as e:
            logger.error(f"Error loading embeddings from H5 file: {str(e)}")
            raise

    def compute_similarities(self) -> None:
        """Compute similarity metrics between embeddings"""
        logger.info("Computing similarities...")

        self.results["similarities"] = self.comparator.compute_similarities(
            self.results["audio_embeddings"], self.results["symbolic_embeddings"]
        )

    def analyze_results(self) -> None:
        """Analyze and visualize the results."""
        logger.info("Analyzing results...")

        # Visualize embeddings
        self.comparator.visualize_embeddings(
            self.results["audio_embeddings"], self.results["symbolic_embeddings"]
        )

        # Analyze correlations
        stats = self.comparator.analyze_correlations(self.results["similarities"])

        # Save analysis
        self.comparator.save_analysis(self.results["similarities"], stats)

    def save_results(self) -> None:
        """Save all experiment results."""
        logger.info("Saving results...")

        # Save similarities
        np.save(self.output_dir / "similarities.npy", self.results["similarities"])

        logger.info(f"Results saved to {self.output_dir}")

    def run(self) -> None:
        """Run the complete experiment pipeline."""
        logger.info(f"Starting static embedding comparison from: {self.h5_path}")

        try:
            self.load_embeddings()
            self.compute_similarities()
            self.analyze_results()
            self.save_results()

            logger.info("Experiment completed successfully!")

        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare average embeddings from audio and symbolic data"
    )
    parser.add_argument(
        "--h5_path",
        type=str,
        required=True,
        help="Path to the HDF5 file containing embeddings",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/static_comparison",
        help="Directory to save experiment results",
    )

    args = parser.parse_args()

    runner = ComparisonRunner(args.h5_path, args.output_dir)
    runner.run()


if __name__ == "__main__":
    main()
