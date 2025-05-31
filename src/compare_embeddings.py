import argparse
from pathlib import Path

import numpy as np

from analysis.comparator import EmbeddingComparator
from audio.processor import AudioProcessor
from symbolic.processor import SymbolicProcessor
from utils.logger import get_logger

logger = get_logger(__name__)


class ComparisonRunner:
    """Class running comparison experiments for generated embeddings"""
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
        self.raw_dir = Path("data/raw") / dataset_name
        self.processed_dir = Path("data/processed") / dataset_name
        self.embeddings_dir = Path("data/embeddings") / dataset_name

        # Create necessary directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        # Initialize processors
        self.audio_processor = AudioProcessor(self.processed_dir / "audio")
        self.symbolic_processor = SymbolicProcessor()
        self.comparator = EmbeddingComparator(self.output_dir)

        # Initialize results storage
        self.results = {
            "audio_embeddings": {},
            "symbolic_embeddings": {},
            "similarities": {},
        }

    def load_embeddings(self) -> None:
        """Load embeddings from directory"""
        logger.info("Loading embeddings...")

        try:
            data = np.load(self.embeddings_dir / "embeddings.npz", allow_pickle=True)
            self.results["audio_embeddings"] = data["audio_embeddings"].item()
            self.results["symbolic_embeddings"] = data["symbolic_embeddings"].item()
        except Exception as e:
            logger.error(f"Did not find files with embeddings. More info: {str(e)}")
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
        logger.info(f"Starting experiment for dataset: {self.dataset_name}")

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
        description="Run audio vs symbolic embeddings analysis"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset to process (e.g., GiantMIDI-PIano)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save experiment results",
    )

    args = parser.parse_args()

    runner = ComparisonRunner(args.dataset, args.output_dir)
    runner.run()


if __name__ == "__main__":
    main()
