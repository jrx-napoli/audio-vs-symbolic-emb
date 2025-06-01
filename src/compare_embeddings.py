import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import umap

from utils.logger import get_logger

logger = get_logger(__name__)


class ComparisonRunner:
    """Class running comparison experiments for generated embeddings"""
    def __init__(self, h5_path: str, output_dir: str, label_name: str = None, embedding_type: str = "average"):
        """
        Initialize the experiment runner.

        Args:
            h5_path: Path to the HDF5 file containing embeddings
            output_dir: Directory to save experiment results
            label_name: Optional name of the label to use for visualization
            embedding_type: Type of embedding to use (average, sequence)
        """
        self.h5_path = Path(h5_path)
        self.output_dir = Path(output_dir)
        self.label_name = label_name
        self.embedding_type = embedding_type
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize results storage
        if label_name:
            # For label-based analysis
            self.results = {
                "audio_embeddings": [],      # List of embeddings
                "audio_labels": [],          # List of labels
                "symbolic_embeddings": [],    # List of embeddings
                "symbolic_labels": []         # List of labels
            }
        else:
            # For audio vs symbolic comparison
            self.results = {
                "audio_embeddings": [],
                "symbolic_embeddings": [],
                "similarities": []
            }

    def _get_simplified_label(self, label: str) -> str:
        """Get simplified label by taking part before '/'."""
        return label.split('/')[0]

    def load_embeddings(self) -> None:
        """Load embeddings and their labels from HDF5 file"""
        logger.info("Loading embeddings...")

        try:
            with h5py.File(self.h5_path, 'r') as f:
                # Load audio embeddings
                audio_group = f['audio_embeddings']
                for key in audio_group.keys():
                    embedding = np.array(audio_group[key][self.embedding_type])
                    if self.label_name:
                        label = audio_group[key][self.label_name][()].decode('utf-8')
                        label = self._get_simplified_label(label)
                        self.results["audio_embeddings"].append(embedding)
                        self.results["audio_labels"].append(label)
                    else:
                        self.results["audio_embeddings"].append(embedding)
                
                # Load symbolic embeddings
                symbolic_group = f['symbolic_embeddings']
                for key in symbolic_group.keys():
                    embedding = np.array(symbolic_group[key][self.embedding_type])
                    if self.label_name:
                        label = symbolic_group[key][self.label_name][()].decode('utf-8')
                        label = self._get_simplified_label(label)
                        self.results["symbolic_embeddings"].append(embedding)
                        self.results["symbolic_labels"].append(label)
                    else:
                        self.results["symbolic_embeddings"].append(embedding)

            if self.label_name:
                # Print statistics for label-based analysis
                logger.info("\nAudio embedding statistics:")
                label_counts = defaultdict(int)
                for label in self.results["audio_labels"]:
                    label_counts[label] += 1
                for label, count in sorted(label_counts.items()):
                    logger.info(f"{label:<20}: {count:>5} samples")
                
                logger.info("\nSymbolic embedding statistics:")
                label_counts = defaultdict(int)
                for label in self.results["symbolic_labels"]:
                    label_counts[label] += 1
                for label, count in sorted(label_counts.items()):
                    logger.info(f"{label:<20}: {count:>5} samples")
            else:
                # Print statistics for audio vs symbolic comparison
                logger.info(f"\nLoaded {len(self.results['audio_embeddings'])} audio embeddings")
                logger.info(f"Loaded {len(self.results['symbolic_embeddings'])} symbolic embeddings")
            
        except Exception as e:
            logger.error(f"Error loading embeddings from H5 file: {str(e)}")
            raise

    def _create_label_visualization(self, embeddings, labels, title, output_path):
        """Create PCA, t-SNE, and UMAP visualizations for labeled embeddings."""
        # Convert embeddings and labels to numpy arrays
        X = np.array(embeddings)
        unique_labels = sorted(set(labels))
        
        # Create figure with three subplots
        plt.figure(figsize=(25, 8))
        
        # Create color map
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        label_to_color = dict(zip(unique_labels, colors))
        
        # PCA plot
        plt.subplot(131)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        for label in unique_labels:
            mask = np.array(labels) == label
            plt.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                c=[label_to_color[label]],
                label=label,
                alpha=0.6
            )
        
        plt.title(f"{title} - PCA")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # t-SNE plot
        plt.subplot(132)
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)
        
        for label in unique_labels:
            mask = np.array(labels) == label
            plt.scatter(
                X_tsne[mask, 0],
                X_tsne[mask, 1],
                c=[label_to_color[label]],
                label=label,
                alpha=0.6
            )
        
        plt.title(f"{title} - t-SNE")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # UMAP plot
        plt.subplot(133)
        reducer = umap.UMAP(random_state=42)
        X_umap = reducer.fit_transform(X)
        
        for label in unique_labels:
            mask = np.array(labels) == label
            plt.scatter(
                X_umap[mask, 0],
                X_umap[mask, 1],
                c=[label_to_color[label]],
                label=label,
                alpha=0.6
            )
        
        plt.title(f"{title} - UMAP")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

    def _create_comparison_visualization(self, audio_embeddings, symbolic_embeddings, output_path):
        """Create visualization comparing audio and symbolic embeddings."""
        # Combine embeddings
        X = np.vstack([audio_embeddings, symbolic_embeddings])
        labels = ['Audio'] * len(audio_embeddings) + ['Symbolic'] * len(symbolic_embeddings)
        
        # Create figure with three subplots
        plt.figure(figsize=(25, 8))
        
        # Create color map
        colors = {'Audio': 'blue', 'Symbolic': 'red'}
        
        # PCA plot
        plt.subplot(131)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        for label in ['Audio', 'Symbolic']:
            mask = np.array(labels) == label
            plt.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                c=colors[label],
                label=label,
                alpha=0.6
            )
        
        plt.title("Audio vs Symbolic Embeddings - PCA")
        plt.legend()
        
        # t-SNE plot
        plt.subplot(132)
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)
        
        for label in ['Audio', 'Symbolic']:
            mask = np.array(labels) == label
            plt.scatter(
                X_tsne[mask, 0],
                X_tsne[mask, 1],
                c=colors[label],
                label=label,
                alpha=0.6
            )
        
        plt.title("Audio vs Symbolic Embeddings - t-SNE")
        plt.legend()

        # UMAP plot
        plt.subplot(133)
        reducer = umap.UMAP(random_state=42)
        X_umap = reducer.fit_transform(X)
        
        for label in ['Audio', 'Symbolic']:
            mask = np.array(labels) == label
            plt.scatter(
                X_umap[mask, 0],
                X_umap[mask, 1],
                c=colors[label],
                label=label,
                alpha=0.6
            )
        
        plt.title("Audio vs Symbolic Embeddings - UMAP")
        plt.legend()
        
        # Save plot
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

    def analyze_results(self) -> None:
        """Create visualizations based on analysis mode."""
        logger.info("Creating visualizations...")
        
        if self.label_name:
            # Create separate visualizations for audio and symbolic embeddings with labels
            self._create_label_visualization(
                self.results["audio_embeddings"],
                self.results["audio_labels"],
                "Audio Embeddings",
                self.output_dir / "audio_embeddings.png"
            )
            
            self._create_label_visualization(
                self.results["symbolic_embeddings"],
                self.results["symbolic_labels"],
                "Symbolic Embeddings",
                self.output_dir / "symbolic_embeddings.png"
            )
            
            # Analyze clustering for each type
            logger.info("\nPerforming clustering analysis...")
            audio_clustering = self._analyze_clustering(
                self.results["audio_embeddings"],
                self.results["audio_labels"],
                "Audio Embeddings"
            )
            symbolic_clustering = self._analyze_clustering(
                self.results["symbolic_embeddings"],
                self.results["symbolic_labels"],
                "Symbolic Embeddings"
            )
            
            # Save clustering results
            clustering_results = {
                "audio": audio_clustering,
                "symbolic": symbolic_clustering
            }
            np.save(self.output_dir / "clustering_analysis.npy", clustering_results)
        else:
            # Create visualization comparing audio and symbolic embeddings
            self._create_comparison_visualization(
                np.array(self.results["audio_embeddings"]),
                np.array(self.results["symbolic_embeddings"]),
                self.output_dir / "embeddings_comparison.png"
            )

    def _analyze_clustering(self, embeddings, true_labels, title):
        """Analyze clustering quality using K-means."""
        X = np.array(embeddings)
        n_clusters = len(set(true_labels))
        
        # Convert string labels to numeric
        unique_labels = sorted(set(true_labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        numeric_labels = np.array([label_to_idx[label] for label in true_labels])
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        # Calculate metrics
        silhouette = silhouette_score(X, cluster_labels)
        ari = adjusted_rand_score(numeric_labels, cluster_labels)
        
        logger.info(f"\nClustering Analysis for {title}:")
        logger.info(f"Number of clusters: {n_clusters}")
        logger.info(f"Silhouette Score: {silhouette:.4f}")
        logger.info(f"Adjusted Rand Index: {ari:.4f}")
        
        return {
            "n_clusters": n_clusters,
            "silhouette": silhouette,
            "ari": ari
        }

    def run(self) -> None:
        """Run the complete analysis pipeline."""
        logger.info(f"Starting embedding analysis from: {self.h5_path}")

        try:
            self.load_embeddings()
            self.analyze_results()
            logger.info("Analysis completed successfully!")

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze embeddings from audio and symbolic data"
    )
    parser.add_argument(
        "--h5_path",
        type=str,
        required=True,
        help="Path to the HDF5 file containing embeddings",
    )
    parser.add_argument(
        "--label",
        type=str,
        help="Optional: Name of the label to use for class-based analysis (e.g., 'style', 'genre')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/embedding_analysis",
        help="Directory to save analysis results",
    )
    parser.add_argument(
        "--embedding_type",
        type=str,
        default="average",
        help="Type of embedding to use (average, sequence)",
    )

    args = parser.parse_args()

    runner = ComparisonRunner(args.h5_path, args.output_dir, args.label, args.embedding_type)
    runner.run()


if __name__ == "__main__":
    main()
