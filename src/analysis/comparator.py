import numpy as np
from typing import Dict, List, Tuple
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)

class EmbeddingComparator:
    def __init__(self, output_dir: Path):
        """
        Initialize the embedding comparator.
        
        Args:
            output_dir: Directory to save analysis results
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_similarities(self, audio_embeddings: Dict[str, np.ndarray],
                           symbolic_embeddings: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute similarity scores between audio and symbolic embeddings.
        
        Args:
            audio_embeddings: Dictionary of audio embeddings
            symbolic_embeddings: Dictionary of symbolic embeddings
            
        Returns:
            Dictionary of similarity scores
        """
        similarities = {}
        
        # Ensure we only compare files that exist in both sets
        common_files = set(audio_embeddings.keys()) & set(symbolic_embeddings.keys())
        
        for filename in common_files:
            try:
                # Get embeddings
                audio_emb = audio_embeddings[filename]
                symbolic_emb = symbolic_embeddings[filename]
                
                # Flatten embeddings if they're not 1D
                audio_emb = audio_emb.flatten()
                symbolic_emb = symbolic_emb.flatten()
                
                # Compute cosine similarity
                similarity = cosine_similarity(
                    audio_emb.reshape(1, -1),
                    symbolic_emb.reshape(1, -1)
                )[0][0]
                
                similarities[filename] = similarity
                
            except Exception as e:
                logger.error(f"Error computing similarity for {filename}: {str(e)}")
                continue
        
        return similarities
    
    def visualize_embeddings(self, audio_embeddings: Dict[str, np.ndarray],
                           symbolic_embeddings: Dict[str, np.ndarray]) -> None:
        """
        Visualize embeddings using dimensionality reduction.
        
        Args:
            audio_embeddings: Dictionary of audio embeddings
            symbolic_embeddings: Dictionary of symbolic embeddings
        """
        # Prepare data
        common_files = set(audio_embeddings.keys()) & set(symbolic_embeddings.keys())
        
        if not common_files:
            logger.warning("No common files found between audio and symbolic embeddings")
            return
        
        # Combine embeddings
        all_embeddings = []
        labels = []
        
        for filename in common_files:
            try:
                audio_emb = audio_embeddings[filename].flatten()
                symbolic_emb = symbolic_embeddings[filename].flatten()
                
                if len(audio_emb) > 0 and len(symbolic_emb) > 0:
                    all_embeddings.append(audio_emb)
                    labels.append(f"{filename}_audio")
                    
                    all_embeddings.append(symbolic_emb)
                    labels.append(f"{filename}_symbolic")
            except Exception as e:
                logger.error(f"Error processing embeddings for {filename}: {str(e)}")
                continue
        
        if not all_embeddings:
            logger.warning("No valid embeddings found for visualization")
            return
        
        X = np.array(all_embeddings)
        
        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)
        
        # Plot results
        plt.figure(figsize=(15, 7))
        
        # PCA plot
        plt.subplot(1, 2, 1)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
        plt.title('PCA of Audio and Symbolic Embeddings')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        
        # t-SNE plot
        plt.subplot(1, 2, 2)
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.5)
        plt.title('t-SNE of Audio and Symbolic Embeddings')
        plt.xlabel('t-SNE1')
        plt.ylabel('t-SNE2')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'embedding_visualization.png')
        plt.close()
    
    def analyze_correlations(self, similarities: Dict[str, float]) -> Dict[str, float]:
        """
        Analyze correlations between embedding similarities and metadata.
        
        Args:
            similarities: Dictionary of similarity scores
            
        Returns:
            Dictionary of correlation statistics
        """
        # Convert similarities to array
        similarity_scores = np.array(list(similarities.values()))
        
        # Compute basic statistics
        stats = {
            'mean_similarity': float(np.mean(similarity_scores)),
            'std_similarity': float(np.std(similarity_scores)),
            'min_similarity': float(np.min(similarity_scores)),
            'max_similarity': float(np.max(similarity_scores)),
            'median_similarity': float(np.median(similarity_scores))
        }
        
        # Plot similarity distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(similarity_scores, kde=True)
        plt.title('Distribution of Embedding Similarities')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Count')
        plt.savefig(self.output_dir / 'similarity_distribution.png')
        plt.close()
        
        return stats
    
    def save_analysis(self, similarities: Dict[str, float],
                     stats: Dict[str, float]) -> None:
        """
        Save analysis results.
        
        Args:
            similarities: Dictionary of similarity scores
            stats: Dictionary of correlation statistics
        """
        # Save similarities
        np.save(self.output_dir / 'similarities.npy', similarities)
        
        # Save statistics
        with open(self.output_dir / 'statistics.txt', 'w') as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        
        logger.info(f"Analysis results saved to {self.output_dir}") 