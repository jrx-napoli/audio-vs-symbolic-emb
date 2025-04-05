import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from typing import List, Dict, Any, Tuple

class EmbeddingComparator:
    def __init__(self, config: dict):
        """
        Initialize embedding comparator with configuration.
        
        Args:
            config: Configuration dictionary containing analysis parameters
        """
        self.config = config['analysis']
        self.similarity_metrics = self.config['similarity_metrics']
        self.dim_reduction = self.config['dimensionality_reduction']
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray, metric: str = "cosine") -> float:
        """
        Compute similarity between two embeddings.
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            metric: Similarity metric to use
            
        Returns:
            Similarity score
        """
        if metric == "cosine":
            return cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
        elif metric == "euclidean":
            return 1 / (1 + np.linalg.norm(emb1 - emb2))
        elif metric == "manhattan":
            return 1 / (1 + np.sum(np.abs(emb1 - emb2)))
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
    
    def reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Reduce dimensionality of embeddings for visualization.
        
        Args:
            embeddings: Array of embeddings
            
        Returns:
            Reduced dimensionality embeddings
        """
        if self.dim_reduction['method'] == "pca":
            pca = PCA(n_components=self.dim_reduction['n_components'])
            return pca.fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {self.dim_reduction['method']}")
    
    def compare_embeddings(self, 
                         audio_embeddings: np.ndarray, 
                         symbolic_embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Compare audio and symbolic embeddings using multiple metrics.
        
        Args:
            audio_embeddings: Array of audio embeddings
            symbolic_embeddings: Array of symbolic embeddings
            
        Returns:
            Dictionary containing similarity scores and reduced embeddings
        """
        results = {
            'similarities': {},
            'reduced_embeddings': {}
        }
        
        # Compute similarities for each metric
        for metric in self.similarity_metrics:
            similarities = []
            for audio_emb, sym_emb in zip(audio_embeddings, symbolic_embeddings):
                similarities.append(self.compute_similarity(audio_emb, sym_emb, metric))
            results['similarities'][metric] = np.mean(similarities)
        
        # Reduce dimensions for visualization
        results['reduced_embeddings']['audio'] = self.reduce_dimensions(audio_embeddings)
        results['reduced_embeddings']['symbolic'] = self.reduce_dimensions(symbolic_embeddings)
        
        return results 