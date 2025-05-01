from pathlib import Path
from typing import Dict


import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ks_2samp
import pandas as pd
from statsmodels.multivariate.manova import MANOVA
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingComparator:
    def __init__(self, output_dir: Path):
        """
        Initialize the embedding comparator.

        Args:
            output_dir: Directory to save analysis results
        """
        self.symbolic_embeddings = None
        self.audio_embeddings = None
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def pad_to_length(self, arr, target_length):
        if len(arr) >= target_length:
            return arr[:target_length]  # przycięcie (opcjonalne)
        padding = np.zeros(target_length - len(arr))
        return np.concatenate([arr, padding])

    def compute_similarities(
        self,
        audio_embeddings: Dict[str, np.ndarray],
        symbolic_embeddings: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
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
                max_len = max(len(audio_emb), len(symbolic_emb))
                audio_emb = self.pad_to_length(audio_emb, max_len)
                symbolic_emb = self.pad_to_length(symbolic_emb, max_len)
                # Compute cosine similarity
                similarity = cosine_similarity(
                    audio_emb.reshape(1, -1), symbolic_emb.reshape(1, -1)
                )[0][0]

                similarities[filename] = similarity

            except Exception as e:
                logger.error(f"Error computing similarity for {filename}: {str(e)}")
                continue

        return similarities

    def visualize_embeddings(
        self,
        audio_embeddings: Dict[str, np.ndarray],
        symbolic_embeddings: Dict[str, np.ndarray],
    ) -> None:
        """
        Visualize embeddings using dimensionality reduction.

        Args:
            audio_embeddings: Dictionary of audio embeddings
            symbolic_embeddings: Dictionary of symbolic embeddings
        """
        # Prepare data
        common_files = set(audio_embeddings.keys()) & set(symbolic_embeddings.keys())

        if not common_files:
            logger.warning(
                "No common files found between audio and symbolic embeddings"
            )
            return

        # Combine embeddings
        all_embeddings = []
        labels = []
        types = []  # Track embedding type (audio or symbolic)

        for filename in common_files:
            try:
                audio_emb = audio_embeddings[filename].flatten()
                symbolic_emb = symbolic_embeddings[filename].flatten()
                if len(audio_emb) > 0 and len(symbolic_emb) > 0:
                    max_len = max(len(audio_emb), len(symbolic_emb))
                    audio_emb = self.pad_to_length(audio_emb, max_len)
                    symbolic_emb = self.pad_to_length(symbolic_emb, max_len)
                    all_embeddings.append(audio_emb)
                    labels.append(f"{filename}_audio")
                    types.append("audio")

                    all_embeddings.append(symbolic_emb)
                    labels.append(f"{filename}_symbolic")
                    types.append("symbolic")
            except Exception as e:
                logger.error(f"Error processing embeddings for {filename}: {str(e)}")
                continue

        if not all_embeddings:
            logger.warning("No valid embeddings found for visualization")
            return

        X = np.array(all_embeddings)
        types = np.array(types)

        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1))
        X_tsne = tsne.fit_transform(X)

        # Plot results
        plt.figure(figsize=(15, 7))

        # PCA plot
        plt.subplot(1, 2, 1)
        for emb_type in ["audio", "symbolic"]:
            mask = types == emb_type
            color = "blue" if emb_type == "audio" else "red"
            plt.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                alpha=0.5,
                color=color,
                label=emb_type.capitalize(),
            )
        plt.title("PCA of Audio and Symbolic Embeddings")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()

        # t-SNE plot
        plt.subplot(1, 2, 2)
        for emb_type in ["audio", "symbolic"]:
            mask = types == emb_type
            color = "blue" if emb_type == "audio" else "red"
            plt.scatter(
                X_tsne[mask, 0],
                X_tsne[mask, 1],
                alpha=0.5,
                color=color,
                label=emb_type.capitalize(),
            )
        plt.title("t-SNE of Audio and Symbolic Embeddings")
        plt.xlabel("t-SNE1")
        plt.ylabel("t-SNE2")
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / "embedding_visualization.png")
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
            "mean_similarity": float(np.mean(similarity_scores)),
            "std_similarity": float(np.std(similarity_scores)),
            "min_similarity": float(np.min(similarity_scores)),
            "max_similarity": float(np.max(similarity_scores)),
            "median_similarity": float(np.median(similarity_scores)),
        }

        # Plot similarity distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(similarity_scores, kde=True)
        plt.title("Distribution of Embedding Similarities")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Count")
        plt.savefig(self.output_dir / "similarity_distribution.png")
        plt.close()

        return stats

    def save_analysis(
        self, similarities: Dict[str, float], stats: Dict[str, float]
    ) -> None:
        """
        Save analysis results.

        Args:
            similarities: Dictionary of similarity scores
            stats: Dictionary of correlation statistics
        """
        # Save similarities
        np.save(self.output_dir / "similarities.npy", similarities)

        # Save statistics
        with open(self.output_dir / "statistics.txt", "w") as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")

        logger.info(f"Analysis results saved to {self.output_dir}")

    def analyze_centroids(self):
        '''
        Analiza centroidów: oblicza średnie wektory (centroidy) dla audio i symbolic,
        następnie wyznacza ich odległość euklidesową oraz podobieństwo cosinusowe.
        '''
        # Wyznaczenie wspólnych plików/kluczy
        keys = set(self.audio_embeddings.keys()) & set(self.symbolic_embeddings.keys())
        if not keys:
            logger.info("Brak wspólnych plików pomiędzy audio i symbolic embeddings.")
            return
        keys = sorted(keys)
        # Budowa macierzy embeddingów (tylko dla wspólnych plików)
        audio_matrix = np.vstack([self.audio_embeddings[k] for k in keys])
        symbolic_matrix = np.vstack([self.symbolic_embeddings[k] for k in keys])
        # Obliczenie centroidów (średnich wektorów)
        centroid_audio = np.mean(audio_matrix, axis=0)
        centroid_symbolic = np.mean(symbolic_matrix, axis=0)
        # Odległość euklidesowa między centroidami
        euclid_dist = np.linalg.norm(centroid_audio - centroid_symbolic)
        # Podobieństwo cosinusowe między centroidami
        cos_sim = cosine_similarity(centroid_audio.reshape(1, -1),
                                    centroid_symbolic.reshape(1, -1))[0, 0]
        # Logowanie wyników
        logger.info(f"Odległość euklidesowa między centroidami: {euclid_dist:.4f}")
        logger.info(f"Podobieństwo cosinusowe między centroidami: {cos_sim:.4f}")
        # Zapis wyników do pliku
        os.makedirs(self.output_dir, exist_ok=True)
        outfile = os.path.join(self.output_dir, "centroid_analysis.txt")
        with open(outfile, 'w') as f:
            f.write(f"Euclidean distance between centroids: {euclid_dist:.6f}\n")
            f.write(f"Cosine similarity between centroids: {cos_sim:.6f}\n")
        logger.info(f"Wyniki analizy centroidów zapisano w {outfile}")

    def cluster_kmeans(self, max_clusters=10):
        '''
        Klasteryzacja KMeans dla audio i symbolic embeddingów.
        Dla każdego z zestawów obliczany jest wskaźnik silhouette dla różnych K,
        wybierany najlepszy model oraz tworzona wizualizacja zależności silhouette od liczby klastrów.
        '''
        # Wspólne pliki/klucze
        keys = set(self.audio_embeddings.keys()) & set(self.symbolic_embeddings.keys())
        if not keys:
            logger.info("Brak wspólnych plików do klasteryzacji KMeans.")
            return
        keys = sorted(keys)
        audio_matrix = np.vstack([self.audio_embeddings[k] for k in keys])
        symbolic_matrix = np.vstack([self.symbolic_embeddings[k] for k in keys])
        for modality, data_matrix in [("audio", audio_matrix), ("symbolic", symbolic_matrix)]:
            n_samples = data_matrix.shape[0]
            if n_samples < 2:
                logger.info(f"Niewystarczająca liczba próbek w {modality} do klasteryzacji.")
                continue
            max_k = min(max_clusters, n_samples - 1)
            silhouette_vals = []
            k_range = range(2, max_k + 1)
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=0)
                labels = kmeans.fit_predict(data_matrix)
                # Obliczanie silhouette score
                score = silhouette_score(data_matrix, labels)
                silhouette_vals.append(score)
            # Wybór najlepszego K
            best_k = k_range[int(np.argmax(silhouette_vals))]
            best_score = max(silhouette_vals)
            logger.info(f"Najlepsza liczba klastrów dla {modality}: {best_k} z silhouette {best_score:.4f}")
            # Wizualizacja silhouette vs liczba klastrów
            plt.figure()
            plt.plot(list(k_range), silhouette_vals, marker='o')
            plt.title(f"Silhouette score dla klasteryzacji KMeans ({modality})")
            plt.xlabel("Liczba klastrów (K)")
            plt.ylabel("Średni wskaźnik Silhouette")
            plt.xticks(list(k_range))
            plt.grid(True)
            os.makedirs(self.output_dir, exist_ok=True)
            plot_file = os.path.join(self.output_dir, f"silhouette_{modality}.png")
            plt.savefig(plot_file)
            plt.close()
            logger.info(f"Wykres silhouette dla {modality} zapisano w {plot_file}")

    def heatmap_similarity(self):
        '''
        Tworzy mapy cieplne (heatmapy) podobieństw cosinusowych wewnątrz zbioru audio i symbolic.
        '''
        keys = set(self.audio_embeddings.keys()) & set(self.symbolic_embeddings.keys())
        if not keys:
            logger.info("Brak wspólnych plików do analizy heatmap.")
            return
        keys = sorted(keys)
        audio_matrix = np.vstack([self.audio_embeddings[k] for k in keys])
        symbolic_matrix = np.vstack([self.symbolic_embeddings[k] for k in keys])
        for modality, data_matrix in [("audio", audio_matrix), ("symbolic", symbolic_matrix)]:
            if data_matrix.shape[0] == 0:
                continue
            # Macierz podobieństw cosinusowych
            sim_matrix = cosine_similarity(data_matrix)
            plt.figure(figsize=(6, 5))
            plt.imshow(sim_matrix, aspect='auto', origin='lower', cmap='viridis')
            plt.title(f"Heatmapa podobieństw ({modality})")
            plt.colorbar(label="Podobieństwo cosinusowe")
            plt.xlabel("Wpisy")
            plt.ylabel("Wpisy")
            os.makedirs(self.output_dir, exist_ok=True)
            heatmap_file = os.path.join(self.output_dir, f"heatmap_{modality}.png")
            plt.savefig(heatmap_file)
            plt.close()
            logger.info(f"Heatmapa podobieństw dla {modality} zapisana w {heatmap_file}")

    def kolmogorov_smirnov_test(self):
        '''
        Przeprowadza dwuprobną analizę Kolmogorova-Smirnova dla rozkładów podobieństw cosinusowych w audio vs symbolic.
        '''
        keys = set(self.audio_embeddings.keys()) & set(self.symbolic_embeddings.keys())
        if not keys:
            logger.info("Brak wspólnych plików do testu Kolmogorov-Smirnova.")
            return
        keys = sorted(keys)
        audio_matrix = np.vstack([self.audio_embeddings[k] for k in keys])
        symbolic_matrix = np.vstack([self.symbolic_embeddings[k] for k in keys])
        # Rozkłady podobieństw (spłaszczenie macierzy podobieństw)
        audio_sims = cosine_similarity(audio_matrix).flatten()
        symbolic_sims = cosine_similarity(symbolic_matrix).flatten()
        # Test dwuproba Kolmogorova-Smirnova
        stat, p_value = ks_2samp(audio_sims, symbolic_sims)
        logger.info(f"Kolmogorov-Smirnov D: {stat:.4f}, p-value: {p_value:.4f}")
        os.makedirs(self.output_dir, exist_ok=True)
        ks_file = os.path.join(self.output_dir, "ks_test_results.txt")
        with open(ks_file, 'w') as f:
            f.write(f"D-statistic: {stat:.6f}\n")
            f.write(f"p-value: {p_value:.6f}\n")
        logger.info(f"Wyniki testu Kolmogorova-Smirnova zapisano w {ks_file}")

    def manova_test(self):
        '''
        Przeprowadza MANOVA (wielowymiarową analizę wariancji) porównującą wektory embeddingów audio i symbolic.
        '''
        keys = set(self.audio_embeddings.keys()) & set(self.symbolic_embeddings.keys())
        if not keys:
            logger.info("Brak wspólnych plików do testu MANOVA.")
            return
        keys = sorted(keys)
        audio_matrix = np.vstack([self.audio_embeddings[k] for k in keys])
        symbolic_matrix = np.vstack([self.symbolic_embeddings[k] for k in keys])
        n = audio_matrix.shape[0]
        dim = audio_matrix.shape[1]
        # Przygotowanie DataFrame dla MANOVA
        df = pd.DataFrame(
            np.vstack([audio_matrix, symbolic_matrix]),
            columns=[f"X{i}" for i in range(dim)]
        )
        df["group"] = ["audio"] * n + ["symbolic"] * n
        # Formuła MANOVA: wszystkie wymiary vs grupa
        dep_vars = "+".join([f"X{i}" for i in range(dim)])
        formula = dep_vars + " ~ group"
        maov = MANOVA.from_formula(formula, data=df)
        result = maov.mv_test()
        os.makedirs(self.output_dir, exist_ok=True)
        manova_file = os.path.join(self.output_dir, "manova_results.txt")
        with open(manova_file, 'w') as f:
            f.write(str(result))
        logger.info(f"Wyniki testu MANOVA zapisano w {manova_file}")