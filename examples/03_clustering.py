"""
MedVectors Clustering Example
=============================

This example demonstrates clustering medical documents using MedVectors embeddings.

Use Cases:
- Group similar medical concepts
- Discover disease categories
- Organize clinical notes
- Find patterns in medical literature
"""

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from typing import List, Dict
import json


class MedVectorsClusterer:
    """Cluster medical documents using MedVectors embeddings."""

    def __init__(
        self,
        model_name: str = "kiranbeethoju/MedVectors-base-v0.1",
        device: str = None,
        max_length: int = 512
    ):
        """
        Initialize MedVectors clusterer.

        Args:
            model_name: Hugging Face model name
            device: Device to use (cuda, mps, cpu). Auto-detects if None
            max_length: Maximum sequence length for tokenization
        """
        self.model_name = model_name
        self.max_length = max_length

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """Encode texts into embeddings."""
        all_embeddings = []

        for batch_texts in [texts[i:i + 32] for i in range(0, len(texts), 32)]:
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Mean pooling
            embeddings = self._mean_pooling(
                outputs.last_hidden_state,
                inputs["attention_mask"]
            )

            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def _mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean pooling using attention mask."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def kmeans_cluster(
        self,
        texts: List[str],
        n_clusters: int = 3,
        random_state: int = 42
    ) -> Dict:
        """
        Cluster texts using K-Means.

        Args:
            texts: List of texts to cluster
            n_clusters: Number of clusters
            random_state: Random seed for reproducibility

        Returns:
            Dict with cluster assignments and other metrics
        """
        # Encode texts
        embeddings = self.encode(texts)

        # Apply K-Means
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )
        cluster_labels = kmeans.fit_predict(embeddings)

        # Compute silhouette score
        silhouette = silhouette_score(embeddings, cluster_labels)

        return {
            'cluster_labels': cluster_labels,
            'centroids': kmeans.cluster_centers_,
            'silhouette_score': silhouette,
            'n_clusters': n_clusters,
            'embeddings': embeddings
        }

    def dbscan_cluster(
        self,
        texts: List[str],
        eps: float = 0.5,
        min_samples: int = 2
    ) -> Dict:
        """
        Cluster texts using DBSCAN (density-based clustering).

        Args:
            texts: List of texts to cluster
            eps: Maximum distance between samples
            min_samples: Minimum samples in a neighborhood

        Returns:
            Dict with cluster assignments and other metrics
        """
        # Encode texts
        embeddings = self.encode(texts)

        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        cluster_labels = dbscan.fit_predict(embeddings)

        # Count clusters (excluding noise labeled as -1)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)

        # Compute silhouette score (only if more than 1 cluster)
        silhouette = None
        if n_clusters > 1:
            # Filter out noise points
            mask = cluster_labels != -1
            if mask.sum() > 1:
                silhouette = silhouette_score(
                    embeddings[mask],
                    cluster_labels[mask]
                )

        return {
            'cluster_labels': cluster_labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette_score': silhouette,
            'eps': eps,
            'min_samples': min_samples,
            'embeddings': embeddings
        }

    def hierarchical_cluster(
        self,
        texts: List[str],
        n_clusters: int = 3,
        linkage: str = 'ward'
    ) -> Dict:
        """
        Cluster texts using hierarchical clustering.

        Args:
            texts: List of texts to cluster
            n_clusters: Number of clusters
            linkage: Linkage criterion ('ward', 'complete', 'average', 'single')

        Returns:
            Dict with cluster assignments and other metrics
        """
        # Encode texts
        embeddings = self.encode(texts)

        # Apply hierarchical clustering
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )
        cluster_labels = hierarchical.fit_predict(embeddings)

        # Compute silhouette score
        silhouette = silhouette_score(embeddings, cluster_labels)

        return {
            'cluster_labels': cluster_labels,
            'silhouette_score': silhouette,
            'n_clusters': n_clusters,
            'linkage': linkage,
            'embeddings': embeddings
        }

    def get_cluster_contents(
        self,
        texts: List[str],
        cluster_labels: np.ndarray,
        cluster_id: int = None
    ) -> Dict:
        """
        Get contents of a specific cluster or all clusters.

        Args:
            texts: Original texts
            cluster_labels: Cluster assignments
            cluster_id: Specific cluster to return (None for all)

        Returns:
            Dict mapping cluster IDs to list of texts
        """
        unique_labels = np.unique(cluster_labels)
        contents = {}

        if cluster_id is not None:
            unique_labels = [cluster_id]

        for label in unique_labels:
            if label == -1:
                name = "Noise"
            else:
                name = f"Cluster {label}"

            indices = np.where(cluster_labels == label)[0]
            contents[name] = [texts[i] for i in indices]

        return contents

    def print_cluster_summary(
        self,
        texts: List[str],
        cluster_labels: np.ndarray,
        max_display: int = 5
    ):
        """
        Print a summary of clusters.

        Args:
            texts: Original texts
            cluster_labels: Cluster assignments
            max_display: Maximum items to show per cluster
        """
        unique_labels = np.unique(cluster_labels)

        print("\n" + "=" * 70)
        print("CLUSTER SUMMARY")
        print("=" * 70)

        for label in sorted(unique_labels):
            if label == -1:
                name = "Noise"
            else:
                name = f"Cluster {label}"

            indices = np.where(cluster_labels == label)[0]
            cluster_texts = [texts[i] for i in indices]

            print(f"\n{name} ({len(cluster_texts)} items)")
            print("-" * 70)

            for i, text in enumerate(cluster_texts[:max_display]):
                print(f"{i + 1}. {text[:100]}...")

            if len(cluster_texts) > max_display:
                print(f"  ... and {len(cluster_texts) - max_display} more items")


def example_kmeans_clustering():
    """Example: K-Means clustering of medical texts."""
    print("\n" + "=" * 70)
    print("Example 1: K-Means Clustering")
    print("=" * 70)

    clusterer = MedVectorsClusterer(model_name="kiranbeethoju/MedVectors-small-v0.1")

    # Sample medical texts from different categories
    texts = [
        # Cardiovascular
        "Myocardial infarction presents with chest pain and elevated cardiac enzymes.",
        "Hypertension is defined as blood pressure above 130/80 mmHg.",
        "Atrial fibrillation causes irregular heart rhythms and increases stroke risk.",
        "Heart failure symptoms include shortness of breath and edema.",

        # Endocrine
        "Type 2 diabetes results from insulin resistance and high blood glucose.",
        "Hypothyroidism causes fatigue, weight gain, and cold intolerance.",
        "Type 1 diabetes requires insulin therapy due to autoimmune destruction.",
        "Hyperthyroidism can cause weight loss, tachycardia, and tremors.",

        # Respiratory
        "Asthma is characterized by reversible airway obstruction and wheezing.",
        "Pneumonia inflames the alveoli and causes fever and cough.",
        "COPD includes chronic bronchitis and emphysema.",
        "Pulmonary embolism occurs when a blood clot blocks pulmonary arteries."
    ]

    # Cluster into 3 groups
    results = clusterer.kmeans_cluster(texts, n_clusters=3)

    print(f"\nSilhouette Score: {results['silhouette_score']:.4f}")

    # Print cluster summary
    clusterer.print_cluster_summary(texts, results['cluster_labels'])


def example_disease_categorization():
    """Example: Categorize diseases by body system."""
    print("\n" + "=" * 70)
    print("Example 2: Disease Categorization")
    print("=" * 70)

    clusterer = MedVectorsClusterer(model_name="kiranbeethoju/MedVectors-base-v0.1")

    # Various diseases
    diseases = [
        "Coronary artery disease",
        "Stroke",
        "Arrhythmia",
        "Diabetes mellitus",
        "Thyroid cancer",
        "Adrenal insufficiency",
        "Bronchitis",
        "Emphysema",
        "Pleural effusion",
        "Gastritis",
        "Hepatitis",
        "Crohn's disease",
        "Glomerulonephritis",
        "Kidney stones",
        "Cystitis",
        "Osteoporosis",
        "Rheumatoid arthritis",
        "Gout",
        "Alzheimer's disease",
        "Parkinson's disease",
        "Epilepsy",
        "Migraine",
        "Depression",
        "Anxiety disorder"
    ]

    # Cluster diseases
    results = clusterer.kmeans_cluster(diseases, n_clusters=6)

    print(f"\nSilhouette Score: {results['silhouette_score']:.4f}")
    clusterer.print_cluster_summary(diseases, results['cluster_labels'])


def example_clinical_notes_clustering():
    """Example: Cluster clinical notes by topic."""
    print("\n" + "=" * 70)
    print("Example 3: Clinical Notes Clustering")
    print("=" * 70)

    clusterer = MedVectorsClusterer(model_name="kiranbeethoju/MedVectors-small-v0.1")

    # Clinical note excerpts
    notes = [
        "Patient presents with chest pain radiating to left arm. ECG shows ST elevation. Troponin elevated.",
        "Chief complaint: headache and blurred vision. BP 180/110 mmHg. Started on antihypertensives.",
        "Patient reports wheezing and shortness of breath. Peak flow reduced. Inhaled corticosteroids prescribed.",
        "Fasting glucose 180 mg/dL, HbA1c 8.5%. Polyuria and polydipsia present. Diabetes type 2 diagnosed.",
        "Right knee pain worse with activity. X-ray shows joint space narrowing. NSAIDs recommended.",
        "Sudden onset right-sided weakness. Speech difficulty. MRI confirms left MCA infarct.",
        "Chronic dry cough. CT scan shows pulmonary fibrosis. Pulmonology referral made.",
        "Patient reports fatigue and weight gain. TSH elevated. Levothyroxine started.",
        "Recurrent UTI with fever. Urinalysis positive for WBCs. Antibiotics prescribed.",
        "Abdominal pain and bloating. Colonoscopy reveals ulcerative colitis. Mesalamine initiated."
    ]

    # Cluster notes
    results = clusterer.hierarchical_cluster(notes, n_clusters=4, linkage='ward')

    print(f"\nSilhouette Score: {results['silhouette_score']:.4f}")
    clusterer.print_cluster_summary(notes, results['cluster_labels'])


def example_dbscan_anomaly_detection():
    """Example: Detect anomalies/outliers using DBSCAN."""
    print("\n" + "=" * 70)
    print("Example 4: Anomaly Detection with DBSCAN")
    print("=" * 70)

    clusterer = MedVectorsClusterer(model_name="kiranbeethoju/MedVectors-small-v0.1")

    # Texts with one outlier
    texts = [
        "Patient has chest pain and elevated cardiac markers.",
        "Chest pain with ST elevation on ECG.",
        "Patient reports severe chest pain radiating to jaw.",
        "Acute MI management: antiplatelets, beta-blockers, and statins.",
        "Chest pain evaluation includes ECG, cardiac enzymes, and imaging.",
        # Outlier - different topic
        "Patient presents with severe allergic reaction and anaphylaxis."
    ]

    # Use DBSCAN to find outliers
    results = clusterer.dbscan_cluster(texts, eps=0.4, min_samples=2)

    print(f"\nNumber of clusters: {results['n_clusters']}")
    print(f"Number of outliers (noise): {results['n_noise']}")

    if results['silhouette_score']:
        print(f"Silhouette Score: {results['silhouette_score']:.4f}")

    clusterer.print_cluster_summary(texts, results['cluster_labels'])


def example_optimal_clusters():
    """Example: Find optimal number of clusters."""
    print("\n" + "=" * 70)
    print("Example 5: Finding Optimal Number of Clusters")
    print("=" * 70)

    clusterer = MedVectorsClusterer(model_name="kiranbeethoju/MedVectors-small-v0.1")

    texts = [
        "Heart attack symptoms include chest pain, shortness of breath, and sweating.",
        "Stroke presents with sudden weakness, confusion, and speech difficulty.",
        "Arrhythmia causes irregular heartbeat and palpitations.",
        "Heart failure results in fluid retention and fatigue.",
        "Diabetes symptoms include thirst, frequent urination, and weight loss.",
        "Hypothyroidism causes cold intolerance and weight gain.",
        "Hyperthyroidism leads to weight loss and rapid heartbeat.",
        "Adrenal insufficiency presents with fatigue and low blood pressure.",
        "Asthma causes wheezing, coughing, and shortness of breath.",
        "Pneumonia symptoms include fever, cough, and chest pain.",
        "COPD is characterized by chronic airflow limitation.",
        "Bronchitis causes inflammation of the bronchial tubes."
    ]

    # Try different numbers of clusters
    print("\nTesting different numbers of clusters:")
    print("-" * 70)

    best_score = -1
    best_k = 2

    for k in range(2, 8):
        results = clusterer.kmeans_cluster(texts, n_clusters=k, random_state=42)
        score = results['silhouette_score']

        print(f"K={k}: Silhouette Score = {score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    print(f"\n✅ Optimal K: {best_k} (Silhouette Score: {best_score:.4f})")

    # Show clusters with optimal K
    results = clusterer.kmeans_cluster(texts, n_clusters=best_k, random_state=42)
    clusterer.print_cluster_summary(texts, results['cluster_labels'])


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("MedVectors Clustering Examples")
    print("=" * 70)

    example_kmeans_clustering()
    example_disease_categorization()
    example_clinical_notes_clustering()
    example_dbscan_anomaly_detection()
    example_optimal_clusters()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
