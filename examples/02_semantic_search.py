"""
MedVectors Semantic Search Example
================================

This example demonstrates building a semantic search system using MedVectors
for retrieving relevant medical documents.

Use Cases:
- Medical document search
- Clinical note retrieval
- Medical knowledge base search
"""

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Dict, Tuple
import json


class MedVectorsRetriever:
    """Semantic search retriever using MedVectors."""

    def __init__(
        self,
        model_name: str = "kiranbeethoju/MedVectors-base-v0.1",
        device: str = None,
        max_length: int = 512
    ):
        """
        Initialize MedVectors retriever.

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

        # Initialize empty corpus
        self.corpus = []
        self.corpus_embeddings = None

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

    def add_to_corpus(self, documents: List[Dict]):
        """
        Add documents to the search corpus.

        Args:
            documents: List of dicts with 'id' and 'text' keys
        """
        self.corpus = documents
        texts = [doc['text'] for doc in documents]
        print(f"Encoding {len(texts)} documents...")
        self.corpus_embeddings = self.encode(texts)
        print(f"Corpus ready: {len(self.corpus)} documents")

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[Dict]:
        """
        Search the corpus for relevant documents.

        Args:
            query: Search query
            top_k: Number of top results to return
            min_score: Minimum similarity score threshold

        Returns:
            List of dicts with doc info and similarity scores
        """
        if self.corpus_embeddings is None:
            raise ValueError("Corpus is empty. Call add_to_corpus() first.")

        # Encode query
        query_embedding = self.encode([query])[0]

        # Compute cosine similarity
        scores = np.dot(self.corpus_embeddings, query_embedding)

        # Get top-k results
        top_indices = np.argsort(-scores)[:top_k]

        # Filter by minimum score
        results = []
        for idx in top_indices:
            if scores[idx] >= min_score:
                results.append({
                    'document': self.corpus[idx],
                    'score': float(scores[idx]),
                    'rank': len(results) + 1
                })

        return results

    def search_batch(
        self,
        queries: List[str],
        top_k: int = 5
    ) -> Dict[str, List[Dict]]:
        """Search for multiple queries at once."""
        query_embeddings = self.encode(queries)

        results = {}
        for i, query in enumerate(queries):
            scores = np.dot(self.corpus_embeddings, query_embeddings[i])
            top_indices = np.argsort(-scores)[:top_k]

            results[query] = []
            for idx in top_indices:
                results[query].append({
                    'document': self.corpus[idx],
                    'score': float(scores[idx])
                })

        return results


def example_basic_search():
    """Example: Basic semantic search."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Semantic Search")
    print("=" * 70)

    # Initialize retriever
    retriever = MedVectorsRetriever(model_name="kiranbeethoju/MedVectors-base-v0.1")

    # Sample medical corpus
    documents = [
        {'id': 1, 'text': 'Type 2 diabetes is a chronic condition where the body cannot use insulin properly.'},
        {'id': 2, 'text': 'Hypertension, or high blood pressure, affects nearly half of American adults.'},
        {'id': 3, 'text': 'Asthma causes airways to narrow and swell, producing extra mucus.'},
        {'id': 4, 'text': 'Myocardial infarction, or heart attack, occurs when blood flow to the heart is blocked.'},
        {'id': 5, 'text': 'Pneumonia is an infection that inflames air sacs in one or both lungs.'},
        {'id': 6, 'text': 'Osteoarthritis is the most common type of arthritis, affecting millions.'},
        {'id': 7, 'text': 'Depression is a mood disorder that causes persistent sadness and loss of interest.'},
        {'id': 8, 'text': 'Stroke occurs when blood supply to part of the brain is interrupted.'},
    ]

    # Add to corpus
    retriever.add_to_corpus(documents)

    # Search query
    query = "What are the symptoms of a heart attack?"
    print(f"\nQuery: {query}\n")

    results = retriever.search(query, top_k=3)

    print("Top Results:")
    print("-" * 70)
    for result in results:
        print(f"\nRank {result['rank']} (Score: {result['score']:.4f})")
        print(f"  {result['document']['text']}")


def example_clinical_search():
    """Example: Clinical document search."""
    print("\n" + "=" * 70)
    print("Example 2: Clinical Document Search")
    print("=" * 70)

    retriever = MedVectorsRetriever(model_name="kiranbeethoju/MedVectors-base-v0.1")

    # Clinical notes corpus
    documents = [
        {
            'id': 'pt_001',
            'text': 'Patient presents with chest pain radiating to left arm, elevated troponin levels, and ST-segment elevation on ECG. Admitted for acute myocardial infarction.'
        },
        {
            'id': 'pt_002',
            'text': 'Patient with HbA1c of 8.2%, complaints of polyuria and polydipsia. Diagnosed with uncontrolled type 2 diabetes.'
        },
        {
            'id': 'pt_003',
            'text': 'Patient reports episodic wheezing, shortness of breath, and nighttime coughing. Spirometry shows reversible airflow obstruction consistent with asthma.'
        },
        {
            'id': 'pt_004',
            'text': 'BP 165/95 mmHg on multiple readings, headache, and visual disturbances. Started on lisinopril for hypertension.'
        },
        {
            'id': 'pt_005',
            'text': 'Patient with knee pain worsened by activity, X-ray shows joint space narrowing and osteophytes. Diagnosed with osteoarthritis.'
        },
    ]

    retriever.add_to_corpus(documents)

    # Clinical queries
    queries = [
        "Treatment for acute MI",
        "Uncontrolled diabetes symptoms",
        "Asthma diagnosis criteria",
        "Hypertension management",
        "Osteoarthritis knee pain"
    ]

    print(f"Searching for {len(queries)} clinical queries...\n")

    for query in queries:
        results = retriever.search(query, top_k=2)
        print(f"Query: {query}")
        print(f"Top Match (Score: {results[0]['score']:.4f})")
        print(f"  {results[0]['document']['text'][:80]}...")
        print()


def example_threshold_search():
    """Example: Search with minimum score threshold."""
    print("\n" + "=" * 70)
    print("Example 3: Search with Threshold")
    print("=" * 70)

    retriever = MedVectorsRetriever(model_name="kiranbeethoju/MedVectors-small-v0.1")

    documents = [
        {'id': 1, 'text': 'Insulin is a hormone that regulates blood sugar levels.'},
        {'id': 2, 'text': 'Glucagon is a hormone that raises blood sugar levels.'},
        {'id': 3, 'text': 'The heart has four chambers: two atria and two ventricles.'},
        {'id': 4, 'text': 'The brain consists of the cerebrum, cerebellum, and brainstem.'},
        {'id': 5, 'text': 'The liver performs over 500 functions including detoxification and protein synthesis.'},
    ]

    retriever.add_to_corpus(documents)

    # Search with threshold
    query = "What regulates blood glucose?"
    results = retriever.search(query, top_k=5, min_score=0.5)

    print(f"Query: {query}")
    print(f"Results with similarity >= 0.5:\n")

    if results:
        for result in results:
            print(f"{result['document']['id']}. {result['document']['text']}")
            print(f"   Score: {result['score']:.4f}\n")
    else:
        print("No results found above threshold.")


def example_save_load_corpus():
    """Example: Save and load corpus with embeddings."""
    print("\n" + "=" * 70)
    print("Example 4: Save/Load Corpus")
    print("=" * 70)

    import pickle

    # Create and save corpus
    retriever = MedVectorsRetriever(model_name="kiranbeethoju/MedVectors-base-v0.1")

    documents = [
        {'id': 1, 'text': 'Sample medical document 1.'},
        {'id': 2, 'text': 'Sample medical document 2.'},
    ]

    retriever.add_to_corpus(documents)

    # Save to file
    corpus_data = {
        'documents': retriever.corpus,
        'embeddings': retriever.corpus_embeddings
    }

    with open('corpus_cache.pkl', 'wb') as f:
        pickle.dump(corpus_data, f)

    print("✅ Corpus saved to corpus_cache.pkl")

    # Load from file
    with open('corpus_cache.pkl', 'rb') as f:
        loaded_data = pickle.load(f)

    # Create new retriever and load
    retriever2 = MedVectorsRetriever(model_name="kiranbeethoju/MedVectors-base-v0.1")
    retriever2.corpus = loaded_data['documents']
    retriever2.corpus_embeddings = loaded_data['embeddings']

    print("✅ Corpus loaded from cache")

    # Verify
    query = "Sample query"
    results = retriever2.search(query, top_k=2)
    print(f"\nFound {len(results)} results")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("MedVectors Semantic Search Examples")
    print("=" * 70)

    example_basic_search()
    example_clinical_search()
    example_threshold_search()
    example_save_load_corpus()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
