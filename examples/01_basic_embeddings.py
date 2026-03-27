"""
MedVectors Basic Embeddings Example
====================================

This example demonstrates how to generate embeddings for medical text
using the pre-trained MedVectors model from Hugging Face.

Use Cases:
- Generate embeddings for queries
- Generate embeddings for documents/corpus
- Batch processing for large datasets
"""

import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Union
import numpy as np


class MedVectorsEncoder:
    """Simple wrapper for MedVectors embedding generation."""

    def __init__(
        self,
        model_name: str = "abhinand/MedVectors-base-v0.1",
        device: str = None,
        max_length: int = 512
    ):
        """
        Initialize MedVectors encoder.

        Args:
            model_name: Hugging Face model name (small, base, or large)
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

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Encode text(s) into embeddings.

        Args:
            texts: Single text string or list of texts
            normalize: Whether to L2-normalize embeddings
            batch_size: Batch size for processing multiple texts

        Returns:
            Numpy array of shape (num_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Use mean pooling of last hidden states
            embeddings = self._mean_pooling(
                outputs.last_hidden_state,
                inputs["attention_mask"]
            )

            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().numpy())

        # Concatenate all batches
        embeddings = np.concatenate(all_embeddings, axis=0)

        # Return single vector if only one text
        if len(embeddings) == 1:
            embeddings = embeddings[0]

        return embeddings

    def _mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean pooling using attention mask to handle variable length sequences.

        Args:
            token_embeddings: Token embeddings [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Pooled embeddings [batch_size, hidden_dim]
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


def example_basic_usage():
    """Example: Generate embeddings for individual texts."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Embedding Generation")
    print("=" * 70)

    # Initialize encoder
    encoder = MedVectorsEncoder(model_name="abhinand/MedVectors-base-v0.1")

    # Sample medical texts
    query = "What are the symptoms of myocardial infarction?"
    document = "Myocardial infarction, commonly known as a heart attack, presents with chest pain, shortness of breath, sweating, nausea, and palpitations. The pain typically radiates to the left arm, jaw, or back."

    # Generate embeddings
    query_embedding = encoder.encode(query)
    doc_embedding = encoder.encode(document)

    print(f"\nQuery: {query}")
    print(f"Query embedding shape: {query_embedding.shape}")
    print(f"Query embedding (first 5 values): {query_embedding[:5]}")

    print(f"\nDocument length: {len(document)} characters")
    print(f"Document embedding shape: {doc_embedding.shape}")

    # Compute cosine similarity
    similarity = np.dot(query_embedding, doc_embedding)
    print(f"\nCosine similarity: {similarity:.4f}")


def example_batch_processing():
    """Example: Generate embeddings for multiple texts in batch."""
    print("\n" + "=" * 70)
    print("Example 2: Batch Embedding Generation")
    print("=" * 70)

    # Initialize encoder
    encoder = MedVectorsEncoder(model_name="abhinand/MedVectors-small-v0.1")

    # Sample medical documents (simulating a small corpus)
    corpus = [
        "Type 2 diabetes is a chronic condition characterized by insulin resistance and high blood sugar levels.",
        "Hypertension is defined as a systolic blood pressure of 130 mmHg or higher, or diastolic pressure of 80 mmHg or higher.",
        "Asthma is a chronic inflammatory disease of the airways characterized by variable and recurring symptoms.",
        "Pneumonia is an inflammatory condition of the lung primarily affecting the microscopic air sacs.",
        "Osteoarthritis is a type of joint disease that results from breakdown of joint cartilage and underlying bone.",
        "Alzheimer's disease is a neurodegenerative disorder characterized by progressive cognitive decline.",
        "Chronic obstructive pulmonary disease (COPD) is a chronic inflammatory lung disease causing obstructed airflow.",
        "Migraine is a primary headache disorder characterized by recurrent moderate to severe headaches."
    ]

    # Generate embeddings for entire corpus
    print(f"Processing {len(corpus)} documents...")
    embeddings = encoder.encode(corpus, batch_size=4)

    print(f"\nCorpus embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    # Show pairwise similarities
    print("\nPairwise cosine similarities:")
    print("-" * 50)
    for i in range(min(3, len(corpus))):
        for j in range(i + 1, min(i + 3, len(corpus))):
            sim = np.dot(embeddings[i], embeddings[j])
            print(f"Doc {i} <-> Doc {j}: {sim:.4f}")


def example_model_comparison():
    """Example: Compare embeddings from different model sizes."""
    print("\n" + "=" * 70)
    print("Example 3: Model Size Comparison")
    print("=" * 70)

    # Test text
    test_text = "Patient presents with acute myocardial infarction and requires immediate intervention."

    model_sizes = [
        "abhinand/MedVectors-small-v0.1",
        "abhinand/MedVectors-base-v0.1",
        # "abhinand/MedVectors-large-v0.1"  # Uncomment if large model is available
    ]

    for model_name in model_sizes:
        print(f"\nLoading {model_name.split('/')[-1]}...")
        encoder = MedVectorsEncoder(model_name=model_name)

        # Generate embedding
        embedding = encoder.encode(test_text)

        print(f"  Embedding dimension: {embedding.shape[0]}")
        print(f"  Norm (L2): {np.linalg.norm(embedding):.4f}")
        print(f"  Mean value: {np.mean(embedding):.4f}")
        print(f"  Std value: {np.std(embedding):.4f}")


def example_query_document_encoding():
    """Example: Encoding queries and documents for retrieval."""
    print("\n" + "=" * 70)
    print("Example 4: Query-Document Encoding")
    print("=" * 70)

    # Initialize encoder
    encoder = MedVectorsEncoder(model_name="abhinand/MedVectors-base-v0.1")

    # Sample queries (what a user might search)
    queries = [
        "What is the treatment for atrial fibrillation?",
        "Symptoms of type 2 diabetes",
        "How to prevent hypertension?"
    ]

    # Sample documents (knowledge base)
    documents = [
        "Atrial fibrillation treatment includes rate control with beta-blockers or calcium channel blockers, rhythm control with antiarrhythmic drugs, and anticoagulation to prevent stroke.",
        "Type 2 diabetes symptoms include increased thirst, frequent urination, unexplained weight loss, fatigue, blurred vision, and slow-healing sores.",
        "Prevention of hypertension includes maintaining a healthy weight, regular exercise, reducing sodium intake, limiting alcohol, and managing stress."
    ]

    # Encode
    print(f"Encoding {len(queries)} queries and {len(documents)} documents...")
    query_embeddings = encoder.encode(queries)
    doc_embeddings = encoder.encode(documents)

    # Compute query-document similarity matrix
    similarity_matrix = np.dot(query_embeddings, doc_embeddings.T)

    print("\nQuery-Document Similarity Matrix:")
    print("-" * 60)
    for i, query in enumerate(queries):
        print(f"\nQuery {i + 1}: {query[:40]}...")
        for j, doc in enumerate(documents):
            print(f"  Doc {j + 1}: {similarity_matrix[i, j]:.4f}")

    # Show best matches
    print("\nBest matches:")
    print("-" * 60)
    for i in range(len(queries)):
        best_doc_idx = np.argmax(similarity_matrix[i])
        print(f"Query {i + 1} -> Document {best_doc_idx + 1} (sim: {similarity_matrix[i, best_doc_idx]:.4f})")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("MedVectors Basic Embeddings Examples")
    print("=" * 70)

    # Run examples
    example_basic_usage()
    example_batch_processing()
    example_model_comparison()
    example_query_document_encoding()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
