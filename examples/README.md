# MedVectors Usage Examples

This directory contains practical examples and recipes for using MedVectors in various applications.

## Available Examples

### 1. Basic Embeddings (`01_basic_embeddings.py`)
Demonstrates how to generate embeddings for medical text using MedVectors.

**Use Cases:**
- Generate embeddings for queries
- Generate embeddings for documents/corpus
- Batch processing for large datasets

**Run:**
```bash
python examples/01_basic_embeddings.py
```

### 2. Semantic Search (`02_semantic_search.py`)
Shows how to build a semantic search system for retrieving relevant medical documents.

**Use Cases:**
- Medical document search
- Clinical note retrieval
- Medical knowledge base search

**Run:**
```bash
python examples/02_semantic_search.py
```

### 3. Clustering (`03_clustering.py`)
Demonstrates clustering medical documents using MedVectors embeddings.

**Use Cases:**
- Group similar medical concepts
- Discover disease categories
- Organize clinical notes
- Find patterns in medical literature

**Run:**
```bash
python examples/03_clustering.py
```

### 4. Classification (`04_classification.py`)
Shows how to use MedVectors embeddings for classification tasks.

**Use Cases:**
- Disease category prediction
- Medical document classification
- Symptom-based triage
- Medical specialty routing

**Run:**
```bash
python examples/04_classification.py
```

### 5. RAG System (`05_rag_system.py`)
Demonstrates building a Retrieval-Augmented Generation system.

**Use Cases:**
- Medical question answering
- Clinical decision support
- Medical literature search
- Knowledge base retrieval

**Run:**
```bash
python examples/05_rag_system.py
```

### 6. Efficiency Comparison (`06_efficiency_comparison.py`)
Compares MedVectors (embedding model) efficiency vs all-mini-lm-v2 (language model).

**Use Cases:**
- Understanding why embeddings are better for retrieval
- Benchmarking efficiency vs language models
- Making architectural decisions for your application

**Run:**
```bash
python examples/06_efficiency_comparison.py
```

## Installation

Before running the examples, install the required dependencies:

```bash
pip install torch transformers scikit-learn numpy
```

For additional dependencies:
```bash
pip install matplotlib  # For visualization examples
```

## Model Options

All examples support different model sizes:

- `kiranbeethoju/MedVectors-small-v0.1` - Lightweight, faster inference
- `kiranbeethoju/MedVectors-base-v0.1` - Balanced performance
- `kiranbeethoju/MedVectors-large-v0.1` - Highest accuracy

Replace the `model_name` parameter in any example to use a different size.

## Quick Start

### Generate Embeddings

```python
from examples.01_basic_embeddings import MedVectorsEncoder

# Initialize encoder
encoder = MedVectorsEncoder(model_name="kiranbeethoju/MedVectors-base-v0.1")

# Generate embedding
text = "Patient presents with chest pain and elevated cardiac enzymes."
embedding = encoder.encode(text)
print(f"Embedding shape: {embedding.shape}")
```

### Semantic Search

```python
from examples.02_semantic_search import MedVectorsRetriever

# Initialize retriever
retriever = MedVectorsRetriever()

# Add documents to corpus
documents = [
    {'id': 1, 'text': 'Type 2 diabetes is a chronic condition...'},
    {'id': 2, 'text': 'Hypertension affects nearly half of adults...'},
]
retriever.add_to_corpus(documents)

# Search
query = "What causes high blood sugar?"
results = retriever.search(query, top_k=3)

for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Text: {result['document']['text']}")
```

## Advanced Usage

### Custom Device Selection

```python
# Force CPU
encoder = MedVectorsEncoder(device="cpu")

# Force CUDA
encoder = MedVectorsEncoder(device="cuda")

# Force MPS (Mac)
encoder = MedVectorsEncoder(device="mps")
```

### Batch Processing

```python
# Process many texts efficiently
texts = [...1000 medical texts...]
embeddings = encoder.encode(texts, batch_size=64)
```

### Saving/Loading Embeddings

```python
import numpy as np
import pickle

# Save embeddings
np.save('embeddings.npy', embeddings)

# Load embeddings
loaded_embeddings = np.load('embeddings.npy')
```

## Tips for Best Results

1. **Use the right model size:**
   - Small for real-time applications
   - Base for general use
   - Large for maximum accuracy

2. **Normalize embeddings:**
   ```python
   embedding = encoder.encode(text, normalize=True)
   ```
   All examples normalize by default for cosine similarity.

3. **Batch processing:**
   Process multiple texts together for better performance.

4. **Device optimization:**
   - Use MPS on Apple Silicon (Mac M1/M2/M3)
   - Use CUDA on NVIDIA GPUs
   - CPU is acceptable for small batches

## Common Issues

### Out of Memory
Reduce batch size or use a smaller model:
```python
encoder = MedVectorsEncoder(
    model_name="kiranbeethoju/MedVectors-small-v0.1",
    max_length=256  # Reduce from 512
)
embeddings = encoder.encode(texts, batch_size=16)  # Reduce from 32
```

### Slow Inference
- Ensure MPS/CUDA is being used (check device output)
- Reduce `max_length` if text is shorter
- Use the small model for faster results

## Contributing

Have a great use case for MedVectors? Submit your example as a pull request!

## License

These examples are part of the MedVectors project.
