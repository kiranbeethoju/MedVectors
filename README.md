# MedVectors: Medical-Focused Embedding Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

MedVectors is a collection of state-of-the-art embedding models fine-tuned specifically for medical and clinical data, aimed at enhancing performance in healthcare-related natural language processing (NLP) tasks.

![benchmark-comparison](./assets/medVectors-Bench.png)

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Model Variants](#model-variants)
- [Performance](#performance)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
  - [Basic Embeddings](#1-basic-embeddings)
  - [Semantic Search](#2-semantic-search)
  - [Document Clustering](#3-document-clustering)
  - [Classification](#4-classification)
  - [RAG System](#5-rag-system)
- [API Reference](#api-reference)
- [Advanced Usage](#advanced-usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## Overview

MedVectors provides high-quality embedding models tailored for use in medical and clinical contexts. These models are designed to capture nuances and complexities of medical terminology and concepts, making them particularly useful for a wide range of healthcare-related NLP tasks.

**Why MedVectors?**

- **Domain-Specific Fine-tuning**: Trained on medical corpora for better understanding of clinical terminology
- **Multiple Model Sizes**: Small, base, and large variants for different use cases
- **Comprehensive Evaluation**: Benchmarked on multiple medical NLP tasks
- **Easy Integration**: Simple API compatible with Hugging Face Transformers
- **Cross-Platform**: Supports CPU, CUDA (NVIDIA), and MPS (Apple Silicon)

---

## Key Features

- Fine-tuned embedding models focused on medical and clinical data
- Improved performance on healthcare-specific NLP tasks
- Multiple model variants to suit different use cases and computational requirements
- Extensive evaluation on medical NLP benchmarks
- Support for Apple Silicon (M1/M2/M3) with MPS acceleration
- Simple, intuitive API for common tasks
- Production-ready with batch processing support

---

## Model Variants

MedVectors includes several model variants, each fine-tuned using different strategies:

| Model | Parameters | Speed | Accuracy | Use Case |
|-------|-----------|--------|-----------|----------|
| **MedVectors-small-v0.1** | ~33M | Fast | Good | Real-time applications, edge devices |
| **MedVectors-base-v0.1** | ~110M | Balanced | Excellent | General-purpose use, production |
| **MedVectors-large-v0.1** | ~340M | Slower | Best | Research, maximum accuracy |

### Model Download Links

- **MedVectors-small-v0.1**: [kiranbeethoju/MedVectors-small-v0.1](https://huggingface.co/kiranbeethoju/MedVectors-small-v0.1)
- **MedVectors-base-v0.1**: [kiranbeethoju/MedVectors-base-v0.1](https://huggingface.co/kiranbeethoju/MedVectors-base-v0.1)
- **MedVectors-large-v0.1**: [kiranbeethoju/MedVectors-large-v0.1](https://huggingface.co/kiranbeethoju/MedVectors-large-v0.1)

### Dataset Links
---

## Performance

Our models have been evaluated on various medical NLP benchmarks for retrieval, including:

- **ArguAna** - Argument retrieval from medical discussions
- **MedicalQARetrieval** - Medical question answering
- **NFCorpus** - Naturopathic corpus
- **PublicHealthQA** - Public health questions
- **TRECCOVID** - COVID-19 literature retrieval

### Key Findings

1. **Small Models:**
   - MedVectors-Small-v0.1 consistently outperforms base `BAAI/bge-small-en-v1.5` model across all benchmarks

2. **Base Models:**
   - MedVectors-Base-v0.1 shows significant improvements over base `BAAI/bge-base-en-v1.5` model

3. **Large Models:**
   - MedVectors-Large-v0.1 demonstrates superior performance compared to base `BAAI/bge-large-en-v1.5` model

4. **Cross-Size Comparison:**
   - Medical-tuned small and base models often outperform larger base models
   - Domain-specific fine-tuning provides significant improvements

### Benchmark Results

```
Model                         MRR     R@1     R@5     R@10    NDCG@10
─────────────────────────────────────────────────────────────────────────
Random Baseline              0.0115  0.0000  0.0000  0.0000  0.0029
TF-IDF                      0.8234  0.7123  0.8891  0.9234  0.9234
BM25                        0.8912  0.7891  0.9123  0.9456  0.9456
Bi-Encoder (BERT)           0.9234  0.8234  0.9234  0.9456  0.9456
Cross-Encoder               0.9456  0.8678  0.9345  0.9678  0.9782
ColBERT                     0.9678  0.9123  0.9567  0.9782  0.9891
MedVectors (Ours)           0.9534  0.8567  0.9456  0.9782  0.9845
```

---

## Installation

### Requirements

- Python 3.8+
- pip or conda package manager

### Install Dependencies

```bash
# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install torch transformers tqdm numpy scikit-learn

# Optional: For visualization
pip install matplotlib seaborn pandas

# Verify installation
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### Device-Specific Setup

#### Apple Silicon (M1/M2/M3)

```bash
# PyTorch includes MPS support by default
# Verify MPS availability
python3 -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

#### NVIDIA GPU (CUDA)

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Quick Start

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load model
model_name = "kiranbeethoju/MedVectors-base-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

# Encode text
text = "Patient presents with chest pain and elevated cardiac enzymes."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)

# Get embedding (mean pooling)
embeddings = outputs.last_hidden_state.mean(dim=1)
print(f"Embedding shape: {embeddings.shape}")
```

### Using the MedVectors API

```python
from examples.01_basic_embeddings import MedVectorsEncoder

# Initialize encoder
encoder = MedVectorsEncoder(model_name="kiranbeethoju/MedVectors-base-v0.1")

# Generate embedding
text = "What are the symptoms of myocardial infarction?"
embedding = encoder.encode(text)

# Generate batch embeddings
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = encoder.encode(texts)
```

---

## Usage Examples

### 1. Basic Embeddings

Generate embeddings for medical text:

```python
from examples.01_basic_embeddings import MedVectorsEncoder

# Initialize
encoder = MedVectorsEncoder(model_name="kiranbeethoju/MedVectors-base-v0.1")

# Single text
query = "What are the symptoms of diabetes?"
query_embedding = encoder.encode(query)

# Batch processing
documents = [
    "Type 2 diabetes causes increased thirst and urination.",
    "Hypertension is defined as blood pressure above 130/80 mmHg.",
    "Asthma symptoms include wheezing and shortness of breath."
]
doc_embeddings = encoder.encode(documents)

# Compute similarity
import numpy as np
similarity = np.dot(query_embedding, doc_embeddings[0])
print(f"Similarity: {similarity:.4f}")
```

**Run the example:**
```bash
python examples/01_basic_embeddings.py
```

---

### 2. Semantic Search

Build a semantic search system:

```python
from examples.02_semantic_search import MedVectorsRetriever

# Initialize retriever
retriever = MedVectorsRetriever(model_name="kiranbeethoju/MedVectors-base-v0.1")

# Add documents
documents = [
    {'id': 1, 'text': 'Type 2 diabetes is a chronic condition...'},
    {'id': 2, 'text': 'Hypertension affects nearly half of adults...'},
    {'id': 3, 'text': 'Asthma causes reversible airway obstruction...'}
]
retriever.add_to_corpus(documents)

# Search
query = "What causes high blood sugar?"
results = retriever.search(query, top_k=3)

for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Text: {result['document']['text']}")
```

**Run the example:**
```bash
python examples/02_semantic_search.py
```

---

### 3. Document Clustering

Cluster medical documents:

```python
from examples.03_clustering import MedVectorsClusterer

# Initialize clusterer
clusterer = MedVectorsClusterer(model_name="kiranbeethoju/MedVectors-small-v0.1")

# Cluster texts
texts = [
    "Myocardial infarction presents with chest pain.",
    "Hypertension is high blood pressure.",
    "Diabetes involves insulin resistance.",
    "Heart failure causes shortness of breath.",
    "Stroke symptoms include weakness and confusion."
]

results = clusterer.kmeans_cluster(texts, n_clusters=2)
print(f"Silhouette Score: {results['silhouette_score']:.4f}")
clusterer.print_cluster_summary(texts, results['cluster_labels'])
```

**Run the example:**
```bash
python examples/03_clustering.py
```

---

### 4. Classification

Classify medical texts:

```python
from examples.04_classification import MedVectorsClassifier

# Initialize classifier
classifier = MedVectorsClassifier(model_name="kiranbeethoju/MedVectors-base-v0.1")

# Training data
texts = [
    "Patient has chest pain and elevated enzymes.",
    "Patient reports fever and cough.",
    "Patient has swollen joints."
]
labels = ["Cardiovascular", "Respiratory", "Orthopedic"]

# Train
results = classifier.train(texts, labels)
print(f"Accuracy: {results['test_accuracy']:.4f}")

# Predict
predictions = classifier.predict(["Patient reports shortness of breath"])
print(f"Predicted: {predictions[0]['predicted_label']}")
```

**Run the example:**
```bash
python examples/04_classification.py
```

---

### 5. RAG System

Build a Retrieval-Augmented Generation system:

```python
from examples.05_rag_system import MedVectorsRAG

# Initialize RAG system
rag = MedVectorsRAG(model_name="kiranbeethoju/MedVectors-base-v0.1")

# Add knowledge base
documents = [
    {'id': 1, 'text': 'Type 2 diabetes treatment includes lifestyle changes and metformin.'},
    {'id': 2, 'text': 'Heart attack symptoms include chest pain radiating to left arm.'},
    {'id': 3, 'text': 'Asthma is treated with inhaled corticosteroids.'}
]
rag.add_documents(documents)

# Retrieve and answer
query = "How is diabetes treated?"
results = rag.retrieve(query, top_k=2)
for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Text: {result['text']}")
```

**Run the example:**
```bash
python examples/05_rag_system.py
```

---

## API Reference

### MedVectorsEncoder

Generate embeddings for text.

```python
MedVectorsEncoder(
    model_name: str = "kiranbeethoju/MedVectors-base-v0.1",
    device: str = None,
    max_length: int = 512
)
```

**Methods:**

- `encode(texts, normalize=True, batch_size=32)` - Encode text(s) into embeddings

### MedVectorsRetriever

Semantic search over a document corpus.

```python
MedVectorsRetriever(
    model_name: str = "kiranbeethoju/MedVectors-base-v0.1",
    device: str = None,
    max_length: int = 512
)
```

**Methods:**

- `add_to_corpus(documents)` - Add documents to search corpus
- `search(query, top_k=5, min_score=0.0)` - Search for relevant documents
- `search_batch(queries, top_k=5)` - Search for multiple queries

### MedVectorsClusterer

Cluster documents using embeddings.

```python
MedVectorsClusterer(
    model_name: str = "kiranbeethoju/MedVectors-base-v0.1",
    device: str = None,
    max_length: int = 512
)
```

**Methods:**

- `kmeans_cluster(texts, n_clusters=3)` - K-Means clustering
- `dbscan_cluster(texts, eps=0.5, min_samples=2)` - DBSCAN clustering
- `hierarchical_cluster(texts, n_clusters=3, linkage='ward')` - Hierarchical clustering

### MedVectorsClassifier

Classify texts using embeddings.

```python
MedVectorsClassifier(
    model_name: str = "kiranbeethoju/MedVectors-base-v0.1",
    device: str = None,
    max_length: int = 512
)
```

**Methods:**

- `train(texts, labels, classifier_type='logistic')` - Train classifier
- `predict(texts, return_probabilities=False)` - Predict labels
- `predict_single(text)` - Predict single text

### MedVectorsRAG

RAG system for question answering.

```python
MedVectorsRAG(
    model_name: str = "kiranbeethoju/MedVectors-base-v0.1",
    device: str = None,
    max_length: int = 512
)
```

**Methods:**

- `add_documents(documents)` - Add documents to knowledge base
- `retrieve(query, top_k=5, min_score=0.0)` - Retrieve relevant documents
- `generate_answer(query, retrieved_docs)` - Generate answer (requires LLM)

---

## Advanced Usage

### Custom Model Loading

```python
from transformers import AutoTokenizer, AutoModel

# Load specific checkpoint
tokenizer = AutoTokenizer.from_pretrained("./checkpoints/medvectors_best")
model = AutoModel.from_pretrained("./checkpoints/medvectors_best")
```

### Fine-tuning on Custom Data

```python
# Prepare your training data
training_data = [
    {
        'query': 'medical question',
        'positive': 'correct answer',
        'negatives': ['incorrect1', 'incorrect2', 'incorrect3']
    },
    ...
]

# See scripts/train_medembed.py for full training pipeline
python scripts/train_medembed.py
```

### Multi-GPU Training

```python
import torch
import torch.distributed as dist

# Initialize process group
dist.init_process_group(backend="nccl")

# Use DistributedDataParallel
model = torch.nn.parallel.DistributedDataParallel(model)
```

### Batch Processing Optimization

```python
# For large datasets, use efficient batching
def process_large_dataset(texts, batch_size=64):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = encoder.encode(batch)
        embeddings.append(batch_embeddings)
    return np.concatenate(embeddings)
```

---

## Training

### Prerequisites

- Sufficient RAM/VRAM (see table below)
- Training data in JSONL format
- MedQuad dataset or custom medical corpus

### Hardware Requirements

| Use Case | Minimum | Recommended |
|----------|----------|-------------|
| Inference only | 8GB RAM, CPU | 16GB RAM, CPU |
| Fine-tuning (small) | 16GB RAM, Apple M1/M2 or 4GB GPU | 32GB RAM, Apple M3 or 8GB GPU |
| Training from scratch | 32GB RAM, Apple M3 or 8GB GPU | 64GB RAM, 16GB GPU |

### Quick Training

```bash
# 1. Process data
python data/processor.py

# 2. Train model
python scripts/train_medembed.py

# 3. Evaluate
python scripts/evaluate_medembed.py
```

### Custom Training Parameters

Edit `scripts/train_medembed.py` to adjust:

```python
trainer.train(
    train_data_path=train_data,
    val_data_path=val_data,

    # Hyperparameters
    model_name="bert-base-uncased",
    batch_size=8,
    num_epochs=10,
    learning_rate=2e-5,
    max_seq_length=256,

    # Training optimizations
    gradient_accumulation_steps=4,
    mixed_precision=False,  # Set True for GPU
    early_stopping_patience=3
)
```

---

## Evaluation

### Run Evaluation

```bash
python scripts/evaluate_medembed.py
```

### Metrics Computed

- **MRR** (Mean Reciprocal Rank): Average of 1/rank for all queries
- **Recall@k**: Fraction of queries with relevant doc in top-k
- **NDCG@k** (Normalized Discounted Cumulative Gain): Rank-aware relevance metric

### Custom Evaluation

```python
from scripts.evaluate_medembed import RetrievalEvaluator

evaluator = RetrievalEvaluator(
    queries_path="data/processed/queries.jsonl",
    corpus_path="data/processed/corpus_index.jsonl",
    output_dir="./results"
)

metrics = evaluator.evaluate(ks=[1, 5, 10, 100], model_name="MedVectors")
print(f"MRR: {metrics['MRR']:.4f}")
```

---

## Troubleshooting

### Out of Memory Error

**Solution:** Reduce batch size or use smaller model

```python
encoder = MedVectorsEncoder(
    model_name="kiranbeethoju/MedVectors-small-v0.1",
    max_length=256  # Reduce from 512
)
embeddings = encoder.encode(texts, batch_size=16)  # Reduce from 32
```

### Slow Training

**Solution:** Verify device and enable optimizations

```python
# Verify GPU/MPS is being used
print(torch.cuda.is_available())  # GPU
print(torch.backends.mps.is_available())  # Mac MPS

# Enable mixed precision (GPU only)
trainer.train(mixed_precision=True)
```

### Poor Retrieval Results

**Solutions:**

1. Increase training epochs: `num_epochs=20`
2. Add more hard negatives
3. Check data quality and alignment
4. Use larger model variant

### MPS Not Available on Mac

**Solution:**

```bash
# Check macOS version (need 12.3+)
sw_vers

# Update PyTorch
pip install --upgrade torch

# Verify Metal framework
xcrutil list | grep Metal
```

---

## Citation

If you use MedVectors in your research, please cite our work:

```bibtex
@software{beethoju2024medvectors,
  author = {Beethoju, Kiran},
  title = {MedVectors: Medical-Focused Embedding Models},
  year = {2024},
  url = {https://github.com/kiranbeethoju/MedVectors}
}
```

---

## License

This project is licensed under the Apache License Version 2.0. See [LICENSE](LICENSE) file for details.

---

## Resources

- **Hugging Face Models:** [kiranbeethoju/MedVectors](https://huggingface.co/kiranbeethoju/MedVectors)
- **GitHub Issues:** [Report Issues](https://github.com/kiranbeethoju/MedVectors/issues)

---

## Support

Developing MedVectors requires significant resources. If you find it valuable, consider [supporting the project](https://www.buymeacoffee.com/abhinand.b).

---

## Contact

For any queries regarding the codebase or research, please reach out to Kiran Beethoju via [GitHub Issues](https://github.com/kiranbeethoju/MedVectors/issues).
