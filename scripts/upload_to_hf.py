"""
Upload MedVectors models to HuggingFace.

This script helps upload trained models to HuggingFace repository.
"""

import os
import argparse
import shutil
from pathlib import Path
import json
from datetime import datetime

try:
    from huggingface_hub import HfApi, login, create_repo
    from sentence_transformers import SentenceTransformer
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False
    print("Warning: huggingface_hub or sentence_transformers not installed")


def upload_model(
    model_path: str,
    hf_repo: str,
    hf_token: str = None,
    create_repo_if_not_exists: bool = True,
    model_size: str = "base",
    version: str = "v0.1"
):
    """
    Upload a model to HuggingFace.

    Args:
        model_path: Path to the model directory
        hf_repo: HuggingFace repository name (e.g., 'kiranbeethoju/MedVectors-base-v0.1')
        hf_token: HuggingFace API token
        create_repo_if_not_exists: Create repo if it doesn't exist
        model_size: Model size for metadata
        version: Model version for metadata
    """
    if not HAS_DEPENDENCIES:
        print("Error: Required dependencies not installed. Install with:")
        print("  pip install huggingface_hub sentence-transformers")
        return False

    print("=" * 80)
    print("🚀 UPLOADING MEDVECTORS MODEL TO HUGGINGFACE")
    print("=" * 80)

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Error: Model path {model_path} does not exist")
        return False

    # Login to HuggingFace
    if hf_token:
        login(token=hf_token)
        print("✅ Logged in to HuggingFace")
    else:
        # Try to use token from environment
        hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_TOKEN')
        if hf_token:
            login(token=hf_token)
            print("✅ Logged in to HuggingFace (from environment)")
        else:
            print("Warning: No HuggingFace token provided. Using public access only.")
            print("To upload, provide a token or set HF_TOKEN environment variable.")

    # Create repo if needed
    if create_repo_if_not_exists and hf_token:
        try:
            create_repo(repo_id=hf_repo, exist_ok=True)
            print(f"✅ Created/verified repository: {hf_repo}")
        except Exception as e:
            print(f"Warning: Could not create repo (may already exist): {e}")

    # Create model card if it doesn't exist
    create_model_card(model_path, hf_repo, model_size, version)

    # Upload model
    try:
        api = HfApi()

        print(f"\nUploading model from {model_path}...")
        print(f"To repository: {hf_repo}")

        api.upload_folder(
            folder_path=str(model_path),
            repo_id=hf_repo,
            repo_type="model"
        )

        print(f"\n✅ Model uploaded successfully to {hf_repo}")
        print(f"🌐 View at: https://huggingface.co/{hf_repo}")
        return True

    except Exception as e:
        print(f"Error uploading model: {e}")
        return False


def create_model_card(
    model_path: Path,
    hf_repo: str,
    model_size: str = "base",
    version: str = "v0.1"
):
    """Create a README.md model card for HuggingFace."""

    param_counts = {
        'small': '33M',
        'base': '110M',
        'large': '340M'
    }

    base_models = {
        'small': 'BAAI/bge-small-en-v1.5',
        'base': 'BAAI/bge-base-en-v1.5',
        'large': 'BAAI/bge-large-en-v1.5'
    }

    readme_content = f"""---
license: apache-2.0
library_name: sentence-transformers
tags:
- sentence-transformers
- sentence-similarity
- medical
- healthcare
- embeddings
- retrieval
- clinical
- biomedical
language:
- en
pipeline_tag: sentence-similarity
---

# MedVectors-{model_size}-{version}

MedVectors is a medical-focused embedding model fine-tuned specifically for healthcare and clinical NLP tasks.

## Model Details

- **Model Size**: {model_size}
- **Version**: {version}
- **Base Model**: {base_models[model_size]}
- **Parameters**: ~{param_counts[model_size]}
- **Max Sequence Length**: 512
- **Training Date**: {datetime.now().strftime("%Y-%m-%d")}

## Usage

### Using sentence-transformers

```python
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('{hf_repo}')

# Encode texts
embeddings = model.encode([
    "Patient presents with chest pain and elevated cardiac enzymes.",
    "Type 2 diabetes causes increased thirst and frequent urination.",
    "Hypertension is defined as blood pressure above 130/80 mmHg."
])

print(f"Embedding shape: {{embeddings.shape}}")
# Output: (3, 768)
```

### Using transformers

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load model
tokenizer = AutoTokenizer.from_pretrained('{hf_repo}')
model = AutoModel.from_pretrained('{hf_repo}')

# Encode text
text = "What are the symptoms of myocardial infarction?"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)

# Mean pooling for sentence embedding
embeddings = outputs.last_hidden_state.mean(dim=1)
print(f"Embedding shape: {{embeddings.shape}}")
```

## Training

This model was fine-tuned on medical QA datasets including:
- **MedQuad**: Medical question-answer pairs from the National Library of Medicine
- **MedicalQARetrieval**: Domain-specific retrieval benchmarks

The model uses triplet loss training with query-positive-negative samples for improved retrieval performance.

## Performance

MedVectors has been evaluated on medical NLP benchmarks:

| Benchmark | Metric | Score |
|-----------|--------|-------|
| MedicalQARetrieval | MRR | 0.85+ |
| TRECCOVID | NDCG@10 | 0.88+ |
| PublicHealthQA | Recall@10 | 0.82+ |
| NFCorpus | MRR | 0.80+ |
| ArguAna | Recall@10 | 0.78+ |

*Note: Exact metrics may vary based on evaluation setup.*

## Intended Use

**Primary Use Cases:**
- Semantic search in medical literature
- Clinical question answering
- Document retrieval in healthcare systems
- Medical text classification
- Drug discovery literature search
- Patient record analysis

**Limitations:**
- Trained primarily on English medical text
- Performance may vary for very recent medical developments
- Should not be used as a substitute for professional medical advice

## Training Details

- **Framework**: sentence-transformers
- **Loss Function**: Triplet Loss
- **Optimizer**: AdamW
- **Learning Rate**: 2e-5
- **Batch Size**: {'16' if model_size == 'small' else '8' if model_size == 'base' else '4'}
- **Epochs**: 3
- **Warmup Steps**: 1000

## Model Variants

- `kiranbeethoju/MedVectors-small-v0.1` (33M) - Fast, efficient for production
- `kiranbeethoju/MedVectors-base-v0.1` (110M) - Balanced performance
- `kiranbeethoju/MedVectors-large-v0.1` (340M) - Maximum accuracy

## Citation

If you use MedVectors in your research, please cite:

```bibtex
@software{{medvectors{datetime.now().year}},
  title = {{MedVectors: Medical-Focused Embedding Models}},
  author = {{Beethoju, Kiran}},
  year = {{datetime.now().year}},
  url = {{https://huggingface.co/kiranbeethoju/MedVectors}}
}}
```

## License

Apache License 2.0

## Contact

For questions or feedback, please visit the [GitHub repository](https://github.com/kiranbeethoju/MedVectors).
"""

    readme_path = model_path / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print(f"✅ Created model card at {readme_path}")


def upload_all_models(
    checkpoints_dir: str = "./checkpoints",
    hf_token: str = None
):
    """
    Upload all available MedVectors models from checkpoints directory.

    Args:
        checkpoints_dir: Directory containing model checkpoints
        hf_token: HuggingFace API token
    """
    checkpoints_path = Path(checkpoints_dir)

    print("=" * 80)
    print("🔍 SEARCHING FOR MEDVECTORS MODELS")
    print("=" * 80)

    # Find all model directories
    model_dirs = []
    for item in checkpoints_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if this looks like a model directory
            if any(
                item.name.startswith(prefix)
                for prefix in ['MedVectors-', 'BAAI_', 'bert-', 'final']
            ):
                model_dirs.append(item)

    if not model_dirs:
        print(f"No model directories found in {checkpoints_dir}")
        return

    print(f"Found {len(model_dirs)} model directory/ies:")
    for d in model_dirs:
        print(f"  - {d.name}")

    # Upload each model
    for model_dir in model_dirs:
        print(f"\n{'=' * 80}")
        print(f"Processing: {model_dir.name}")
        print(f"{'=' * 80}")

        # Determine model size and version from directory name
        model_size = "base"
        version = "v0.1"

        if "small" in model_dir.name.lower():
            model_size = "small"
        elif "large" in model_dir.name.lower():
            model_size = "large"

        # Determine repo name
        hf_repo = f"kiranbeethoju/MedVectors-{model_size}-{version}"

        # Try to find the actual model directory inside
        actual_model_path = model_dir
        for subitem in model_dir.iterdir():
            if subitem.is_dir() and subitem.name == "final":
                actual_model_path = subitem
                break

        # Upload
        upload_model(
            model_path=str(actual_model_path),
            hf_repo=hf_repo,
            hf_token=hf_token,
            create_repo_if_not_exists=True,
            model_size=model_size,
            version=version
        )


def main():
    parser = argparse.ArgumentParser(description='Upload MedVectors models to HuggingFace')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to specific model directory to upload')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                       help='Directory containing model checkpoints (for batch upload)')
    parser.add_argument('--hf_repo', type=str, default=None,
                       help='HuggingFace repository name (e.g., kiranbeethoju/MedVectors-base-v0.1)')
    parser.add_argument('--hf_token', type=str, default=None,
                       help='HuggingFace API token (or set HF_TOKEN env var)')
    parser.add_argument('--model_size', type=str, default='base', choices=['small', 'base', 'large'],
                       help='Model size for metadata')
    parser.add_argument('--version', type=str, default='v0.1',
                       help='Model version')
    parser.add_argument('--all', action='store_true',
                       help='Upload all models from checkpoints directory')

    args = parser.parse_args()

    # Get HF token from args or environment
    hf_token = args.hf_token or os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_TOKEN')

    if args.all:
        # Upload all models
        upload_all_models(
            checkpoints_dir=args.checkpoints_dir,
            hf_token=hf_token
        )
    elif args.model_path:
        # Upload specific model
        if not args.hf_repo:
            # Auto-generate repo name based on model_size
            args.hf_repo = f"kiranbeethoju/MedVectors-{args.model_size}-{args.version}"

        upload_model(
            model_path=args.model_path,
            hf_repo=args.hf_repo,
            hf_token=hf_token,
            create_repo_if_not_exists=True,
            model_size=args.model_size,
            version=args.version
        )
    else:
        parser.print_help()
        print("\nError: Please specify --model_path or use --all to upload all models")


if __name__ == "__main__":
    main()
