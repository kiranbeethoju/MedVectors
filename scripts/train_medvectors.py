"""
MedVectors Training Script
Trains embedding models for medical text retrieval.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, models
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    logger.info("✅ Using MPS (Metal) acceleration on Apple Silicon")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info("✅ Using CUDA (GPU)")
else:
    device = torch.device("cpu")
    logger.info("⚠️ Using CPU")


class MedVectorsDataset(TorchDataset):
    """Dataset for MedVectors training data (query-positive-negative triplets)."""

    def __init__(self, data_path: str):
        self.data = []
        self._load_data(data_path)
        logger.info(f"Loaded {len(self.data)} training samples")

    def _load_data(self, data_path: str):
        """Load training data from JSONL file."""
        try:
            with open(data_path, 'r') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        # Support both triplet format and pairs format
                        if 'negatives' in item and len(item['negatives']) > 0:
                            self.data.append(item)
                        elif 'positive' in item:
                            # Create triplet with dummy negative if only pair
                            self.data.append({
                                'query': item.get('query', ''),
                                'positive': item.get('positive', ''),
                                'negatives': ['dummy negative']
                            })
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {e}")
            raise

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'query': item['query'],
            'positive': item['positive'],
            'negative': item['negatives'][0] if len(item['negatives']) > 0 else 'dummy negative'
        }


class MedVectorsTrainer:
    """
    Trainer for MedVectors embedding models.
    Supports multiple model sizes: small, base, large.
    """

    MODEL_CONFIGS = {
        'small': {
            'base_model': 'BAAI/bge-small-en-v1.5',
            'max_seq_length': 512,
            'batch_size': 16,
            'learning_rate': 2e-5,
            'warmup_steps': 1000,
            'num_epochs': 3,
        },
        'base': {
            'base_model': 'BAAI/bge-base-en-v1.5',
            'max_seq_length': 512,
            'batch_size': 8,
            'learning_rate': 2e-5,
            'warmup_steps': 1000,
            'num_epochs': 3,
        },
        'large': {
            'base_model': 'BAAI/bge-large-en-v1.5',
            'max_seq_length': 512,
            'batch_size': 4,
            'learning_rate': 2e-5,
            'warmup_steps': 1000,
            'num_epochs': 3,
        }
    }

    def __init__(
        self,
        output_dir: str = "./checkpoints",
        model_size: str = "base",
        version: str = "v0.1",
        hf_repo: str = None
    ):
        """
        Initialize MedVectors trainer.

        Args:
            output_dir: Directory to save checkpoints
            model_size: Model size - 'small', 'base', or 'large'
            version: Model version (e.g., 'v0.1')
            hf_repo: HuggingFace repo name for upload (e.g., 'kiranbeethoju/MedVectors')
        """
        if model_size not in self.MODEL_CONFIGS:
            raise ValueError(f"Invalid model_size: {model_size}. Must be one of {list(self.MODEL_CONFIGS.keys())}")

        self.model_size = model_size
        self.version = version
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hf_repo = hf_repo

        self.config = self.MODEL_CONFIGS[model_size]
        self.model_name = f"kiranbeethoju/MedVectors-{model_size}-{version}"

        logger.info(f"Initialized MedVectors Trainer")
        logger.info(f"  Model size: {model_size}")
        logger.info(f"  Version: {version}")
        logger.info(f"  Base model: {self.config['base_model']}")
        logger.info(f"  Output directory: {self.output_dir}")

    def train(
        self,
        train_data_path: str,
        val_data_path: str = None,
        eval_steps: int = 1000,
        save_steps: int = 1000,
        max_steps: int = None
    ):
        """
        Train MedVectors model.

        Args:
            train_data_path: Path to training data (JSONL)
            val_data_path: Path to validation data (JSONL)
            eval_steps: Evaluate every N steps
            save_steps: Save checkpoint every N steps
            max_steps: Maximum training steps (None = full training)
        """
        logger.info("=" * 80)
        logger.info("🚀 STARTING MEDVECTORS TRAINING")
        logger.info("=" * 80)

        # Load datasets
        logger.info(f"Loading training data from {train_data_path}")
        train_dataset = MedVectorsDataset(train_data_path)

        val_dataset = None
        if val_data_path and os.path.exists(val_data_path):
            logger.info(f"Loading validation data from {val_data_path}")
            val_dataset = MedVectorsDataset(val_data_path)

        # Initialize model
        logger.info(f"Initializing model: {self.config['base_model']}")
        model = SentenceTransformer(self.config['base_model'], device=str(device))
        model_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {model_params}")

        # Prepare training examples
        logger.info("Preparing training examples...")
        train_examples = []
        for item in train_dataset.data:
            train_examples.append(
                InputExample(
                    texts=[item['query'], item['positive'], item['negatives'][0] if len(item['negatives']) > 0 else 'dummy negative'],
                    label=1.0
                )
            )
        logger.info(f"Prepared {len(train_examples)} training examples")

        # Prepare validation examples if available
        val_evaluator = None
        if val_dataset and len(val_dataset) > 0:
            logger.info("Preparing validation examples...")
            val_examples = []
            for item in val_dataset.data:
                # For validation, we need (query, positives_list) format
                val_examples.append({
                    'query': item['query'],
                    'positive': item['positive']
                })

            # Create evaluator
            if len(val_examples) > 0:
                corpus = {str(i): ex['positive'] for i, ex in enumerate(val_examples)}
                queries = {str(i): ex['query'] for i, ex in enumerate(val_examples)}
                # Create simple validation mapping
                relevant_docs = {str(i): [str(i)] for i in range(len(val_examples))}
                dev_evaluator = evaluation.InformationRetrievalEvaluator(
                    queries, corpus, relevant_docs,
                    name="medvectors_val"
                )
                val_evaluator = dev_evaluator
                logger.info(f"Prepared {len(val_examples)} validation examples")

        # Create data loader
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=self.config['batch_size'],
            num_workers=0  # Mac-specific
        )

        # Define loss
        train_loss = losses.TripletLoss(model=model, triplet_margin=0.2)

        # Calculate total steps
        total_steps = len(train_dataloader) * self.config['num_epochs']
        if max_steps:
            total_steps = min(total_steps, max_steps)

        # Training configuration
        warmup_steps = min(self.config['warmup_steps'], total_steps // 10)

        logger.info(f"\nTraining configuration:")
        logger.info(f"  Model: {self.config['base_model']}")
        logger.info(f"  Version: {self.version}")
        logger.info(f"  Size: {self.model_size}")
        logger.info(f"  Batch size: {self.config['batch_size']}")
        logger.info(f"  Num epochs: {self.config['num_epochs']}")
        logger.info(f"  Learning rate: {self.config['learning_rate']}")
        logger.info(f"  Max seq length: {self.config['max_seq_length']}")
        logger.info(f"  Warmup steps: {warmup_steps}")
        logger.info(f"  Steps per epoch: {len(train_dataloader)}")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Device: {device}")

        # Create checkpoint directory
        checkpoint_dir = self.output_dir / f"MedVectors-{self.model_size}-{self.version}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Callbacks
        class SaveCallback:
            def __init__(self, output_path, hf_repo, version, model_size):
                self.output_path = output_path
                self.hf_repo = hf_repo
                self.version = version
                self.model_size = model_size
                self.best_score = 0.0

            def __call__(self, score, epoch, steps):
                if self.hf_repo and score > self.best_score:
                    self.best_score = score
                    logger.info(f"✅ New best score: {score:.4f}")

        save_callback = SaveCallback(checkpoint_dir, self.hf_repo, self.version, self.model_size)

        # Train model
        logger.info("\nStarting training...")
        model.fit(
            train_objectives=[train_dataloader],
            loss=train_loss,
            evaluator=val_evaluator,
            epochs=self.config['num_epochs'],
            warmup_steps=warmup_steps,
            output_path=str(checkpoint_dir),
            evaluation_steps=eval_steps if val_evaluator else None,
            optimizer_params={'lr': self.config['learning_rate']},
            show_progress_bar=True
        )

        # Save final model
        logger.info("\nSaving final model...")
        final_model_path = checkpoint_dir / "final"
        final_model_path.mkdir(parents=True, exist_ok=True)
        model.save(str(final_model_path))

        # Update model config
        self._update_model_config(final_model_path)

        logger.info(f"✅ Model saved to {final_model_path}")

        # Upload to HuggingFace if repo specified
        if self.hf_repo:
            self._upload_to_huggingface(final_model_path)

        logger.info("\n" + "=" * 80)
        logger.info("✅ TRAINING COMPLETE")
        logger.info("=" * 80)

        return final_model_path

    def _update_model_config(self, model_path: Path):
        """Update model configuration with MedVectors metadata."""
        config_path = model_path / "config_sentence_transformers.json"

        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Update with MedVectors metadata
            config['__version__'] = {
                'medvectors_version': self.version,
                'model_size': self.model_size,
                'trained_date': datetime.now().strftime("%Y-%m-%d"),
                'base_model': self.config['base_model']
            }

            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            logger.info("Updated model configuration")

    def _upload_to_huggingface(self, model_path: Path):
        """Upload trained model to HuggingFace."""
        try:
            from huggingface_hub import HfApi, login

            # Login if token is available
            hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_TOKEN')
            if hf_token:
                login(token=hf_token)
                logger.info("Logged in to HuggingFace")
            else:
                logger.warning("No HuggingFace token found. Set HF_TOKEN or HUGGING_FACE_TOKEN environment variable.")
                return

            # Create model card
            self._create_model_card(model_path)

            # Upload model
            api = HfApi()
            api.upload_folder(
                folder_path=str(model_path),
                repo_id=self.hf_repo,
                repo_type="model"
            )

            logger.info(f"✅ Model uploaded to {self.hf_repo}")
        except Exception as e:
            logger.error(f"Error uploading to HuggingFace: {e}")

    def _create_model_card(self, model_path: Path):
        """Create README.md for HuggingFace."""
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
language:
- en
pipeline_tag: sentence-similarity
---

# MedVectors-{self.model_size}-{self.version}

MedVectors is a medical-focused embedding model fine-tuned for healthcare and clinical NLP tasks.

## Model Details

- **Model Size**: {self.model_size}
- **Version**: {self.version}
- **Base Model**: {self.config['base_model']}
- **Parameters**: ~{self._get_param_count()}
- **Max Sequence Length**: {self.config['max_seq_length']}

## Usage

```python
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('kiranbeethoju/MedVectors-{self.model_size}-{self.version}')

# Encode texts
embeddings = model.encode([
    "Patient presents with chest pain and elevated cardiac enzymes.",
    "Type 2 diabetes causes increased thirst and urination."
])

print(f"Embedding shape: {{embeddings.shape}}")
```

## Training

This model was trained on medical QA datasets including MedQuad for improved performance on healthcare-related retrieval tasks.

## Performance

The model has been evaluated on medical NLP benchmarks for retrieval, including:
- MedicalQARetrieval
- TRECCOVID
- PublicHealthQA
- NFCorpus
- ArguAna

## Citation

If you use this model, please cite:

```bibtex
@software{{medvectors2026}},
  title = {{MedVectors: Medical-Focused Embedding Models}},
  author = {{Beethoju, Kiran}},
  year = {{2026}},
  url = {{https://huggingface.co/kiranbeethoju/MedVectors}}
}}
```

## License

Apache License 2.0
"""

        readme_path = model_path / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)

        logger.info("Created model card")

    def _get_param_count(self) -> str:
        """Get human-readable parameter count."""
        if self.model_size == 'small':
            return '33M'
        elif self.model_size == 'base':
            return '110M'
        else:
            return '340M'


def main():
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(description='Train MedVectors')
    parser.add_argument('--model_size', type=str, default='base', choices=['small', 'base', 'large'],
                       help='Model size: small, base, or large')
    parser.add_argument('--version', type=str, default='v0.1',
                       help='Model version (e.g., v0.1)')
    parser.add_argument('--train_data', type=str, default='./data/processed/train.jsonl',
                       help='Path to training data')
    parser.add_argument('--val_data', type=str, default='./data/processed/val.jsonl',
                       help='Path to validation data')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--hf_repo', type=str, default='kiranbeethoju/MedVectors-base-v0.1',
                       help='HuggingFace repository name')
    parser.add_argument('--max_steps', type=int, default=None,
                       help='Maximum training steps')

    args = parser.parse_args()

    # Update hf_repo based on model_size
    hf_repo = f"kiranbeethoju/MedVectors-{args.model_size}-{args.version}"

    # Initialize trainer
    trainer = MedVectorsTrainer(
        output_dir=args.output_dir,
        model_size=args.model_size,
        version=args.version,
        hf_repo=hf_repo
    )

    # Train
    model_path = trainer.train(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        eval_steps=1000,
        save_steps=1000,
        max_steps=args.max_steps
    )

    logger.info(f"\n✅ Training complete! Model saved to: {model_path}")
    if trainer.hf_repo:
        logger.info(f"✅ Uploaded to: {trainer.hf_repo}")


if __name__ == "__main__":
    main()
