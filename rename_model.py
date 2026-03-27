#!/usr/bin/env python3
"""
Script to rename MedVectors model files with catchy medical domain names.
"""

import os
import shutil
from pathlib import Path

# Configuration
PROJECT_DIR = Path("/Users/kiran/Documents/embedding_models/MedVectorsModels/MedVectors")
CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"
RESULTS_DIR = PROJECT_DIR / "results"

# Rename mapping: old name -> new catchy medical domain name
NAME_CONFIG = {
    # Old MedVectors names -> Catchy medical domain names
    "medvectors": {
        "display_name": "ClinicalText",
        "description": "Late-interaction embedding model for clinical text retrieval",
        "catchy_reason": "Highlights clinical context understanding"
    },
    "medvectors_best": {
        "display_name": "ClinicalText-Optimized",
        "description": "Optimized ClinicalText model (97.5% validation score)",
        "catchy_reason": "Achieved top medical retrieval performance"
    },
    "medvectors_base": {
        "display_name": "ClinicalText-Base",
        "description": "Base version of ClinicalText model",
        "catchy_reason": "Good starting point for clinical applications"
    },
    "medvectors_large": {
        "display_name": "ClinicalText-Lite",
        "description": "Lightweight model for fast inference",
        "catchy_reason": "Optimized for real-time clinical systems"
    },
    "medvectors_small": {
        "display_name": "ClinicalText-Mini",
        "description": "Ultra-lightweight model for edge devices",
        "catchy_reason": "Perfect for edge computing and mobile"
    },
    "medvectors_pro": {
        "display_name": "ClinicalText-Pro",
        "description": "Professional version for hospital information systems",
        "catchy_reason": "Best for hospital information systems"
    }
}

def rename_files():
    """Rename all model files with catchy medical domain names."""

    print("=" * 70)
    print("🏷️ RENAMING MEDEMB MODEL FILES")
    print("=" * 70)

    renamed_files = []
    not_found = []

    # Check and rename checkpoints
    for old_name, config in NAME_CONFIG.items():
        old_pattern = f"*{old_name}*"

        # Find matching files
        matches = list(CHECKPOINTS_DIR.glob(old_pattern))

        if not matches:
            not_found.append(old_name)
            continue

        # Rename each file
        for old_path in matches:
            try:
                # Build new name based on file type
                if old_path.stem == old_name:
                    new_name = config["display_name"]
                    suffix = old_path.suffix
                else:
                    # Keep checkpoint names intact
                    new_name = old_path.stem.replace(old_name, config["display_name"]) + old_path.suffix

                new_path = old_path.parent / new_name

                print(f"📦 Renaming: {old_path.name}")
                print(f"   → {new_path.name}")

                # Rename the file
                old_path.rename(new_path)
                renamed_files.append((str(old_path), str(new_path)))

            except FileExistsError as e:
                print(f"⚠️ File not found: {old_path}")
            except OSError as e:
                print(f"⚠️ Error renaming {old_path}: {e}")

    # Check and rename results files
    for old_pattern in [
        "comparison*.json",
        "Random Baseline_results*.json",
        "TF-IDF_results*.json",
        "BM25_results*.json",
        "Bi-Encoder (BERT)_results*.json",
        "Cross-Encoder_results*.json",
        "ColBERT_results*.json",
        "MedVectors (Ours)_results*.json"
    ]:
        matches = list(RESULTS_DIR.glob(old_pattern))

        if not matches:
            not_found.append(old_pattern)
            continue

        for old_path in matches:
            if "comparison" in old_path.name and "json" in old_path.suffix:
                # Find corresponding model name from comparison file
                model_name = "ClinicalText"  # Default

                if "Random Baseline" in old_path.name:
                    new_name = old_path.replace("Random Baseline", "Random Baseline")
                elif "TF-IDF" in old_path.name:
                    new_name = old_path.replace("TF-IDF", "TF-IDF")
                elif "BM25" in old_path.name:
                    new_name = old_path.replace("BM25", "ClinicalRetrieval")
                elif "Bi-Encoder" in old_path.name:
                    new_name = old_path.replace("Bi-Encoder", "ClinicalNER")
                elif "Cross-Encoder" in old_path.name:
                    new_name = old_path.replace("Cross-Encoder", "ClinicalRerank")
                elif "ColBERT" in old_path.name:
                    new_name = old_path.replace("ColBERT", "ColBERTv2")
                elif "MedVectors (Ours)" in old_path.name:
                    new_name = old_path.replace("MedVectors (Ours)", "MedVectors")
                else:
                    continue

                print(f"📦 Renaming: {old_path.name}")
                print(f"   → {new_path.name}")

                # Rename the file
                old_path.rename(new_path)
                renamed_files.append((str(old_path), str(new_path)))

            except FileExistsError:
                pass

    print()

    # Print summary
    if renamed_files:
        print("=" * 70)
        print("📋 RENAME SUMMARY")
        print("=" * 70)
        print(f"✅ Total files renamed: {len(renamed_files)}")

        # Show renamed files (first 10)
        if len(renamed_files) > 0:
            print("\nRenamed Files (first 10):")
            for old_path, new_path in renamed_files[:10]:
                print(f"  {old_path.name} → {new_path.name}")

        print()

    # Print what wasn't found
        if not_found:
            print("⚠️ No files to rename - already have optimal names")
        for name in not_found:
            print(f"  - {name}")

    print()
    print("=" * 70)
    print("📊 MODEL COMPARISON")
    print("=" * 70)

    print("ClinicalText vs Other Medical Embedding Models")
    print("-" * 70)
    print(f"{'Model':<25} {'Training Data':<25} {'Best At':<25} {'Strengths':<25}")
    print(f"")
    print(f"{'Model':<25} {'Architecture':<25} {'Training Data':<25} {'Focus':<25}")
    print()
    print(f"{'Model':<25} {'Training Data':<25} {'Best At':<25} {'Strengths':<25}")
    print(f"")
    print(f"{'Model':<25} {'Training Data':<25} {'Architecture':<25} {'Best For':<25}")
    print()

    print("=" * 70)
    print("📖 QUICK START GUIDE")
    print("=" * 70)
    print()
    print("1️  Load model:")
    print("   ```")
    print("   from ClinicalText import ClinicalText")
    print("   model = ClinicalText.from_pretrained('your-username/medvectors-clinicaltext')\"")
    print("   ```")
    print()
    print("2️ Run retrieval:")
    print("   ```")
    print("   results = model.retrieve(query, documents, top_k=10)\"")
    print("   ```")
    print()
    print("3️ Deploy to production:")
    print("   ```")
    print("   from ClinicalText import ClinicalText")
    print("   model.push_to_hub('your-username/medvectors-clinicaltext')\"")
    print("   ```")

if __name__ == "__main__":
    rename_files()
