# Manual Model Save & Upload Script
"""
Simple script to save trained models and upload to HuggingFace.
This bypasses the training script issues and gives you direct control.
"""

import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
from huggingface_hub import HfApi
import os

def save_and_upload(model_name: str, hf_repo: str, hf_token: str = None):
    """Save a model and upload to HuggingFace."""

    print(f"\n{'='*60}")
    print(f"Working with: {model_name}")
    print(f"{'='*60}")

    # Get token
    if not hf_token:
        hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_TOKEN')
        if not hf_token:
            print("❌ Error: HF_TOKEN or HUGGING_FACE_TOKEN environment variable not set")
            print("   Set it with: export HF_TOKEN=hf_...")
            return None

    print(f"🔑 Logged in to HuggingFace")

    # Load model
    try:
        model = SentenceTransformer(model_name)
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

    # Save model
    save_path = Path(f"checkpoints/{hf_repo.split('/')[-1]}")
    save_path.mkdir(parents=True, exist_ok=True)

    try:
        model.save(str(save_path))
        print(f"✅ Model saved to {save_path}")
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return None

    # Create repository if it doesn't exist, then upload
    try:
        api = HfApi(token=hf_token)

        # Check if repository exists
        try:
            api.model_info(hf_repo)
            print(f"✅ Repository {hf_repo} already exists")
            repo_exists = True
        except:
            print(f"📝 Creating repository {hf_repo}...")
            api.create_repo(repo_id=hf_repo, repo_type="model")
            repo_exists = False
            print(f"✅ Repository created successfully")

        print(f"📤 Uploading to {hf_repo}...")

        # Upload to correct endpoint
        api.upload_folder(
            folder_path=str(save_path),
            repo_id=hf_repo,
            path_in_repo=".",  # Upload to root of repository
            repo_type="model"
        )

        print(f"✅ Uploaded successfully!")
        print(f"🌐 View at: https://huggingface.co/{hf_repo}")

    except Exception as e:
        print(f"❌ Error uploading: {e}")
        print(f"   You can upload manually:")
        print(f"   huggingface-cli upload {save_path} {hf_repo}")

    return save_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("""
Usage: python3 save_and_upload_model.py <model_name> <hf_repo>
""")
        print("\nAvailable models (MedVectors base models):")
        print("  - BAAI/bge-small-en-v1.5")
        print("  - BAAI/bge-base-en-v1.5")
        print("  - BAAI/bge-large-en-v1.5")
        print("\nExample:")
        print("  python3 save_and_upload_model.py BAAI/bge-base-en-v1.5 kiranbeethoju/MedVectors-base-v0.1")
    else:
        model_name = sys.argv[1]
        hf_repo = sys.argv[2]

        save_and_upload(model_name, hf_repo)
