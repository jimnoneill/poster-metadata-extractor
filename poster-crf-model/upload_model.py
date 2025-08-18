#!/usr/bin/env python3
"""
Upload the trained CRF model to HuggingFace Hub
"""

from huggingface_hub import HfApi, create_repo, upload_file
from pathlib import Path
import json
import argparse
import os

def upload_model_to_hub(repo_name="poster-metadata-crf", organization="jimnoneill"):
    """Upload model files to HuggingFace Hub"""
    
    # Initialize the API
    api = HfApi()
    
    # Full repository name
    repo_id = f"{organization}/{repo_name}"
    
    print(f"🚀 Uploading model to HuggingFace Hub: {repo_id}")
    
    # Check if user is logged in
    try:
        user_info = api.whoami()
        print(f"✅ Logged in as: {user_info['name']}")
    except Exception as e:
        print("❌ Not logged in to HuggingFace!")
        print("   Please run: huggingface-cli login")
        return False
    
    # Create repository if it doesn't exist
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"✅ Repository ready: {repo_id}")
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        return False
    
    # Files to upload
    files_to_upload = [
        "pytorch_model.bin",
        "config.json",
        "README.md",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt"
    ]
    
    # Upload each file
    model_dir = Path(".")
    uploaded_files = []
    
    for filename in files_to_upload:
        file_path = model_dir / filename
        if file_path.exists():
            try:
                print(f"📤 Uploading {filename}...")
                upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=filename,
                    repo_id=repo_id,
                    repo_type="model"
                )
                uploaded_files.append(filename)
                print(f"   ✅ {filename} uploaded successfully")
            except Exception as e:
                print(f"   ❌ Error uploading {filename}: {e}")
        else:
            print(f"   ⚠️  {filename} not found, skipping...")
    
    if uploaded_files:
        print(f"\n✅ Successfully uploaded {len(uploaded_files)} files to {repo_id}")
        print(f"🌐 View your model at: https://huggingface.co/{repo_id}")
        return True
    else:
        print("\n❌ No files were uploaded")
        return False

def main():
    parser = argparse.ArgumentParser(description="Upload CRF model to HuggingFace")
    parser.add_argument("--repo", default="poster-metadata-crf", help="Repository name")
    parser.add_argument("--org", default="jimnoneill", help="Organization/username")
    parser.add_argument("--dry-run", action="store_true", help="Just list files, don't upload")
    args = parser.parse_args()
    
    if args.dry_run:
        print("🔍 DRY RUN - Files that would be uploaded:")
        model_dir = Path(".")
        for pattern in ["*.bin", "*.json", "*.txt", "*.md"]:
            for file in model_dir.glob(pattern):
                print(f"   • {file.name} ({file.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        # Check for HuggingFace token
        if not os.getenv("HF_TOKEN") and not Path.home().joinpath(".huggingface/token").exists():
            print("⚠️  No HuggingFace token found!")
            print("   Please login first: huggingface-cli login")
            print("   Or set HF_TOKEN environment variable")
            return
        
        success = upload_model_to_hub(args.repo, args.org)
        if not success:
            print("\n💡 Tips:")
            print("   1. Make sure you're logged in: huggingface-cli login")
            print("   2. Check that all model files exist")
            print("   3. Ensure you have write permissions to the organization/namespace")

if __name__ == "__main__":
    main()


