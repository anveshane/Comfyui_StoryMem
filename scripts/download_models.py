#!/usr/bin/env python3
"""
Download script for StoryMem models.

Downloads all required models from HuggingFace:
- Wan2.2-T2V-A14B (~20-30 GB)
- Wan2.2-I2V-A14B (~20-30 GB)
- StoryMem M2V LoRA weights (~1-2 GB)

Total: ~40-60 GB
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def get_comfyui_models_dir():
    """Try to locate ComfyUI models directory."""
    # Try to find ComfyUI directory
    current = Path(__file__).resolve().parent.parent

    # Look for ComfyUI structure
    possible_paths = [
        current.parent.parent / "models" / "storymem",  # custom_nodes/../models/storymem
        current / "models",  # local models directory
        Path.home() / "ComfyUI" / "models" / "storymem",  # ~/ComfyUI/models/storymem
    ]

    for path in possible_paths:
        if path.parent.exists() or path.exists():
            return path

    # Default fallback
    return current / "models"


def check_disk_space(path: Path, required_gb: float = 60):
    """Check if sufficient disk space is available."""
    import shutil

    try:
        stat = shutil.disk_usage(path.parent if path.exists() else path.parent.parent)
        free_gb = stat.free / (1024 ** 3)

        print(f"Available disk space: {free_gb:.1f} GB")
        print(f"Required disk space: {required_gb} GB")

        if free_gb < required_gb:
            print(f"WARNING: Low disk space! You need at least {required_gb}GB free.")
            response = input("Continue anyway? (y/n): ")
            return response.lower() == 'y'

        return True
    except Exception as e:
        print(f"Could not check disk space: {e}")
        return True


def check_huggingface_cli():
    """Check if huggingface-cli is installed."""
    try:
        result = subprocess.run(
            ["huggingface-cli", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✓ HuggingFace CLI found: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass

    print("ERROR: huggingface-cli not found!")
    print("\nPlease install it with:")
    print("  pip install 'huggingface_hub[cli]'")
    return False


def download_model(repo_id: str, local_dir: Path, model_name: str):
    """
    Download a model from HuggingFace.

    Args:
        repo_id: HuggingFace repository ID
        local_dir: Local directory to save model
        model_name: Display name for progress
    """
    print(f"\n{'='*60}")
    print(f"Downloading {model_name}")
    print(f"Repository: {repo_id}")
    print(f"Destination: {local_dir}")
    print(f"{'='*60}\n")

    # Create directory if needed
    local_dir.mkdir(parents=True, exist_ok=True)

    # Download using huggingface-cli
    cmd = [
        "huggingface-cli",
        "download",
        repo_id,
        "--local-dir", str(local_dir),
        "--local-dir-use-symlinks", "False",
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"\n✓ {model_name} downloaded successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed to download {model_name}")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n\nDownload interrupted by user")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download StoryMem models from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all models to default location
  python download_models.py

  # Download to specific directory
  python download_models.py --output /path/to/models

  # Download only specific models
  python download_models.py --models t2v i2v

  # Skip disk space check
  python download_models.py --no-check
        """
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for models (default: auto-detect ComfyUI models dir)"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["t2v", "i2v", "m2v", "all"],
        default=["all"],
        help="Which models to download (default: all)"
    )

    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip disk space check"
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output:
        models_dir = args.output
    else:
        models_dir = get_comfyui_models_dir()

    print(f"\n{'='*60}")
    print("StoryMem Model Downloader")
    print(f"{'='*60}\n")
    print(f"Target directory: {models_dir}")

    # Check prerequisites
    if not check_huggingface_cli():
        sys.exit(1)

    # Check disk space
    if not args.no_check:
        if not check_disk_space(models_dir, required_gb=60):
            sys.exit(1)

    # Determine which models to download
    download_all = "all" in args.models
    download_t2v = download_all or "t2v" in args.models
    download_i2v = download_all or "i2v" in args.models
    download_m2v = download_all or "m2v" in args.models

    # Model definitions
    models = []

    if download_t2v:
        models.append({
            "name": "Wan2.2-T2V-A14B (Text-to-Video)",
            "repo_id": "Wan-AI/Wan2.2-T2V-A14B",
            "local_dir": models_dir / "Wan2.2-T2V-A14B",
        })

    if download_i2v:
        models.append({
            "name": "Wan2.2-I2V-A14B (Image-to-Video)",
            "repo_id": "Wan-AI/Wan2.2-I2V-A14B",
            "local_dir": models_dir / "Wan2.2-I2V-A14B",
        })

    if download_m2v:
        models.append({
            "name": "StoryMem M2V LoRA Weights",
            "repo_id": "Kevin-thu/StoryMem",
            "local_dir": models_dir / "StoryMem",
        })

    # Show summary
    print(f"\nModels to download: {len(models)}")
    for model in models:
        print(f"  - {model['name']}")

    print("\nThis will download approximately 40-60 GB of data.")
    print("Downloads are resumable if interrupted.\n")

    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Download cancelled.")
        sys.exit(0)

    # Download models
    success_count = 0
    failed = []

    for model in models:
        if download_model(model["repo_id"], model["local_dir"], model["name"]):
            success_count += 1
        else:
            failed.append(model["name"])

    # Final summary
    print(f"\n{'='*60}")
    print("Download Summary")
    print(f"{'='*60}")
    print(f"Successful: {success_count}/{len(models)}")

    if failed:
        print(f"\nFailed downloads:")
        for name in failed:
            print(f"  ✗ {name}")
        print("\nYou can re-run this script to retry failed downloads.")
        sys.exit(1)
    else:
        print(f"\n✓ All models downloaded successfully!")
        print(f"\nModels location: {models_dir}")
        print("\nYou can now use StoryMem nodes in ComfyUI!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
