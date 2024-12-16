"""
Script to sync local test data to the Hub.
Usage:
poetry run python tests/test_data/sync_test_data.py
"""

import os
from pathlib import Path
from typing import Set

import yaml
from dotenv import load_dotenv
from huggingface_hub import HfApi


# Load token from .env file
load_dotenv()


def sync_test_files():
    """Upload all local YAML/JSON example prompts to the Hub for each directory."""
    token = os.environ.get("HF_TOKEN")

    if not token:
        raise ValueError(
            "No HF_TOKEN found in environment variables. "
            "Please set it with your Hugging Face write token from: "
            "https://huggingface.co/settings/tokens"
        )

    test_data_dir = Path(__file__).parent

    # Get all directories in the test data folder, excluding __pycache__
    directories = [d for d in test_data_dir.iterdir() if d.is_dir() and d.name != "__pycache__"]

    if not directories:
        print("No directories found in test_data directory")
        return

    # Initialize API
    api = HfApi()

    # Process each directory
    for directory in directories:
        dir_name = directory.name
        repo_id = f"MoritzLaurer/{dir_name}"

        # Determine repo_type based on README.md tags
        repo_type = "model" if has_model_prompts_tag(directory) else "dataset"

        print(f"\nProcessing directory: {dir_name} as {repo_type}")

        # Ensure repository exists
        try:
            api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True, token=token)
        except Exception as e:
            print(f"Note about repo creation (can usually be ignored): {e}")
            continue

        # Get list of all local prompt files in this directory
        local_files = set()
        for extension in [".yaml", ".yml", ".json", ".md", ".py"]:
            local_files.update(directory.glob(f"*{extension}"))

        if not local_files:
            print(f"No YAML/JSON/MD/PY files found in {dir_name} directory")
            continue

        try:
            # Get existing hub files to check for deletions
            hub_files: Set[str] = {
                f
                for f in api.list_repo_files(repo_id=repo_id, repo_type=repo_type, token=token)
                if f.endswith((".yaml", ".yml", ".json", ".md", ".py"))
            }

            # Upload local files
            for local_file in local_files:
                print(f"Uploading {local_file.name} from {str(local_file)} with repo_id {repo_id}...")
                try:
                    api.upload_file(
                        path_or_fileobj=str(local_file),
                        path_in_repo=local_file.name,
                        repo_id=repo_id,
                        repo_type=repo_type,
                        token=token,
                    )
                    hub_files.discard(local_file.name)
                except Exception as e:
                    print(f"Error uploading {local_file.name}: {e}")
                    continue

            # Delete files on hub that exist on hub but not locally
            for obsolete_file in hub_files:
                print(f"Deleting {obsolete_file} from hub (no longer exists locally)...")
                try:
                    api.delete_file(
                        path_in_repo=obsolete_file,
                        repo_id=repo_id,
                        repo_type=repo_type,
                        token=token,
                    )
                except Exception as e:
                    print(f"Error deleting {obsolete_file}: {e}")
                    continue

            print(f"\nSynced {len(local_files)} files to {repo_id}")
            if hub_files:
                print(f"Deleted {len(hub_files)} obsolete files from hub")
            print(f"View repository at: https://huggingface.co/{repo_id}")

        except Exception as e:
            print(f"Error processing directory {dir_name}: {e}")
            continue


def has_model_prompts_tag(directory: Path) -> bool:
    """Check if README.md in directory has model-prompts tag.
    This enables us to sync the test data to the Hub as a model repo as opposed to the default dataset repo if the README.md contains the tag."""
    readme_path = directory / "README.md"
    if not readme_path.exists():
        return False

    try:
        # Read the README file
        content = readme_path.read_text()

        # Check if there's a YAML front matter
        if not content.startswith("---"):
            return False

        # Extract YAML front matter
        yaml_content = content.split("---")[1]
        metadata = yaml.safe_load(yaml_content)

        # Check if tags exist and if model-prompts is in tags
        return isinstance(metadata.get("tags"), list) and "model-prompts" in metadata["tags"]
    except Exception:
        return False


if __name__ == "__main__":
    sync_test_files()
