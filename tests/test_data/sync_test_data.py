"""
Script to sync local test data to the Hub.
Usage:
poetry run python tests/test_data/sync_test_data.py
"""

import os
from pathlib import Path
from typing import Set

from dotenv import load_dotenv
from huggingface_hub import HfApi


# Load token from .env file
load_dotenv()


def sync_test_files():
    """Upload all local YAML/JSON example prompts to the Hub."""
    token = os.environ.get("HF_TOKEN")

    if not token:
        raise ValueError(
            "No HF_TOKEN found in environment variables. "
            "Please set it with your Hugging Face write token from: "
            "https://huggingface.co/settings/tokens"
        )

    test_data_dir = Path(__file__).parent
    repo_id = "MoritzLaurer/example_prompts"

    # Initialize API
    api = HfApi()

    # Ensure repository exists (this won't overwrite if it already exists)
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)
    except Exception as e:
        print(f"Note about repo creation (can usually be ignored): {e}")

    # Get list of all local prompt files
    local_files = set()
    for extension in [".yaml", ".yml", ".json"]:
        local_files.update(test_data_dir.glob(f"*{extension}"))

    if not local_files:
        print("No YAML/JSON files found in test_data directory")
        return

    try:
        # Get existing hub files to check for deletions (pass token)
        hub_files: Set[str] = {
            f
            for f in api.list_repo_files(repo_id=repo_id, repo_type="model", token=token)
            if f.endswith((".yaml", ".yml", ".json"))
        }

        # Upload local files
        for local_file in local_files:
            print(f"Uploading {local_file.name} from {str(local_file)} with repo_id {repo_id}...")
            try:
                api.upload_file(
                    path_or_fileobj=str(local_file),
                    path_in_repo=local_file.name,
                    repo_id=repo_id,
                    repo_type="model",
                    token=token,
                )
                # Remove from hub_files set to track what was uploaded
                hub_files.discard(local_file.name)
            except Exception as e:
                print(f"Error uploading {local_file.name}: {e}")
                continue

        # Delete files that exist on hub but not locally
        for obsolete_file in hub_files:
            print(f"Deleting {obsolete_file} from hub (no longer exists locally)...")
            try:
                api.delete_file(
                    path_in_repo=obsolete_file,
                    repo_id=repo_id,
                    repo_type="model",
                    token=token,  # Pass the token here
                )
            except Exception as e:
                print(f"Error deleting {obsolete_file}: {e}")
                continue

        print(f"\nSynced {len(local_files)} files to {repo_id}")
        if hub_files:
            print(f"Deleted {len(hub_files)} obsolete files from hub")

    except Exception as e:
        raise ValueError(f"Error syncing files: {e}") from e


if __name__ == "__main__":
    sync_test_files()
