"""
Script to sync local test data to the Hub.
Usage:
poetry run python tests/test_data/sync_test_data.py [--force]
"""

import argparse
import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from huggingface_hub import HfApi
from tqdm import tqdm

from prompt_templates import ChatPromptTemplate, TextPromptTemplate


MODEL_REPOS = ["open_models_special_prompts"]
JINJA2_PROMPTS = ["translate_jinja2"]


# Load token from .env file
load_dotenv()


def sync_test_files(force: bool = False):
    """Upload all local YAML/JSON example prompt templates to the Hub for each directory.

    Args:
        force: If True, skip confirmation for local file updates
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "No HF_TOKEN found in environment variables. "
            "Please set it with your Hugging Face write token from: "
            "https://huggingface.co/settings/tokens"
        )

    test_data_dir = Path(__file__).parent
    api = HfApi(token=token)

    # Get all directories in the test data folder, excluding __pycache__
    directories = [d for d in test_data_dir.iterdir() if d.is_dir() and d.name != "__pycache__"]

    if not directories:
        print("No directories found in test_data directory")
        return

    for directory in tqdm(directories, desc="Processing directories", leave=False):
        dir_name = directory.name
        repo_id = f"MoritzLaurer/{dir_name}"

        # Determine repo_type
        repo_type = "model" if dir_name in MODEL_REPOS else "dataset"

        print(f"\nProcessing directory: {dir_name}   (as repo type {repo_type})")

        # Get all local and remote files for the respective directory and it's mirror Hub repo
        local_files = {f for ext in [".yaml", ".yml", ".json"] for f in directory.glob(f"*{ext}")}
        try:
            hub_files = {
                f
                for f in api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
                if f.endswith((".yaml", ".yml", ".json"))
            }
        except Exception as e:
            print(f"Error accessing repo {repo_id}: {e}")
            continue

        # Process local files
        for local_file in local_files:
            try:
                # Load and standardize through library
                populator = "jinja2" if local_file.stem in JINJA2_PROMPTS else "double_brace"
                try:
                    template = ChatPromptTemplate.load_from_local(path=local_file, populator=populator)
                except Exception:
                    template = TextPromptTemplate.load_from_local(path=local_file, populator=populator)

                # Save standardized version locally
                standardized_path = directory / f"{local_file.stem}_standardized{local_file.suffix}"
                template.save_to_local(standardized_path)

                # Compare standardized with original
                if not files_are_equivalent(local_file, standardized_path):
                    print(f"Warning: {local_file.name} was modified during standardization")
                    # Skip confirmation if force flag is set
                    if force or input("Update local file? [y/N]: ").lower() == "y":
                        standardized_path.replace(local_file)
                else:
                    standardized_path.unlink()  # Remove if identical

                # Upload to hub
                template.save_to_hub(
                    repo_id=repo_id,
                    filename=local_file.name,
                    repo_type=repo_type,
                    token=token,
                    create_repo=True,
                    exist_ok=True,
                )
                print(f"Synced {local_file.name}")

            except Exception as e:
                print(f"Error processing {local_file}: {e}")
                continue

        # Delete files that exist on hub but not locally
        obsolete_files = hub_files - {f.name for f in local_files}
        for obsolete_file in obsolete_files:
            try:
                api.delete_file(
                    path_in_repo=obsolete_file,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    token=token,
                )
                print(f"Deleted {obsolete_file} from hub, because it was not present in local directory")
            except Exception as e:
                print(f"Error deleting {obsolete_file}: {e}")

        print(f"View repository at: https://huggingface.co/{repo_id}")


def files_are_equivalent(file1: Path, file2: Path) -> bool:
    """Compare two YAML/JSON files, ignoring formatting differences."""

    def load_file(path: Path) -> dict:
        with open(path) as f:
            return yaml.safe_load(f) if path.suffix in [".yaml", ".yml"] else json.load(f)

    try:
        return load_file(file1) == load_file(file2)
    except Exception:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync local test data to the Hub")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompts for local file updates")
    args = parser.parse_args()

    sync_test_files(force=args.force)
