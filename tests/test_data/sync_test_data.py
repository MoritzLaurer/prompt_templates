"""
Script to sync example prompts from the Hub to local test data.
Usage:
poetry run python tests/test_data/sync_test_data.py
"""

from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def sync_test_files():
    """Download all YAML/JSON example prompts from Hub and save to test_data directory."""
    test_data_dir = Path(__file__).parent.parent / "test_data"
    test_data_dir.mkdir(exist_ok=True, parents=True)

    # Get list of all files in the repo
    api = HfApi()
    repo_id = "MoritzLaurer/example_prompts"
    all_files = api.list_repo_files(repo_id)

    # Filter for YAML and JSON files
    prompt_files = [file for file in all_files if file.endswith((".yaml", ".yml", ".json"))]

    # Download each file
    for file in prompt_files:
        print(f"Downloading {file}...")
        hub_file = hf_hub_download(repo_id=repo_id, filename=file)
        local_file = test_data_dir / Path(file).name
        # Copy content to local test file
        with open(hub_file, "r") as src, open(local_file, "w") as dst:
            dst.write(src.read())
        print(f"Saved to {local_file}")

    print(f"\nSynced {len(prompt_files)} files to {test_data_dir}")


if __name__ == "__main__":
    sync_test_files()
