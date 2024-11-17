from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def mock_hf_hub(monkeypatch):
    """Mock Hugging Face Hub API calls for doctests."""

    def mock_download(*args, **kwargs):
        # Map Hub files to local test files
        test_data_dir = Path(__file__).parent / "test_data"

        if "translate.yaml" in str(args):
            return str(test_data_dir / "translate.yaml")
        elif "code_teacher.yaml" in str(args):
            return str(test_data_dir / "code_teacher.yaml")

        raise ValueError(f"Unknown test file requested: {args}")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", mock_download)
