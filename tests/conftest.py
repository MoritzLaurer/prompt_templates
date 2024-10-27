import pytest
import yaml


@pytest.fixture
def sample_text_yaml():
    return {
        "prompt": {
            "template": "Hello {name}, how are you?",
            "input_variables": ["name"],
            "metadata": {"type": "greeting"},
        }
    }


@pytest.fixture
def sample_chat_yaml():
    return {
        "prompt": {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello {name}, how are you?"},
            ],
            "input_variables": ["name"],
            "metadata": {"type": "chat_greeting"},
        }
    }


@pytest.fixture
def temp_yaml_file(tmp_path, sample_text_yaml):
    """Creates a temporary YAML file for testing hub downloads"""
    file_path = tmp_path / "test_prompt.yaml"
    with open(file_path, "w") as f:
        yaml.dump(sample_text_yaml, f)
    return file_path
