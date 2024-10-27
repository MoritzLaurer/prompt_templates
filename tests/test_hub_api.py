# tests/test_hub_api.py
import pytest
from unittest.mock import Mock, patch
import yaml
import tempfile
import os

from hf_hub_prompts import download_prompt, list_prompts
from hf_hub_prompts import TextPromptTemplate, ChatPromptTemplate

# Sample YAML contents for testing
TEXT_PROMPT_YAML = """
prompt:
  template: "Hello {name}"
  input_variables: ["name"]
"""

CHAT_PROMPT_YAML = """
prompt:
  messages:
    - role: system
      content: "You are helpful."
    - role: user
      content: "Hello {name}"
  input_variables: ["name"]
"""

INVALID_YAML = """
not_prompt:
  template: "Hello {name}"
"""

@pytest.fixture
def mock_yaml_file():
    """Create a temporary YAML file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(TEXT_PROMPT_YAML)
        return f.name

@pytest.fixture
def mock_chat_yaml_file():
    """Create a temporary YAML file for testing chat prompts."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(CHAT_PROMPT_YAML)
        return f.name

def test_download_prompt_text(mock_yaml_file):
    """Test downloading a text prompt template."""
    with patch('hf_hub_prompts.hub_api.hf_hub_download') as mock_download:
        mock_download.return_value = mock_yaml_file
        
        template = download_prompt("test/repo", "prompt.yaml")
        
        assert isinstance(template, TextPromptTemplate)
        assert template.template == "Hello {name}"
        assert template.input_variables == ["name"]

def test_download_prompt_chat(mock_chat_yaml_file):
    """Test downloading a chat prompt template."""
    with patch('hf_hub_prompts.hub_api.hf_hub_download') as mock_download:
        mock_download.return_value = mock_chat_yaml_file
        
        template = download_prompt("test/repo", "prompt.yaml")
        
        assert isinstance(template, ChatPromptTemplate)
        assert len(template.messages) == 2
        assert template.input_variables == ["name"]

def test_list_prompts():
    """Test listing prompt files from a repository."""
    with patch('hf_hub_prompts.hub_api.HfApi') as mock_hf_api:
        mock_api_instance = Mock()
        mock_api_instance.list_repo_files.return_value = [
            "prompt1.yaml",
            "prompt2.yml",
            "not_a_prompt.txt"
        ]
        mock_hf_api.return_value = mock_api_instance
        
        files = list_prompts("test/repo")
        
        assert len(files) == 2
        assert "prompt1.yaml" in files
        assert "prompt2.yml" in files
        assert "not_a_prompt.txt" not in files

# Cleanup temp files after tests
def teardown_module(module):
    """Clean up any temporary files created during testing."""
    for fixture in [mock_yaml_file, mock_chat_yaml_file]:
        try:
            os.unlink(fixture)
        except:
            pass