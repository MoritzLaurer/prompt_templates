from unittest.mock import Mock, patch

import pytest
from ruamel.yaml.scalarstring import LiteralScalarString

from prompt_templates.utils import (
    format_for_client,
    format_template_content,
    list_prompt_templates,
)


# Test data fixtures
@pytest.fixture
def sample_messages():
    return [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]


@pytest.fixture
def sample_messages_no_system():
    return [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]


class TestListPromptTemplates:
    @patch("prompt_templates.utils.HfApi")
    def test_list_prompt_templates(self, mock_hf_api):
        # Mock HfApi response
        mock_api_instance = Mock()
        mock_api_instance.list_repo_files.return_value = [
            "template1.yaml",
            "template2.yml",
            "not_a_template.txt",
            "subfolder/template3.yaml",
        ]
        mock_hf_api.return_value = mock_api_instance

        result = list_prompt_templates("test/repo")

        # Verify results
        assert result == ["subfolder/template3.yaml", "template1.yaml", "template2.yml"]
        mock_hf_api.assert_called_once_with(token=None)
        mock_api_instance.list_repo_files.assert_called_once_with("test/repo", repo_type="dataset")

    @patch("prompt_templates.utils.HfApi")
    def test_list_prompt_templates_with_token(self, mock_hf_api):
        mock_api_instance = Mock()
        mock_api_instance.list_repo_files.return_value = ["template1.yaml"]
        mock_hf_api.return_value = mock_api_instance

        list_prompt_templates("test/repo", token="test_token")
        mock_hf_api.assert_called_once_with(token="test_token")
        mock_api_instance.list_repo_files.assert_called_once_with("test/repo", repo_type="dataset")

    @patch("prompt_templates.utils.HfApi")
    def test_list_prompt_templates_custom_repo_type(self, mock_hf_api):
        mock_api_instance = Mock()
        mock_api_instance.list_repo_files.return_value = ["template1.yaml"]
        mock_hf_api.return_value = mock_api_instance

        list_prompt_templates("test/repo", repo_type="model")
        mock_hf_api.assert_called_once_with(token=None)
        mock_api_instance.list_repo_files.assert_called_once_with("test/repo", repo_type="model")


class TestFormatForClient:
    def test_format_for_openai(self, sample_messages):
        result = format_for_client(sample_messages, "openai")
        assert result == sample_messages

    def test_format_for_anthropic(self, sample_messages):
        result = format_for_client(sample_messages, "anthropic")
        assert result["system"] == "You are helpful"
        assert len(result["messages"]) == 2
        assert result["messages"][0]["content"] == "Hi"

    def test_format_for_anthropic_no_system(self, sample_messages_no_system):
        result = format_for_client(sample_messages_no_system, "anthropic")
        assert result["system"] is None
        assert len(result["messages"]) == 2

    @patch("google.genai.types")
    def test_format_for_google(self, mock_types, sample_messages):
        # Mock Google types
        mock_content = Mock()
        mock_part = Mock()
        mock_content.parts = [Mock(text="Hi")]  # Add parts attribute for single message case
        mock_types.Content = Mock(return_value=mock_content)
        mock_types.Part.from_text = Mock(return_value=mock_part)

        result = format_for_client(sample_messages, "google")

        assert result["system_instruction"] == "You are helpful"
        # Verify Content creation for user and assistant messages
        assert mock_types.Content.call_count == 2
        mock_types.Content.assert_any_call(parts=[mock_part], role="user")
        mock_types.Content.assert_any_call(parts=[mock_part], role="model")

    @patch("google.genai.types")
    def test_format_for_google_single_message(self, mock_types):
        messages = [{"role": "user", "content": "Hi"}]
        # Mock Google types
        mock_content = Mock()
        mock_part = Mock()
        mock_content.parts = [mock_part]
        mock_part.text = "Hi"
        mock_types.Content = Mock(return_value=mock_content)
        mock_types.Part.from_text = Mock(return_value=mock_part)

        result = format_for_client(messages, "google")

        assert result["system_instruction"] is None
        assert result["contents"] == "Hi"  # For single message, should be just the text

    @patch("google.genai.types")
    def test_format_for_google_invalid_role(self, mock_types):
        messages = [{"role": "invalid", "content": "Hi"}]
        with pytest.raises(ValueError, match="Unsupported role"):
            format_for_client(messages, "google")

    def test_invalid_client(self, sample_messages):
        with pytest.raises(ValueError, match="Unsupported client format"):
            format_for_client(sample_messages, "invalid_client")

    def test_invalid_message_format(self):
        with pytest.raises(TypeError, match="Messages must be a list of dictionaries"):
            format_for_client("not a list", "openai")
        with pytest.raises(TypeError, match="Messages must be a list of dictionaries"):
            format_for_client([1, 2, 3], "openai")


class TestFormatTemplateContent:
    def test_format_dict_content(self):
        input_dict = {"content": "test\nstring", "other": "value"}
        result = format_template_content(input_dict)
        assert isinstance(result["content"], LiteralScalarString)
        assert result["other"] == "value"

    def test_format_long_string(self):
        long_string = "x" * 100
        result = format_template_content(long_string)
        assert isinstance(result, LiteralScalarString)

    def test_format_multiline_string(self):
        multiline = "line1\nline2"
        result = format_template_content(multiline)
        assert isinstance(result, LiteralScalarString)

    def test_format_short_string(self):
        short_string = "short"
        result = format_template_content(short_string)
        assert result == short_string
        assert not isinstance(result, LiteralScalarString)

    def test_format_non_string(self):
        non_string = 123
        result = format_template_content(non_string)
        assert result == non_string
