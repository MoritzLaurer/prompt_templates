import json
import tempfile

import pytest

from prompt_templates import ChatPromptTemplate, TextPromptTemplate


@pytest.fixture
def sample_chat_template():
    return {
        "template": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello {{name}}!"},
        ],
        "template_variables": ["name"],
        "metadata": {"name": "Test Template", "description": "A test template", "version": "0.0.1"},
        "client_parameters": {"temperature": 0.7},
        "custom_data": {"test_key": "test_value"},
    }


@pytest.fixture
def sample_text_template():
    return {
        "template": "Hello {{name}}!",
        "template_variables": ["name"],
        "metadata": {"name": "Test Template"},
        "client_parameters": {},
        "custom_data": {},
    }


class TestChatPromptTemplate:
    def test_initialization(self, sample_chat_template):
        template = ChatPromptTemplate(**sample_chat_template)
        assert template.template == sample_chat_template["template"]
        assert template.template_variables == sample_chat_template["template_variables"]
        assert template.metadata == sample_chat_template["metadata"]
        assert template.client_parameters == sample_chat_template["client_parameters"]
        assert template.custom_data == sample_chat_template["custom_data"]

    def test_populate(self, sample_chat_template):
        template = ChatPromptTemplate(**sample_chat_template)
        result = template.populate(name="World")
        assert result[0]["content"] == "You are a helpful assistant"
        assert result[1]["content"] == "Hello World!"

    def test_invalid_template(self):
        with pytest.raises(ValueError):
            ChatPromptTemplate(template=[{"invalid": "format"}])

    def test_missing_required_variables(self, sample_chat_template):
        template = ChatPromptTemplate(**sample_chat_template)
        with pytest.raises(ValueError, match="Missing required variables"):
            template.populate(wrong_var="World")

    def test_unexpected_variables(self, sample_chat_template):
        template = ChatPromptTemplate(**sample_chat_template)
        with pytest.raises(ValueError, match="Unexpected variables provided"):
            template.populate(name="World", extra_var="Extra")

    def test_save_and_load_local_yaml(self, sample_chat_template):
        template = ChatPromptTemplate(**sample_chat_template)
        with tempfile.NamedTemporaryFile(suffix=".yaml") as tmp:
            # Save
            template.save_to_local(tmp.name)
            # Load
            loaded = ChatPromptTemplate.load_from_local(tmp.name)
            assert template == loaded

    # def test_save_and_load_local_json(self, sample_chat_template):
    #    template = ChatPromptTemplate(**sample_chat_template)
    #    with tempfile.NamedTemporaryFile(suffix='.json') as tmp:
    #        # Save
    #        template.save_to_local(tmp.name)
    #        # Load
    #        loaded = ChatPromptTemplate.load_from_local(tmp.name)
    #        assert template == loaded

    def test_display_json(self, sample_chat_template, capsys):
        template = ChatPromptTemplate(**sample_chat_template)
        template.display(format="json")
        captured = capsys.readouterr()
        displayed_dict = json.loads(captured.out)
        assert displayed_dict["template"] == sample_chat_template["template"]

    def test_display_yaml(self, sample_chat_template, capsys):
        template = ChatPromptTemplate(**sample_chat_template)
        template.display(format="yaml")
        captured = capsys.readouterr()
        assert "template:" in captured.out
        assert "metadata:" in captured.out

    def test_invalid_file_extension(self, sample_chat_template):
        template = ChatPromptTemplate(**sample_chat_template)
        with pytest.raises(ValueError, match="Cannot infer format"):
            template.save_to_local("invalid.txt")

    def test_format_mismatch(self, sample_chat_template):
        template = ChatPromptTemplate(**sample_chat_template)
        with pytest.raises(ValueError, match="does not match file extension"):
            template.save_to_local("test.yaml", format="json")

    def test_different_populator_types(self, sample_chat_template):
        # Test Jinja2
        template_jinja = ChatPromptTemplate(**sample_chat_template, populator="jinja2")
        result_jinja = template_jinja.populate(name="World")
        assert result_jinja[1]["content"] == "Hello World!"

        # Test double brace
        template_double = ChatPromptTemplate(**sample_chat_template, populator="double_brace_regex")
        result_double = template_double.populate(name="World")
        assert result_double[1]["content"] == "Hello World!"

        # Test single brace
        template_single = ChatPromptTemplate(
            template=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello {name}!"},
            ],
            template_variables=["name"],
            populator="single_brace_regex",
        )
        result_single = template_single.populate(name="World")
        assert result_single[1]["content"] == "Hello World!"

    def test_jinja2_security_levels(self, sample_chat_template):
        # Test strict mode
        template_strict = ChatPromptTemplate(
            **sample_chat_template, populator="jinja2", jinja2_security_level="strict"
        )
        result_strict = template_strict.populate(name="World")
        assert result_strict[1]["content"] == "Hello World!"

        # Test invalid security level
        with pytest.raises(ValueError, match="Invalid security level"):
            ChatPromptTemplate(**sample_chat_template, populator="jinja2", jinja2_security_level="invalid")


class TestTextPromptTemplate:
    def test_initialization(self, sample_text_template):
        template = TextPromptTemplate(**sample_text_template)
        assert template.template == sample_text_template["template"]
        assert template.template_variables == sample_text_template["template_variables"]
        assert template.metadata == sample_text_template["metadata"]

    def test_populate(self, sample_text_template):
        template = TextPromptTemplate(**sample_text_template)
        result = template.populate(name="World")
        assert result == "Hello World!"

    def test_missing_required_variables(self, sample_text_template):
        template = TextPromptTemplate(**sample_text_template)
        with pytest.raises(ValueError, match="Missing required variables"):
            template.populate(wrong_var="World")

    def test_save_and_load_local(self, sample_text_template):
        template = TextPromptTemplate(**sample_text_template)
        with tempfile.NamedTemporaryFile(suffix=".yaml") as tmp:
            template.save_to_local(tmp.name)
            loaded = TextPromptTemplate.load_from_local(tmp.name)
            assert template == loaded

    def test_equality(self, sample_text_template):
        template1 = TextPromptTemplate(**sample_text_template)
        template2 = TextPromptTemplate(**sample_text_template)
        template3 = TextPromptTemplate(template="Different {{name}}!", template_variables=["name"])

        assert template1 == template2
        assert template1 != template3
        assert template1 != "not a template"
