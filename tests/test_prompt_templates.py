import pytest

from hf_hub_prompts import ChatPromptTemplate, TextPromptTemplate


def test_text_prompt_template_initialization(sample_text_yaml):
    prompt_data = sample_text_yaml["prompt"]
    template = TextPromptTemplate(**prompt_data)

    assert template.template == "Hello {name}, how are you?"
    assert template.input_variables == ["name"]
    assert template.metadata == {"type": "greeting"}


def test_text_prompt_template_population(sample_text_yaml):
    prompt_data = sample_text_yaml["prompt"]
    template = TextPromptTemplate(**prompt_data)

    populated = template.populate_template(name="Alice")
    assert populated.content == "Hello Alice, how are you?"


def test_chat_prompt_template_initialization(sample_chat_yaml):
    prompt_data = sample_chat_yaml["prompt"]
    template = ChatPromptTemplate(**prompt_data)

    assert len(template.messages) == 2
    assert template.messages[0]["role"] == "system"
    assert template.input_variables == ["name"]


@pytest.mark.parametrize(
    "client,expected_type",
    [
        ("openai", list),
        ("anthropic", dict),
    ],
)
def test_chat_prompt_template_client_formats(sample_chat_yaml, client, expected_type):
    prompt_data = sample_chat_yaml["prompt"]
    template = ChatPromptTemplate(**prompt_data)

    populated = template.populate_template(name="Alice")
    formatted = populated.format_for_client(client)
    assert isinstance(formatted, expected_type)
