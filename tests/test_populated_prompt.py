# tests/test_populated_prompt.py
import pytest
from hf_hub_prompts import PopulatedPrompt

def test_populated_prompt_with_string():
    """Test PopulatedPrompt with a string content."""
    prompt = PopulatedPrompt(content="Hello Alice")
    assert prompt.content == "Hello Alice"

def test_populated_prompt_with_messages():
    """Test PopulatedPrompt with a list of messages."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello Alice"}
    ]
    prompt = PopulatedPrompt(content=messages)
    assert prompt.content == messages

def test_format_for_client_with_string_raises_error():
    """Test that format_for_client raises error with string content."""
    prompt = PopulatedPrompt(content="Hello Alice")
    with pytest.raises(ValueError) as exc_info:
        prompt.format_for_client()
    assert "format_for_client is only applicable to chat-based prompts" in str(exc_info.value)
    assert "type: str" in str(exc_info.value)

def test_format_for_client_with_invalid_type_raises_error():
    """Test that format_for_client raises error with invalid content type."""
    prompt = PopulatedPrompt(content={"invalid": "type"})  # type: ignore
    with pytest.raises(ValueError) as exc_info:
        prompt.format_for_client()
    assert "must be either a string or a list of messages" in str(exc_info.value)

def test_format_for_client_with_invalid_client_raises_error():
    """Test that format_for_client raises error with invalid client."""
    messages = [{"role": "user", "content": "Hello"}]
    prompt = PopulatedPrompt(content=messages)
    with pytest.raises(ValueError) as exc_info:
        prompt.format_for_client(client="invalid")
    assert "Unsupported client format: invalid" in str(exc_info.value)

@pytest.mark.parametrize("client,expected_type", [
    ("openai", list),
    ("anthropic", dict)
])
def test_format_for_client_returns_correct_type(client, expected_type):
    """Test that format_for_client returns correct type for each client."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"}
    ]
    prompt = PopulatedPrompt(content=messages)
    formatted = prompt.format_for_client(client=client)
    assert isinstance(formatted, expected_type)

def test_format_for_anthropic():
    """Test specific formatting for Anthropic client."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"}
    ]
    prompt = PopulatedPrompt(content=messages)
    formatted = prompt.format_for_client(client="anthropic")
    
    assert "system" in formatted
    assert "messages" in formatted
    assert formatted["system"] == "You are helpful."
    assert len(formatted["messages"]) == 2  # system message removed
    assert all(msg["role"] != "system" for msg in formatted["messages"])

def test_format_for_anthropic_without_system():
    """Test Anthropic formatting when no system message is present."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"}
    ]
    prompt = PopulatedPrompt(content=messages)
    formatted = prompt.format_for_client(client="anthropic")
    
    assert formatted["system"] is None
    assert len(formatted["messages"]) == 2