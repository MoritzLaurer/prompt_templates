import logging
from typing import Any, Dict, List, Optional, Union, get_args

import yaml as pyyaml
from huggingface_hub import HfApi
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString

from .constants import VALID_PROMPT_EXTENSIONS, ClientType


logger = logging.getLogger(__name__)


def list_prompt_templates(
    repo_id: str, repo_type: Optional[str] = "dataset", token: Optional[str] = None
) -> List[str]:
    """List available prompt template YAML files in a Hugging Face Hub repository.

    Args:
        repo_id (str): The repository ID on Hugging Face Hub.
        repo_type (Optional[str]): The type of repository. Defaults to "dataset".
        token (Optional[str]): An optional authentication token. Defaults to None.

    Returns:
        List[str]: A list of YAML filenames in the repository sorted alphabetically.

    Examples:
        List all prompt templates in a repository:
        >>> from prompt_templates import list_prompt_templates
        >>> files = list_prompt_templates("MoritzLaurer/example_prompts")
        >>> files
        ['code_teacher.yaml', 'translate.yaml', 'translate_jinja2.yaml']

    Note:
        This function simply returns all YAML file names in the repository.
        It does not validate if the files contain valid prompt templates, which would require downloading them.
    """
    logger.info(
        "This function simply returns all YAML file names in the repository. "
        "It does not validate if the files contain valid prompt templates, which would require downloading them."
    )
    api = HfApi(token=token)
    yaml_files = [
        file for file in api.list_repo_files(repo_id, repo_type=repo_type) if file.endswith(VALID_PROMPT_EXTENSIONS)
    ]
    return sorted(yaml_files)


def format_for_client(
    messages: List[Dict[str, Any]], client: ClientType = "openai"
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """Format OpenAI-style chat messages for different LLM clients.

    Args:
        messages: List of message dictionaries in OpenAI format
        client: The client format to use ('openai', 'anthropic', 'google'). Defaults to 'openai'

    Returns:
        Messages formatted for the specified client

    Raises:
        ValueError: If an unsupported client format is specified
        TypeError: If messages is not a list of dicts

    Examples:
        Format messages for different LLM clients:
        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful assistant"},
        ...     {"role": "user", "content": "Hello!"}
        ... ]
        >>> # OpenAI format (default, no change)
        >>> openai_messages = format_for_client(messages)
        >>> print(openai_messages)
        [{'role': 'system', 'content': 'You are a helpful assistant'}, {'role': 'user', 'content': 'Hello!'}]

        >>> # Anthropic format
        >>> anthropic_messages = format_for_client(messages, "anthropic")
        >>> print(anthropic_messages)
        {'system': 'You are a helpful assistant', 'messages': [{'role': 'user', 'content': 'Hello!'}]}

        >>> # Google (Gemini) format
        >>> google_messages = format_for_client(messages, "google")
        >>> print(google_messages)
        {'system_instruction': 'You are a helpful assistant', 'contents': 'Hello!'}
    """
    if not isinstance(messages, list) or not all(isinstance(msg, dict) for msg in messages):
        raise TypeError("Messages must be a list of dictionaries")

    if client == "openai":
        return messages
    elif client == "anthropic":
        return format_for_anthropic(messages)
    elif client == "google":
        return format_for_google(messages)
    else:
        raise ValueError(f"Unsupported client format: {client}. Supported formats are: {list(get_args(ClientType))}")


def format_for_anthropic(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Format OpenAI-style messages for the Anthropic client.

    Converts OpenAI-style messages to Anthropic's expected format by:
    1. Extracting the system message (if any) into a top-level 'system' key
    2. Moving all non-system messages into a 'messages' list

    Args:
        messages: List of message dictionaries in OpenAI format

    Returns:
        Dict with 'system' and 'messages' keys formatted for Anthropic
    """
    return {
        "system": next((msg["content"] for msg in messages if msg["role"] == "system"), None),
        "messages": [msg for msg in messages if msg["role"] != "system"],
    }


def format_for_google(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Format OpenAI-style messages for the Google GenAI SDK's generate_content method.

    Converts OpenAI-style messages to Google Gemini's expected format by:
    1. Extracting the system message (if any) into a top-level 'system_instruction' key
    2. Moving all non-system messages into a 'contents' list of messages as `Part` objects
    (or a single string if there's only one message).

    Args:
        messages: List of message dictionaries in OpenAI format

    Returns:
        Dict with 'system_instruction' and 'contents' keys formatted for Google Gemini
    """
    from google.genai import types

    system_instruction: Optional[str] = None
    contents: List[types.Content] = []

    for msg in messages:
        if msg["role"] == "system":
            system_instruction = msg["content"]
        elif msg["role"] == "user":
            contents.append(types.Content(parts=[types.Part.from_text(msg["content"])], role="user"))
        elif msg["role"] == "assistant":
            contents.append(types.Content(parts=[types.Part.from_text(msg["content"])], role="model"))
        else:
            raise ValueError(f"Unsupported role: {msg['role']}")

    # If there's only one message, simplify to just the text content
    if len(contents) == 1:
        contents = contents[0].parts[0].text

    return {
        "system_instruction": system_instruction,
        "contents": contents,
    }


def create_yaml_handler(library: str = "ruamel") -> Union[YAML, Any]:
    """Create a YAML handler with the specified configuration.
    Ruamel is the default, because it allows for better format preservation and defaults to the newer YAML 1.2.
    Pyyaml can also be used, as it can be faster and is more widely used.

    Args:
        library: The YAML library to use ("ruamel" or "pyyaml"). Defaults to "ruamel".

    Returns:
        A configured YAML handler

    Raises:
        ValueError: If an unsupported YAML library is specified
    """
    if library == "ruamel":
        yaml = YAML(typ="rt")
        yaml.preserve_quotes = True
        yaml.default_flow_style = False
        yaml.width = 120
        yaml.indent(mapping=2, sequence=4, offset=2)
        return yaml
    elif library == "pyyaml":
        return pyyaml
    else:
        raise ValueError(f"Unsupported YAML library: {library}")


def format_template_content(node: Any) -> Any:
    '''Recursively format content strings to use YAML literal block scalars.
    This is used to make the string outputs in a yaml file contain "|-",
    which makes the string behave like a """...""" block in python
    to make strings easier to read and edit.

    Args:
        node: The prompt template content to format

    Returns:
        The formatted content with literal block scalars for multiline strings
    '''
    if isinstance(node, dict):
        for key, value in node.items():
            node[key] = LiteralScalarString(value.strip()) if key in ["content", "text"] else value
        return node
    elif isinstance(node, str):
        if "\n" in node or len(node) > 80:
            return LiteralScalarString(node.strip())
        else:
            return node
    else:
        return node
