from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Union


SUPPORTED_CLIENT_FORMATS = ["openai", "anthropic"]  # TODO: add more clients


@dataclass
class PopulatedPrompt:
    """A class representing a populated prompt that can be formatted for different LLM clients.

    This class serves two main purposes:
    1. Store populated prompts (either in simple text or chat format)
    2. Convert chat prompts between different LLM client formats (e.g., OpenAI, Anthropic)

    The class handles two types of content:
    * **Text prompts**: Simple strings that can be used directly with any LLM
    * **Chat prompts**: Lists or Dicts of messages that are compatible with the format expected by different LLM clients

    Attributes:
        _content: The populated prompt content, either as a string or a list of message dictionaries.

    Access:
        You can access individual elements of the content using standard indexing or key access:
        - For list-based content: `prompt[index]`
        - For dict-based content: `prompt[key]`

    Examples:
        >>> from prompt_templates import PromptTemplateLoader
        >>> prompt_template = PromptTemplateLoader.from_hub(
        ...     repo_id="MoritzLaurer/example_prompts",
        ...     filename="code_teacher.yaml"
        ... )
        >>> prompt = prompt_template.populate_template(
        ...     concept="list comprehension",
        ...     programming_language="Python"
        ... )
        >>> print(prompt)
        [{'role': 'system', 'content': 'You are a coding assistant who explains concepts clearly and provides short examples.'}, {'role': 'user', 'content': 'Explain what list comprehension is in Python.'}]

        You can also access individual elements of the prompt like with standard lists and dicts:
        >>> print(prompt[0]["content"])
        'You are a coding assistant who explains concepts clearly and provides short examples.'
    """

    _content: Union[str, List[Dict[str, Any]], Dict[str, Any]]

    def __init__(self, content: Union[str, List[Dict[str, Any]], Dict[str, Any]]):
        self._content = content

    def __str__(self) -> str:
        return str(self._content)

    def __repr__(self) -> str:
        return f"PopulatedPrompt({self._content!r})"

    def to_dict(self) -> Union[str, List[Dict[str, Any]], Dict[str, Any]]:
        """Make the class JSON serializable for LLM clients by returning its raw content."""
        return self._content

    def __getattr__(self, name: str) -> Any:
        """Allow the class to be used directly with LLM clients by forwarding attribute access to _content."""
        if name == "__dict__":
            return self.to_dict()
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __getitem__(self, key: Union[int, str]) -> Any:
        """Allow direct access to the content elements."""
        if isinstance(self._content, list):
            if not isinstance(key, int):
                raise TypeError(f"List-based content requires integer index, got {type(key).__name__}")
            return self._content[key]
        elif isinstance(self._content, dict):
            if not isinstance(key, str):
                raise TypeError(f"Dict-based content requires string key, got {type(key).__name__}")
            return self._content[key]
        else:
            raise TypeError(f"Content type {type(self._content).__name__} does not support item access")

    def __iter__(self) -> Iterator[Any]:
        """Make the prompt iterable if content is a list."""
        if isinstance(self._content, (list, dict)):
            return iter(self._content)
        raise TypeError("Content is not iterable")

    def __len__(self) -> int:
        """Get the length of the content."""
        if isinstance(self._content, (list, dict)):
            return len(self._content)
        return len(str(self._content))

    def format_for_client(self, client: str = "openai") -> "PopulatedPrompt":
        """Format the chat messages prompt for a specific LLM client.

        Args:
            client: The client format to use ('openai', 'anthropic'). Defaults to 'openai'.

        Returns:
            PopulatedPrompt: A new PopulatedPrompt instance with content formatted for the specified client.

        Raises:
            ValueError: If an unsupported client format is specified or if trying to format a non-messages template.

        Examples:
            Format chat messages for different LLM clients:
            >>> from prompt_templates import PromptTemplateLoader
            >>> prompt_template = PromptTemplateLoader.from_hub(
            ...     repo_id="MoritzLaurer/example_prompts",
            ...     filename="code_teacher.yaml"
            ... )
            >>> prompt = prompt_template.populate_template(
            ...     concept="list comprehension",
            ...     programming_language="Python"
            ... )
            >>> print(prompt)  # By default in OpenAI format
            [{'role': 'system', 'content': 'You are a coding assistant who explains concepts clearly and provides short examples.'}, {'role': 'user', 'content': 'Explain what list comprehension is in Python.'}]

            >>> # Convert to Anthropic format
            >>> anthropic_prompt = prompt.format_for_client("anthropic")
            >>> print(anthropic_prompt)
            {'system': 'You are a coding assistant who explains concepts clearly and provides short examples.', 'messages': [{'role': 'user', 'content': 'Explain what list comprehension is in Python.'}]}
        """
        if isinstance(self._content, list) and any(isinstance(item, dict) for item in self._content):
            if client == "openai":
                return PopulatedPrompt(self._content)
            elif client == "anthropic":
                return self._format_for_anthropic()
            else:
                raise ValueError(
                    f"Unsupported client format: {client}. Supported formats are: {SUPPORTED_CLIENT_FORMATS}"
                )
        else:
            raise ValueError(
                f"format_for_client is only applicable to chat-based prompts with a list of message dictionaries. "
                f"The content of this prompt is of type: {type(self._content).__name__}. "
                "For standard prompts, you can use the content directly with any client."
            )

    def _format_for_anthropic(self) -> "PopulatedPrompt":
        """Format messages for the Anthropic client.

        Converts OpenAI-style messages to Anthropic's expected format by:
        1. Extracting the system message (if any) into a top-level 'system' key
        2. Moving all non-system messages into a 'messages' list

        Returns:
            PopulatedPrompt: A new PopulatedPrompt instance with content formatted for Anthropic.
        """
        if not isinstance(self._content, list):
            raise TypeError("Cannot format non-list content for Anthropic")

        messages_anthropic: Dict[str, Any] = {
            "system": next((msg["content"] for msg in self._content if msg["role"] == "system"), None),
            "messages": [msg for msg in self._content if msg["role"] != "system"],
        }
        return PopulatedPrompt(messages_anthropic)
