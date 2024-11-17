from dataclasses import dataclass
from typing import Any, Dict, List, Union


SUPPORTED_CLIENT_FORMATS = ["openai", "anthropic"]  # TODO: add more clients


@dataclass
class PopulatedPrompt:
    """A class representing a populated prompt that can be formatted to be compatible with different LLM clients.

    This class serves two main purposes:
    1. Store populated prompts (either in simple text or chat format)
    2. Convert chat prompts between different LLM client formats (e.g., OpenAI, Anthropic)

    The class handles two types of content:

    * **Text prompts**: Simple strings that can be used directly with any LLM
    * **Chat prompts**: Lists or Dicts of messages that are compatible with the format expected by different LLM clients

    For examples of converting between client formats, see the [`format_for_client()`][hf_hub_prompts.populated_prompt.PopulatedPrompt.format_for_client] method.
    """

    content: Union[str, List[Dict[str, Any]]]

    def format_for_client(self, client: str = "openai") -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Format the prompt content for a specific client.

        Examples:
            Format chat messages for different clients:
            >>> from hf_hub_prompts import download_prompt_template
            >>> prompt_template = download_prompt_template(
            ...     repo_id="MoritzLaurer/example_prompts",
            ...     filename="code_teacher.yaml"
            ... )
            >>> prompt = prompt_template.populate_template(
            ...     concept="list comprehension",
            ...     programming_language="Python"
            ... )
            >>> prompt.content
            [{'role': 'system', 'content': 'You are a coding assistant who explains concepts clearly and provides short examples.'}, {'role': 'user', 'content': 'Explain what list comprehension is in Python.'}]

            >>> # By default, the populated prompt.content is in the OpenAI messages format
            >>> messages_openai = prompt.format_for_client("openai")
            >>> messages_openai == prompt.content
            True

            >>> # We can also convert the populated prompt to other formats
            >>> messages_anthropic = prompt.format_for_client("anthropic")
            >>> messages_anthropic == prompt.content
            False
            >>> messages_anthropic
            {'system': 'You are a coding assistant who explains concepts clearly and provides short examples.', 'messages': [{'role': 'user', 'content': 'Explain what list comprehension is in Python.'}]}

        Args:
            client (str): The client format to use ('openai', 'anthropic'). Defaults to 'openai'.

        Returns:
            Union[List[Dict[str, Any]], Dict[str, Any]]: Formatted prompt content suitable for the specified client.

        Raises:
            ValueError: If an unsupported client format is specified or if trying to format a text prompt.
        """
        if isinstance(self.content, str):
            # For standard prompts, format_for_client does not add value
            raise ValueError(
                f"format_for_client is only applicable to chat-based prompts with a list of messages. "
                f"The content of this prompt is of type: {type(self.content).__name__}. "
                "For standard prompts, you can use the content directly with any client."
            )
        elif isinstance(self.content, list):
            # For chat prompts, format messages accordingly
            if client == "openai":
                return self.content
            elif client == "anthropic":
                return self._format_for_anthropic(self.content)
            else:
                raise ValueError(
                    f"Unsupported client format: {client}. Supported formats are: {SUPPORTED_CLIENT_FORMATS}"
                )
        else:
            raise ValueError("PopulatedPrompt content must be either a string or a list of messages.")

    def _format_for_anthropic(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format messages for the Anthropic client."""
        messages_anthropic = {
            "system": next((msg["content"] for msg in messages if msg["role"] == "system"), None),
            "messages": [msg for msg in messages if msg["role"] != "system"],
        }
        return messages_anthropic
