from dataclasses import dataclass
from typing import Any, Dict, List, Union


SUPPORTED_CLIENT_FORMATS = ["openai", "anthropic"]  # TODO: add more clients


@dataclass
class PopulatedPrompt:
    """A class representing a populated prompt.

    Examples:
        >>> # For standard prompts
        >>> prompt = template.populate_template(name="Alice")
        >>> text = prompt.content
        >>>
        >>> # For chat prompts
        >>> prompt = chat_template.populate_template(name="Alice")
        >>> messages = prompt.format_for_client(client="anthropic")
    """

    content: Union[str, List[Dict[str, Any]]]

    def format_for_client(self, client: str = "openai") -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Format the prompt content for a specific client.

        Args:
            client (str): The client format to use ('openai', 'anthropic'). Defaults to 'openai'.

        Returns:
            Union[List[Dict[str, Any]], Dict[str, Any]]: Formatted prompt content suitable for the specified client.

        Raises:
            ValueError: If an unsupported client format is specified.
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
