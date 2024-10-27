import json
import logging
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Match, Optional, Union

import yaml

from .populated_prompt import PopulatedPrompt


if TYPE_CHECKING:
    from langchain.prompts import (
        ChatPromptTemplate as LC_ChatPromptTemplate,
    )
    from langchain.prompts import (
        PromptTemplate as LC_PromptTemplate,
    )

logger = logging.getLogger(__name__)


class BasePromptTemplate(ABC):
    """An abstract base class for prompt templates."""

    def __init__(
        self, full_yaml_content: Optional[Dict[str, Any]] = None, prompt_url: Optional[str] = None, **kwargs: Any
    ) -> None:
        # Set all YAML file keys as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.full_yaml_content = full_yaml_content
        self.prompt_url = prompt_url

    @abstractmethod
    def populate_template(self, **input_variables: Any) -> PopulatedPrompt:
        """Abstract method to populate the prompt template with the given variables.

        Args:
            **input_variables: The values to fill placeholders in the template.

        Returns:
            PopulatedPrompt: A PopulatedPrompt object containing the populated content.
        """
        pass

    def display(self, format: Literal["json", "yaml"] = "json") -> None:
        """Display the full prompt YAML file content in the specified format.

        Args:
            format (Literal['json', 'yaml']): The format to display ('json' or 'yaml'). Defaults to 'json'.

        Raises:
            ValueError: If an unsupported format is specified.
        """
        if format == "json":
            print(json.dumps(self.full_yaml_content, indent=2))
        elif format == "yaml":
            print(yaml.dump(self.full_yaml_content, default_flow_style=False, sort_keys=False))
        else:
            raise ValueError(f"Unsupported format: {format}")

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]

    def __repr__(self) -> str:
        attributes = ", ".join(
            f"{key}={repr(value)[:50]}..." if len(repr(value)) > 50 else f"{key}={repr(value)}"
            for key, value in self.__dict__.items()
        )
        return f"{self.__class__.__name__}({attributes})"

    def _fill_placeholders(self, template_part: Any, input_variables: Dict[str, Any]) -> Any:
        """Recursively fill placeholders in strings or nested structures like dicts or lists."""
        pattern = re.compile(r"\{([^{}]+)\}")

        if isinstance(template_part, str):
            # fill placeholders in strings
            def replacer(match: Match[str]) -> str:
                key = match.group(1).strip()
                return str(input_variables.get(key, match.group(0)))

            return pattern.sub(replacer, template_part)

        elif isinstance(template_part, dict):
            # Recursively handle dictionaries
            return {key: self._fill_placeholders(value, input_variables) for key, value in template_part.items()}

        elif isinstance(template_part, list):
            # Recursively handle lists
            return [self._fill_placeholders(item, input_variables) for item in template_part]

        return template_part  # For non-string, non-dict, non-list types, return as is

    def _validate_input_variables(self, input_variables: Dict[str, Any]) -> None:
        """Validate that the provided input variables match the expected ones."""
        if hasattr(self, "input_variables"):
            missing_vars = set(self.input_variables) - set(input_variables.keys())
            extra_vars = set(input_variables.keys()) - set(self.input_variables)

            if missing_vars or extra_vars:
                error_msg = []
                error_msg.append(f"Expected input_variables from the prompt template: {self.input_variables}")
                if missing_vars:
                    error_msg.append(f"Missing variables: {list(missing_vars)}")
                if extra_vars:
                    error_msg.append(f"Unexpected variables: {list(extra_vars)}")
                error_msg.append(f"Provided variables: {list(input_variables.keys())}")
                if self.prompt_url:
                    error_msg.append(f"Template URL: {self.prompt_url}")

                raise ValueError("\n".join(error_msg))
        else:
            logger.warning("No input_variables specified in template. " "Input validation is disabled.")


class TextPromptTemplate(BasePromptTemplate):
    """A class representing a standard prompt template."""

    # Declare types for mypy because attributes are set dynamically via setattr in parent's __init__.
    # These declarations don't create the attributes, they just tell mypy about their types.
    template: str
    input_variables: Optional[List[str]]
    metadata: Optional[Dict[str, Any]]

    def __init__(self, full_yaml_content: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        if "template" not in kwargs:
            raise ValueError("You must always provide 'template' to TextPromptTemplate.")

        super().__init__(full_yaml_content=full_yaml_content, **kwargs)

    def populate_template(self, **input_variables: Any) -> PopulatedPrompt:
        """Populate the prompt by replacing placeholders with provided values.

        Args:
            **input_variables: The values to fill placeholders in the template.

        Returns:
            PopulatedPrompt: A PopulatedPrompt object containing the populated prompt string.
        """
        self._validate_input_variables(input_variables)
        populated_prompt = self._fill_placeholders(self.template, input_variables)
        return PopulatedPrompt(content=populated_prompt)

    def to_langchain_template(self) -> "LC_PromptTemplate":
        """Convert the TextPromptTemplate to a LangChain PromptTemplate.

        Returns:
            PromptTemplate: A LangChain PromptTemplate object.

        Raises:
            ImportError: If LangChain is not installed.
        """
        try:
            from langchain.prompts import PromptTemplate as LC_PromptTemplate
        except ImportError:
            raise ImportError(
                "LangChain is not installed. Please install it with 'pip install langchain' to use this feature."
            ) from None

        lc_prompt_template = LC_PromptTemplate(
            template=self.template,
            input_variables=self.input_variables if hasattr(self, "input_variables") else None,
            metadata=self.metadata if hasattr(self, "metadata") else None,
        )
        return lc_prompt_template


class ChatPromptTemplate(BasePromptTemplate):
    """A class representing a chat prompt template that can be formatted and used with various LLM clients."""

    # Declare types for mypy because attributes are set dynamically via setattr in parent's __init__.
    # These declarations don't create the attributes, they just tell mypy about their types.
    messages: List[Dict[str, Any]]
    input_variables: Optional[List[str]]
    metadata: Optional[Dict[str, Any]]

    def __init__(self, full_yaml_content: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        if "messages" not in kwargs:
            raise ValueError("You must always provide 'messages' to ChatPromptTemplate.")

        super().__init__(full_yaml_content=full_yaml_content, **kwargs)

    def populate_template(self, **input_variables: Any) -> PopulatedPrompt:
        """Populate the prompt messages by replacing placeholders with provided values.

        Args:
            **input_variables: The values to fill placeholders in the messages.

        Returns:
            PopulatedPrompt: A PopulatedPrompt object containing the populated messages.
        """
        self._validate_input_variables(input_variables)

        messages_populated = [
            {**msg, "content": self._fill_placeholders(msg["content"], input_variables)} for msg in self.messages
        ]
        return PopulatedPrompt(content=messages_populated)

    def create_messages(
        self, client: str = "openai", **input_variables: Any
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Convenience method to populate template and format for client in one step.

        Args:
            client (str): The client format to use ('openai', 'anthropic'). Defaults to 'openai'.
            **input_variables: The values to fill placeholders in the messages.

        Returns:
            Union[List[Dict[str, Any]], Dict[str, Any]]: Populated and formatted messages.
        """
        prompt = self.populate_template(**input_variables)
        return prompt.format_for_client(client)

    def to_langchain_template(self) -> "LC_ChatPromptTemplate":
        """Convert the ChatPromptTemplate to a LangChain ChatPromptTemplate.

        Returns:
            ChatPromptTemplate: A LangChain ChatPromptTemplate object.

        Raises:
            ImportError: If LangChain is not installed.
        """
        try:
            from langchain.prompts import ChatPromptTemplate as LC_ChatPromptTemplate
        except ImportError:
            raise ImportError(
                "LangChain is not installed. Please install it with 'pip install langchain' to use this feature."
            ) from None

        lc_chat_prompt_template = LC_ChatPromptTemplate(
            messages=[(msg["role"], msg["content"]) for msg in self.messages],
            input_variables=self.input_variables if hasattr(self, "input_variables") else None,
            metadata=self.metadata if hasattr(self, "metadata") else None,
        )
        return lc_chat_prompt_template
