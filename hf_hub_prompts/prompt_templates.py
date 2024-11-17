import json
import logging
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Match, Optional, Union

import yaml

from .populated_prompt import PopulatedPrompt


if TYPE_CHECKING:
    from langchain_core.prompts import (
        ChatPromptTemplate as LC_ChatPromptTemplate,
    )
    from langchain_core.prompts import (
        PromptTemplate as LC_PromptTemplate,
    )

logger = logging.getLogger(__name__)


class BasePromptTemplate(ABC):
    """An abstract base class for prompt templates.

    This class defines the common interface and shared functionality for all prompt templates.
    Users should not instantiate this class directly, but instead use TextPromptTemplate
    or ChatPromptTemplate, which are subclasses of BasePromptTemplate.
    """

    # Type hints for optional standard attributes shared across all template types
    metadata: Optional[Dict[str, Any]]
    input_variables: Optional[List[str]]
    other_data: Dict[str, Any]

    def __init__(self, prompt_data: Dict[str, Any], prompt_url: Optional[str] = None) -> None:
        # Set template-specific required attributes
        self._set_required_attributes_for_template_type(prompt_data)

        # Set optional standard attributes that are the same across all templates
        self.input_variables = prompt_data.get("input_variables")
        self.metadata = prompt_data.get("metadata")

        # Store any additional optional data that might be present in the prompt data
        self.other_data = {
            k: v
            for k, v in prompt_data.items()
            if k not in ["metadata", "input_variables"] + self._get_required_attributes_for_template_type()
        }

        if prompt_url is not None:
            self.other_data["prompt_url"] = prompt_url

    @abstractmethod
    def _get_required_attributes_for_template_type(self) -> List[str]:
        """Return list of required keys for this template type."""
        pass

    @abstractmethod
    def _set_required_attributes_for_template_type(self, prompt_data: Dict[str, Any]) -> None:
        """Set required attributes for this template type."""
        pass

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
        """Display the prompt configuration in the specified format.

        Examples:
            >>> from hf_hub_prompts import download_prompt_template
            >>> prompt_template = download_prompt_template(
            ...     repo_id="MoritzLaurer/example_prompts",
            ...     filename="translate.yaml"
            ... )
            >>> prompt_template.display(format="yaml")  # doctest: +NORMALIZE_WHITESPACE
            template: 'Translate the following text to {language}:
              {text}'
            input_variables:
            - language
            - text
            metadata:
              name: Simple Translator
              description: A simple translation prompt for illustrating the standard prompt YAML
                format
              tags:
              - translation
              - multilinguality
              version: 0.0.1
              author: Some Person
        """
        # Create a dict of all attributes except other_data
        display_dict = self.__dict__.copy()
        display_dict.pop("other_data", None)

        if format == "json":
            print(json.dumps(display_dict, indent=2), end="")
        elif format == "yaml":
            print(yaml.dump(display_dict, default_flow_style=False, sort_keys=False), end="")

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
        """Validate that the provided input variables match the expected ones.

        Args:
            input_variables: Dictionary of variables to validate.

        Behavior:
            - If prompt_template.input_variables is defined:
                Ensures exact match between provided and expected variables.
            - If prompt_template.input_variables is not defined:
                Skips validation (logs warning) and accepts any variables.

        Raises:
            ValueError: If prompt_template.input_variables is defined and there are
                missing or unexpected variables.
        """
        if self.input_variables:
            missing_vars = set(self.input_variables) - set(input_variables.keys())
            extra_vars = set(input_variables.keys()) - set(self.input_variables)

            if missing_vars or extra_vars:
                error_msg = []
                error_msg.append(f"Expected input_variables from the prompt template: {self.input_variables}")
                if missing_vars:
                    error_msg.append(f"Missing variables: {list(missing_vars)}")
                if extra_vars:
                    error_msg.append(f"Unexpected variables: {list(extra_vars)}")
                if self.other_data["prompt_url"]:
                    error_msg.append(f"Template URL: {self.other_data["prompt_url"]}")

                raise ValueError("\n".join(error_msg))
        else:
            logger.warning(
                "No input_variables specified in prompt template. Input validation is disabled. "
                "To enable validation, specify input_variables when creating the prompt template. "
                "Without validation, misspelled variable names will be left unreplaced in the output."
            )


class TextPromptTemplate(BasePromptTemplate):
    """A class representing a standard text prompt template.

    Examples:
        Download and use a text prompt template:
        >>> from hf_hub_prompts import download_prompt_template
        >>> # Download example translation prompt
        >>> prompt_template = download_prompt_template(
        ...     repo_id="MoritzLaurer/example_prompts",
        ...     filename="translate.yaml"
        ... )
        >>> # Inspect template attributes
        >>> prompt_template.template
        'Translate the following text to {language}:\\n{text}'
        >>> prompt_template.input_variables
        ['language', 'text']
        >>> prompt_template.metadata['name']
        'Simple Translator'

        >>> # Use the template
        >>> prompt = prompt_template.populate_template(
        ...     language="French",
        ...     text="Hello world!"
        ... )
        >>> prompt.content
        'Translate the following text to French:\\nHello world!'
    """

    # Type hints for template-specific attributes
    template: str

    def _get_required_attributes_for_template_type(self) -> List[str]:
        return ["template"]

    def _set_required_attributes_for_template_type(self, prompt_data: Dict[str, Any]) -> None:
        if "template" not in prompt_data:
            raise ValueError("You must provide 'template' in prompt_data")
        self.template = prompt_data["template"]

    def populate_template(self, **input_variables: Any) -> PopulatedPrompt:
        """Populate the prompt by replacing placeholders with provided values.

        Examples:
            >>> from hf_hub_prompts import download_prompt_template
            >>> prompt_template = download_prompt_template(
            ...     repo_id="MoritzLaurer/example_prompts",
            ...     filename="translate.yaml"
            ... )
            >>> prompt_template.template
            'Translate the following text to {language}:\\n{text}'
            >>> prompt = prompt_template.populate_template(
            ...     language="French",
            ...     text="Hello world!"
            ... )
            >>> prompt.content
            'Translate the following text to French:\\nHello world!'

        Args:
            **input_variables: The values to fill placeholders in the prompt template.

        Returns:
            PopulatedPrompt: A PopulatedPrompt object containing the populated prompt string.
        """
        self._validate_input_variables(input_variables)
        populated_prompt = self._fill_placeholders(self.template, input_variables)
        return PopulatedPrompt(content=populated_prompt)

    def to_langchain_template(self) -> "LC_PromptTemplate":
        """Convert the TextPromptTemplate to a LangChain PromptTemplate.

        Examples:
            >>> from hf_hub_prompts import download_prompt_template
            >>> prompt_template = download_prompt_template(
            ...     repo_id="MoritzLaurer/example_prompts",
            ...     filename="translate.yaml"
            ... )
            >>> lc_template = prompt_template.to_langchain_template()
            >>> # test equivalence
            >>> from langchain_core.prompts import PromptTemplate as LC_PromptTemplate
            >>> isinstance(lc_template, LC_PromptTemplate)
            True

        Returns:
            PromptTemplate: A LangChain PromptTemplate object.

        Raises:
            ImportError: If LangChain is not installed.
        """
        try:
            from langchain_core.prompts import PromptTemplate as LC_PromptTemplate
        except ImportError as e:
            raise ImportError("LangChain is not installed. Please install it with 'pip install langchain'") from e

        return LC_PromptTemplate(
            template=self.template,
            input_variables=self.input_variables,
            metadata=self.metadata,
        )


class ChatPromptTemplate(BasePromptTemplate):
    """A class representing a chat prompt template that can be formatted for and used with various LLM clients.

    Examples:
        Download and use a chat prompt template:
        >>> from hf_hub_prompts import download_prompt_template
        >>> # Download example code teaching prompt
        >>> prompt_template = download_prompt_template(
        ...     repo_id="MoritzLaurer/example_prompts",
        ...     filename="code_teacher.yaml"
        ... )
        >>> # Inspect template attributes
        >>> prompt_template.messages
        [{'role': 'system', 'content': 'You are a coding assistant who explains concepts clearly and provides short examples.'}, {'role': 'user', 'content': 'Explain what {concept} is in {programming_language}.'}]
        >>> prompt_template.input_variables
        ['concept', 'programming_language']

        >>> # Populate the template
        >>> prompt = prompt_template.populate_template(
        ...     concept="list comprehension",
        ...     programming_language="Python"
        ... )
        >>> prompt.content
        [{'role': 'system', 'content': 'You are a coding assistant who explains concepts clearly and provides short examples.'}, {'role': 'user', 'content': 'Explain what list comprehension is in Python.'}]

        >>> # By default, the populated prompt is in the OpenAI messages format, as it is adopted by many open-source libraries
        >>> # You can convert to formats used by other LLM clients like Anthropic like this:
        >>> messages_anthropic = prompt.format_for_client("anthropic")
        >>> messages_anthropic
        {'system': 'You are a coding assistant who explains concepts clearly and provides short examples.', 'messages': [{'role': 'user', 'content': 'Explain what list comprehension is in Python.'}]}

        >>> # Convenience method to populate and format in one step
        >>> messages = prompt_template.create_messages(
        ...     client="anthropic",
        ...     concept="list comprehension",
        ...     programming_language="Python"
        ... )
        >>> messages
        {'system': 'You are a coding assistant who explains concepts clearly and provides short examples.', 'messages': [{'role': 'user', 'content': 'Explain what list comprehension is in Python.'}]}
    """

    # Type hints for template-specific attributes
    messages: List[Dict[str, Any]]

    def _get_required_attributes_for_template_type(self) -> List[str]:
        return ["messages"]

    def _set_required_attributes_for_template_type(self, prompt_data: Dict[str, Any]) -> None:
        if "messages" not in prompt_data:
            raise ValueError("You must provide 'messages' in prompt_data")
        self.messages = prompt_data["messages"]

    def populate_template(self, **input_variables: Any) -> PopulatedPrompt:
        """Populate the prompt messages by replacing placeholders with provided values.

        Examples:
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
        """Convenience method to populate a prompt template and format for client in one step.

        Examples:
            >>> from hf_hub_prompts import download_prompt_template
            >>> prompt_template = download_prompt_template(
            ...     repo_id="MoritzLaurer/example_prompts",
            ...     filename="code_teacher.yaml"
            ... )
            >>> # Format for OpenAI (default)
            >>> messages = prompt_template.create_messages(
            ...     concept="list comprehension",
            ...     programming_language="Python"
            ... )
            >>> messages
            [{'role': 'system', 'content': 'You are a coding assistant who explains concepts clearly and provides short examples.'}, {'role': 'user', 'content': 'Explain what list comprehension is in Python.'}]

            >>> # Format for Anthropic
            >>> messages = prompt_template.create_messages(
            ...     client="anthropic",
            ...     concept="list comprehension",
            ...     programming_language="Python"
            ... )
            >>> messages
            {'system': 'You are a coding assistant who explains concepts clearly and provides short examples.', 'messages': [{'role': 'user', 'content': 'Explain what list comprehension is in Python.'}]}

        Args:
            client (str): The client format to use ('openai', 'anthropic'). Defaults to 'openai'.
            **input_variables: The variables to fill into the prompt template. For example, if your template
                expects variables like 'name' and 'age', pass them as keyword arguments:

        Returns:
            Union[List[Dict[str, Any]], Dict[str, Any]]: Populated and formatted messages.
        """
        if "client" in input_variables:
            logger.warning(
                f"'client' was passed both as a parameter for the LLM inference client ('{client}') and in input_variables "
                f"('{input_variables['client']}'). The first parameter version will be used for formatting, "
                "while the second input_variables version will be used in template population."
            )

        prompt = self.populate_template(**input_variables)
        return prompt.format_for_client(client)

    def to_langchain_template(self) -> "LC_ChatPromptTemplate":
        """Convert the ChatPromptTemplate to a LangChain ChatPromptTemplate.

        Examples:
            >>> from hf_hub_prompts import download_prompt_template
            >>> prompt_template = download_prompt_template(
            ...     repo_id="MoritzLaurer/example_prompts",
            ...     filename="code_teacher.yaml"
            ... )
            >>> lc_template = prompt_template.to_langchain_template()
            >>> # test equivalence
            >>> from langchain_core.prompts import ChatPromptTemplate as LC_ChatPromptTemplate
            >>> isinstance(lc_template, LC_ChatPromptTemplate)
            True

        Returns:
            ChatPromptTemplate: A LangChain ChatPromptTemplate object.

        Raises:
            ImportError: If LangChain is not installed.
        """
        try:
            from langchain_core.prompts import ChatPromptTemplate as LC_ChatPromptTemplate
        except ImportError as e:
            raise ImportError("LangChain is not installed. Please install it with 'pip install langchain'") from e

        return LC_ChatPromptTemplate(
            messages=[(msg["role"], msg["content"]) for msg in self.messages],
            input_variables=self.input_variables,
            metadata=self.metadata,
        )
