import json
import logging
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Match, Optional, Set, Tuple, Union

import jinja2
import yaml
from jinja2 import Environment, meta

from .constants import Jinja2SecurityLevel, PopulatorType
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

    def __init__(
        self,
        template: Union[str, List[Dict[str, Any]]],
        input_variables: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        other_data: Optional[Dict[str, Any]] = None,
        populator: Optional[PopulatorType] = None,
        jinja2_security_level: Jinja2SecurityLevel = "standard",
    ) -> None:
        """Initialize a prompt template.

        Args:
            template: The template string or chat messages
            input_variables: List of variables required by the template
            metadata: Optional metadata about the prompt template
            other_data: Optional additional data
            populator: Optional template populator type
            jinja2_security_level: Security level for Jinja2 populator

        Raises:
            TypeError: If input types don't match expected types
            ValueError: If template format is invalid
        """
        # Type validation
        if input_variables is not None and not isinstance(input_variables, list):
            raise TypeError(f"input_variables must be a list, got {type(input_variables).__name__}")
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError(f"metadata must be a dict, got {type(metadata).__name__}")
        if other_data is not None and not isinstance(other_data, dict):
            raise TypeError(f"other_data must be a dict, got {type(other_data).__name__}")

        # Format validation
        self._validate_template_format(template)

        # Initialize attributes
        self.template = template
        self.input_variables = input_variables or []
        self.metadata = metadata or {}
        self.other_data = other_data or {}

        # set up the template populator
        self._set_up_populator(populator, jinja2_security_level)

        # Validate that variables provided in template and input_variables are equal
        if self.input_variables:
            self._validate_template_input_variables_equality()

    @abstractmethod
    def populate_template(self, **user_provided_variables: Any) -> PopulatedPrompt:
        """Abstract method to populate the prompt template with user-provided variables.

        Args:
            **user_provided_variables: The values to fill placeholders in the template.

        Returns:
            PopulatedPrompt: A PopulatedPrompt object containing the populated content.
        """
        pass

    def display(self, format: Literal["json", "yaml"] = "json") -> None:
        """Display the prompt configuration in the specified format.

        Examples:
            >>> from hf_hub_prompts import PromptTemplateLoader
            >>> prompt_template = PromptTemplateLoader.from_hub(
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

        # TODO: display Jinja2 template content properly

        if format == "json":
            print(json.dumps(display_dict, indent=2), end="")
        elif format == "yaml":
            print(yaml.dump(display_dict, default_flow_style=False, sort_keys=False), end="")

    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]

    def __repr__(self) -> str:
        attributes = ", ".join(
            f"{key}={repr(value)[:50]}..." if len(repr(value)) > 50 else f"{key}={repr(value)}"
            for key, value in self.__dict__.items()
        )
        return f"{self.__class__.__name__}({attributes})"

    def _populate_placeholders(self, template_part: Any, user_provided_variables: Dict[str, Any]) -> Any:
        """Recursively fill placeholders in strings or nested structures like dicts or lists."""
        if isinstance(template_part, str):
            # fill placeholders in strings
            return self.populator.populate(template_part, user_provided_variables)
        elif isinstance(template_part, dict):
            # Recursively handle dictionaries
            return {
                key: self._populate_placeholders(value, user_provided_variables)
                for key, value in template_part.items()
            }

        elif isinstance(template_part, list):
            # Recursively handle lists
            return [self._populate_placeholders(item, user_provided_variables) for item in template_part]

        return template_part  # For non-string, non-dict, non-list types, return as is

    def _validate_user_provided_variables(self, user_provided_variables: Dict[str, Any]) -> None:
        """Validate that all required variables are provided by the user.

        Args:
            user_provided_variables: Variables provided by user to populate template

        Raises:
            ValueError: If validation fails
        """
        # We know that template variables and input_variables are equal based on _validate_template_input_variables_equality, so we can validate against either
        required_variables = set(self.input_variables) if self.input_variables else self._get_template_variables()
        provided_variables = set(user_provided_variables.keys())

        # Check for missing and unexpected variables
        missing_vars = required_variables - provided_variables
        unexpected_vars = provided_variables - required_variables

        if missing_vars or unexpected_vars:
            error_parts = []

            if missing_vars:
                error_parts.append(
                    f"Missing required variables:\n"
                    f"  Required: {sorted(missing_vars)}\n"
                    f"  Provided: {sorted(provided_variables)}"
                )

            if unexpected_vars:
                error_parts.append(
                    f"Unexpected variables provided:\n"
                    f"  Expected required variables: {sorted(required_variables)}\n"
                    f"  Extra variables: {sorted(unexpected_vars)}"
                )

            if "prompt_url" in self.other_data:
                error_parts.append(f"\nTemplate URL: {self.other_data['prompt_url']}")

            raise ValueError("\n".join(error_parts))

    def _validate_template_input_variables_equality(self) -> None:
        """Validate that the declared input_variables and template variables are identical."""
        template_variables = self._get_template_variables()
        input_variables = set(self.input_variables or [])

        # Check for mismatches
        undeclared_template_vars = template_variables - input_variables
        unused_input_vars = input_variables - template_variables

        if undeclared_template_vars or unused_input_vars:
            error_parts = []

            if undeclared_template_vars:
                error_parts.append(
                    f"Template contains variables that are not declared in input_variables: {list(undeclared_template_vars)}"
                )
            if unused_input_vars:
                error_parts.append(
                    f"input_variables declares variables that are not used in template: {list(unused_input_vars)}"
                )

            template_extract = (
                str(self.template)[:100] + "..." if len(str(self.template)) > 100 else str(self.template)
            )
            error_parts.append(f"\nTemplate extract: {template_extract}")

            if "prompt_url" in self.other_data:
                error_parts.append(f"Template URL: {self.other_data['prompt_url']}")

            raise ValueError("\n".join(error_parts))

    def _get_template_variables(self) -> Set[str]:
        """Get all variables used as placeholders in the template string or messages dictionary.

        Returns:
            Set of variable names used as placeholders in the template
        """
        template_variables = set()
        if isinstance(self.template, str):
            template_variables = self.populator.get_variable_names(self.template)
        elif isinstance(self.template, list) and any(isinstance(item, dict) for item in self.template):
            for message in self.template:
                template_variables.update(self.populator.get_variable_names(message["content"]))
        return template_variables

    def _detect_double_brace_syntax(self) -> bool:
        """Detect if the template uses simple {{var}} syntax without Jinja2 features."""

        def contains_double_brace(text: str) -> bool:
            # Look for {{var}} pattern but exclude Jinja2-specific patterns
            basic_var = r"\{\{[^{}|.\[]+\}\}"  # Only match simple variables
            return bool(re.search(basic_var, text))

        if isinstance(self.template, str):
            return contains_double_brace(self.template)
        elif isinstance(self.template, list) and any(isinstance(item, dict) for item in self.template):
            return any(contains_double_brace(message["content"]) for message in self.template)
        return False

    def _detect_jinja2_syntax(self) -> bool:
        """Detect if the template uses Jinja2 syntax.

        Looks for Jinja2-specific patterns:
        - {% statement %}    - Control structures
        - {# comment #}     - Comments
        - {{ var|filter }}  - Filters
        - {{ var.attr }}    - Attribute access
        - {{ var['key'] }}  - Dictionary access
        """

        def contains_jinja2(text: str) -> bool:
            patterns = [
                r"{%\s*.*?\s*%}",  # Statements
                r"{#\s*.*?\s*#}",  # Comments
                r"{{\s*.*?\|.*?}}",  # Filters
                r"{{\s*.*?\..*?}}",  # Attribute access
                r"{{\s*.*?\[.*?\].*?}}",  # Dictionary access
            ]
            return any(re.search(pattern, text) for pattern in patterns)

        if isinstance(self.template, str):
            return contains_jinja2(self.template)
        elif isinstance(self.template, list) and any(isinstance(item, dict) for item in self.template):
            return any(contains_jinja2(message["content"]) for message in self.template)
        return False

    def _validate_template_format(self, template: Union[str, List[Dict[str, Any]]]) -> None:
        """Validate the format of the template at initialization."""
        if isinstance(template, list):
            if not all(isinstance(msg, dict) for msg in template):
                raise ValueError("All messages in template must be dictionaries")

            required_keys = {"role", "content"}
            for msg in template:
                missing_keys = required_keys - set(msg.keys())
                if missing_keys:
                    raise ValueError(
                        f"Each message must have a 'role' and a 'content' key. Missing keys: {missing_keys}"
                    )

                if not isinstance(msg["role"], str) or not isinstance(msg["content"], str):
                    raise ValueError("Message 'role' and 'content' must be strings")

                if msg["role"] not in {"system", "user", "assistant"}:
                    raise ValueError(f"Invalid role '{msg['role']}'. Must be one of: system, user, assistant")

    def _set_up_populator(
        self, populator: Optional[PopulatorType], jinja2_security_level: Jinja2SecurityLevel
    ) -> None:
        """Set up the template populator based on specified type or auto-detection.

        Args:
            populator: Optional explicit populator type ('jinja2', 'double_brace', 'single_brace')
            jinja2_security_level: Security level for Jinja2 populator

        Raises:
            ValueError: If an unknown populator type is specified
        """
        self.populator_type: PopulatorType
        self.populator: TemplatePopulator

        # Check and validate populator
        if populator is not None:
            # Use explicitly specified populator
            if populator == "jinja2":
                self.populator_type = "jinja2"
                self.populator = Jinja2TemplatePopulator(security_level=jinja2_security_level)
            elif populator == "double_brace":
                self.populator_type = "double_brace"
                self.populator = DoubleBracePopulator()
            elif populator == "single_brace":
                self.populator_type = "single_brace"
                self.populator = SingleBracePopulator()
            else:
                raise ValueError(
                    f"Unknown populator type: {populator}. Valid options are: double_brace, single_brace, jinja2"
                )
        else:
            # Auto-detect populator
            if self._detect_jinja2_syntax():
                self.populator_type = "jinja2"
                self.populator = Jinja2TemplatePopulator(security_level=jinja2_security_level)
            elif self._detect_double_brace_syntax():
                self.populator_type = "double_brace"
                self.populator = DoubleBracePopulator()
            else:
                self.populator_type = "single_brace"
                self.populator = SingleBracePopulator()


class TextPromptTemplate(BasePromptTemplate):
    """A class representing a standard text prompt template.

    Examples:
        Instantiate a text prompt template:
        >>> from hf_hub_prompts import TextPromptTemplate
        >>> template_text = "Translate the following text to {{language}}:\n{{text}}"
        >>> input_variables = ["language", "text"]
        >>> metadata = {
        ...     "name": "Simple Translator",
        ...     "description": "A simple translation prompt for illustrating the standard prompt YAML format",
        ...     "tags": ["translation", "multilinguality"],
        ...     "version": "0.0.1",
        ...     "author": "Some Person"
        }
        >>> template = TextPromptTemplate(
        ...     template=template_text,
        ...     input_variables=input_variables,
        ...     metadata=metadata
        ... )
        >>> print(template)
        TextPromptTemplate(template='Translate the following text to {{language}}:\n{{text}}', input_variables=['language', 'text'], metadata={'name': 'Simple Translator', 'description': 'A simple translation prompt for illustrating the standard prompt YAML format', 'tags': ['translation', 'multilinguality'], 'version': '0.0.1', 'author': 'Some Person'}, other_data={}, populator_type='double_brace', populator=<hf_hub_prompts.prompt_templates.DoubleBracePopulator object at 0x...>)
        >>> # Inspect template attributes
        >>> template.template
        'Translate the following text to {language}:\\n{text}'
        >>> template.input_variables
        ['language', 'text']
        >>> template.metadata['name']
        'Simple Translator'
        >>> # Populate the template
        >>> prompt = prompt_template.populate_template(
        ...     language="French",
        ...     text="Hello world!"
        ... )
        >>> print(prompt)
        'Translate the following text to French:\\nHello world!'

        Or download the same text prompt template from the Hub:
        >>> from hf_hub_prompts import PromptTemplateLoader
        >>> template = PromptTemplateLoader.from_hub(
        ...     repo_id="MoritzLaurer/example_prompts",
        ...     filename="translate.yaml"
        ... )
    """

    def __init__(
        self,
        template: str,
        input_variables: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        other_data: Optional[Dict[str, Any]] = None,
        populator: Optional[PopulatorType] = None,
        jinja2_security_level: Jinja2SecurityLevel = "standard",
    ) -> None:
        super().__init__(
            template=template,
            input_variables=input_variables,
            metadata=metadata,
            other_data=other_data,
            populator=populator,
            jinja2_security_level=jinja2_security_level,
        )

    def populate_template(self, **user_provided_variables: Any) -> PopulatedPrompt:
        """Populate the prompt by replacing placeholders with provided values.

        Examples:
            >>> from hf_hub_prompts import PromptTemplateLoader
            >>> template = PromptTemplateLoader.from_hub(
            ...     repo_id="MoritzLaurer/example_prompts",
            ...     filename="translate.yaml"
            ... )
            >>> template.template
            'Translate the following text to {language}:\\n{text}'
            >>> prompt = template.populate_template(
            ...     language="French",
            ...     text="Hello world!"
            ... )
            >>> print(prompt)
            'Translate the following text to French:\\nHello world!'

        Args:
            **user_provided_variables: The values to fill placeholders in the prompt template.

        Returns:
            PopulatedPrompt: A PopulatedPrompt object containing the populated prompt string.
        """
        self._validate_user_provided_variables(user_provided_variables)
        populated_prompt = self._populate_placeholders(self.template, user_provided_variables)
        return PopulatedPrompt(content=populated_prompt)

    def to_langchain_template(self) -> "LC_PromptTemplate":
        """Convert the TextPromptTemplate to a LangChain PromptTemplate.

        Examples:
            >>> from hf_hub_prompts import PromptTemplateLoader
            >>> template = PromptTemplateLoader.from_hub(
            ...     repo_id="MoritzLaurer/example_prompts",
            ...     filename="translate.yaml"
            ... )
            >>> lc_template = template.to_langchain_template()
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
        Instantiate a chat prompt template:
        >>> from hf_hub_prompts import ChatPromptTemplate
        >>> template_messages = [
        ...     {"role": "system", "content": "You are a coding assistant who explains concepts clearly and provides short examples."},
        ...     {"role": "user", "content": "Explain what {{concept}} is in {{programming_language}}."}
        ... ]
        >>> input_variables = ["concept", "programming_language"]
        >>> metadata = {
        ...     "name": "Code Teacher",
        ...     "description": "A simple chat prompt for explaining programming concepts with examples",
        ...     "tags": ["programming", "education"],
        ...     "version": "0.0.1",
        ...     "author": "My Awesome Company"
        ... }
        >>> template = ChatPromptTemplate(
        ...     template=template_messages,
        ...     input_variables=input_variables,
        ...     metadata=metadata
        ... )
        >>> print(template)
        ChatPromptTemplate(template=[{'role': 'system', 'content': 'You are a coding a..., input_variables=['concept', 'programming_language'], metadata={'name': 'Code Teacher', 'description': 'A simple ..., other_data={}, populator_type='double_brace', populator=<hf_hub_prompts.prompt_templates.DoubleBracePopula...)
        >>> # Inspect template attributes
        >>> template.template
        [{'role': 'system', 'content': 'You are a coding assistant who explains concepts clearly and provides short examples.'}, {'role': 'user', 'content': 'Explain what {concept} is in {programming_language}.'}]
        >>> template.input_variables
        ['concept', 'programming_language']

        >>> # Populate the template
        >>> messages = template.populate_template(
        ...     concept="list comprehension",
        ...     programming_language="Python"
        ... )
        >>> print(messages)
        [{'role': 'system', 'content': 'You are a coding assistant who explains concepts clearly and provides short examples.'}, {'role': 'user', 'content': 'Explain what list comprehension is in Python.'}]

        >>> # By default, the populated prompt is in the OpenAI messages format, as it is adopted by many open-source libraries
        >>> # You can convert to formats used by other LLM clients like Anthropic like this:
        >>> messages_anthropic = prompt.format_for_client("anthropic")
        >>> print(messages_anthropic)
        {'system': 'You are a coding assistant who explains concepts clearly and provides short examples.', 'messages': [{'role': 'user', 'content': 'Explain what list comprehension is in Python.'}]}

        >>> # Convenience method to populate and format in one step for clients that do not use the OpenAI messages format
        >>> messages_anthropic = prompt_template.create_messages(
        ...     client="anthropic",
        ...     concept="list comprehension",
        ...     programming_language="Python"
        ... )
        >>> print(messages_anthropic)
        {'system': 'You are a coding assistant who explains concepts clearly and provides short examples.', 'messages': [{'role': 'user', 'content': 'Explain what list comprehension is in Python.'}]}

        Or download the same chat prompt template from the Hub:
        >>> from hf_hub_prompts import PromptTemplateLoader
        >>> template = PromptTemplateLoader.from_hub(
        ...     repo_id="MoritzLaurer/example_prompts",
        ...     filename="code_teacher.yaml"
        ... )
    """

    template: List[Dict[str, str]]

    def __init__(
        self,
        template: List[Dict[str, Any]],
        input_variables: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        other_data: Optional[Dict[str, Any]] = None,
        populator: Optional[PopulatorType] = None,
        jinja2_security_level: Jinja2SecurityLevel = "standard",
    ) -> None:
        super().__init__(
            template=template,
            input_variables=input_variables,
            metadata=metadata,
            other_data=other_data,
            populator=populator,
            jinja2_security_level=jinja2_security_level,
        )

    def populate_template(self, **user_provided_variables: Any) -> PopulatedPrompt:
        """Populate the prompt template messages by replacing placeholders with provided values.

        Examples:
            >>> from hf_hub_prompts import PromptTemplateLoader
            >>> template = PromptTemplateLoader.from_hub(
            ...     repo_id="MoritzLaurer/example_prompts",
            ...     filename="code_teacher.yaml"
            ... )
            >>> messages = template.populate_template(
            ...     concept="list comprehension",
            ...     programming_language="Python"
            ... )
            >>> print(messages)
            [{'role': 'system', 'content': 'You are a coding assistant who explains concepts clearly and provides short examples.'}, {'role': 'user', 'content': 'Explain what list comprehension is in Python.'}]

        Args:
            **user_provided_variables: The values to fill placeholders in the messages template.

        Returns:
            PopulatedPrompt: A PopulatedPrompt object containing the populated messages prompt.
        """
        self._validate_user_provided_variables(user_provided_variables)

        messages_template_populated: List[Dict[str, str]] = [
            {
                "role": str(message["role"]),
                "content": self._populate_placeholders(message["content"], user_provided_variables),
            }
            for message in self.template
        ]
        return PopulatedPrompt(content=messages_template_populated)

    def create_messages(self, client: str = "openai", **user_provided_variables: Any) -> PopulatedPrompt:
        """Convenience method that populates a prompt template and formats it for a client in one step.
        This method is only useful if your a client that does not use the OpenAI messages format, because
        populating a ChatPromptTemplate converts it into the OpenAI messages format by default.

        Examples:
            >>> from hf_hub_prompts import PromptTemplateLoader
            >>> template = PromptTemplateLoader.from_hub(
            ...     repo_id="MoritzLaurer/example_prompts",
            ...     filename="code_teacher.yaml"
            ... )
            >>> # Format for OpenAI (default)
            >>> messages = template.create_messages(
            ...     concept="list comprehension",
            ...     programming_language="Python"
            ... )
            >>> print(messages)
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
            **user_provided_variables: The variables to fill into the prompt template. For example, if your template
                expects variables like 'name' and 'age', pass them as keyword arguments.

        Returns:
            PopulatedPrompt: A populated prompt formatted for the specified client.
        """
        if "client" in user_provided_variables:
            logger.warning(
                f"'client' was passed both as a parameter for the LLM inference client ('{client}') and in user_provided_variables "
                f"('{user_provided_variables['client']}'). The first parameter version will be used for formatting, "
                "while the second user_provided_variable version will be used in template population."
            )

        prompt = self.populate_template(**user_provided_variables)
        return prompt.format_for_client(client)

    def to_langchain_template(self) -> "LC_ChatPromptTemplate":
        """Convert the ChatPromptTemplate to a LangChain ChatPromptTemplate.

        Examples:
            >>> from hf_hub_prompts import PromptTemplateLoader
            >>> template = PromptTemplateLoader.from_hub(
            ...     repo_id="MoritzLaurer/example_prompts",
            ...     filename="code_teacher.yaml"
            ... )
            >>> lc_template = template.to_langchain_template()
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

        # LangChain expects a list of tuples of the form (role, content)
        messages: List[Tuple[str, str]] = [
            (str(message["role"]), str(message["content"])) for message in self.template
        ]
        return LC_ChatPromptTemplate(
            messages=messages,
            input_variables=self.input_variables,
            metadata=self.metadata,
        )


class TemplatePopulator(ABC):
    """Abstract base class for template populating strategies."""

    @abstractmethod
    def populate(self, template_str: str, user_provided_variables: Dict[str, Any]) -> str:
        """Populate the template with given user_provided_variables."""
        pass

    @abstractmethod
    def get_variable_names(self, template_str: str) -> Set[str]:
        """Extract variable names from template."""
        pass


class SingleBracePopulator(TemplatePopulator):
    """Template populator using regex for basic {var} substitution."""

    def populate(self, template_str: str, user_provided_variables: Dict[str, Any]) -> str:
        pattern = re.compile(r"\{([^{}]+)\}")

        def replacer(match: Match[str]) -> str:
            key = match.group(1).strip()
            if key not in user_provided_variables:
                raise ValueError(f"Variable '{key}' not found in provided variables")
            return str(user_provided_variables[key])

        return pattern.sub(replacer, template_str)

    def get_variable_names(self, template_str: str) -> Set[str]:
        pattern = re.compile(r"\{([^{}]+)\}")
        return {match.group(1).strip() for match in pattern.finditer(template_str)}


class DoubleBracePopulator(TemplatePopulator):
    """Template populator using regex for {{var}} substitution."""

    def populate(self, template_str: str, user_provided_variables: Dict[str, Any]) -> str:
        pattern = re.compile(r"\{\{([^{}]+)\}\}")

        def replacer(match: Match[str]) -> str:
            key = match.group(1).strip()
            if key not in user_provided_variables:
                raise ValueError(f"Variable '{key}' not found in provided variables")
            return str(user_provided_variables[key])

        return pattern.sub(replacer, template_str)

    def get_variable_names(self, template_str: str) -> Set[str]:
        pattern = re.compile(r"\{\{([^{}]+)\}\}")
        return {match.group(1).strip() for match in pattern.finditer(template_str)}


class Jinja2TemplatePopulator(TemplatePopulator):
    """Jinja2 template populator with configurable security levels.

    Security Levels:
        - strict: Minimal set of features, highest security
            Filters: lower, upper, title, safe
            Tests: defined, undefined, none
            Env: autoescape=True, no caching, no globals, no auto-reload
        - standard (default): Balanced set of features
            Filters: lower, upper, title, capitalize, trim, strip, replace, safe,
                    int, float, join, split, length
            Tests: defined, undefined, none, number, string, sequence
            Env: autoescape=True, limited caching, basic globals, no auto-reload
        - relaxed: Default Jinja2 behavior (use with trusted templates only)
            All default Jinja2 features enabled
            Env: autoescape=False, full caching, all globals, auto-reload allowed

    Args:
        security_level: Level of security restrictions ("strict", "standard", "relaxed")
    """

    def __init__(self, security_level: Jinja2SecurityLevel = "standard"):
        # Store security level for error messages
        self.security_level = security_level

        if security_level == "strict":
            # Most restrictive settings
            self.env = Environment(
                undefined=jinja2.StrictUndefined,
                trim_blocks=True,
                lstrip_blocks=True,
                autoescape=True,  # Force autoescaping
                cache_size=0,  # Disable caching
                auto_reload=False,  # Disable auto reload
            )
            # Remove all globals
            self.env.globals.clear()

            # Minimal set of features
            safe_filters = {"lower", "upper", "title", "safe"}
            safe_tests = {"defined", "undefined", "none"}

        elif security_level == "standard":
            # Balanced settings
            self.env = Environment(
                undefined=jinja2.StrictUndefined,
                trim_blocks=True,
                lstrip_blocks=True,
                autoescape=True,  # Keep autoescaping
                cache_size=100,  # Limited cache
                auto_reload=False,  # Still no auto reload
            )
            # Allow some safe globals
            self.env.globals.update(
                {
                    "range": range,  # Useful for iterations
                    "dict": dict,  # Basic dict operations
                    "len": len,  # Length calculations
                }
            )

            # Balanced set of features
            safe_filters = {
                "lower",
                "upper",
                "title",
                "capitalize",
                "trim",
                "strip",
                "replace",
                "safe",
                "int",
                "float",
                "join",
                "split",
                "length",
            }
            safe_tests = {"defined", "undefined", "none", "number", "string", "sequence"}

        else:  # relaxed
            # Default Jinja2 behavior
            self.env = Environment(
                undefined=jinja2.StrictUndefined,
                trim_blocks=True,
                lstrip_blocks=True,
                autoescape=False,  # Default Jinja2 behavior
                cache_size=400,  # Default cache size
                auto_reload=True,  # Allow auto reload
            )
            # Keep all default globals and features
            return

        # Apply security settings for strict and standard modes
        self._apply_security_settings(safe_filters, safe_tests)

    def _apply_security_settings(self, safe_filters: Set[str], safe_tests: Set[str]) -> None:
        """Apply security settings by removing unsafe filters and tests."""
        # Remove unsafe filters
        unsafe_filters = set(self.env.filters.keys()) - safe_filters
        for unsafe in unsafe_filters:
            self.env.filters.pop(unsafe, None)

        # Remove unsafe tests
        unsafe_tests = set(self.env.tests.keys()) - safe_tests
        for unsafe in unsafe_tests:
            self.env.tests.pop(unsafe, None)

    def populate(self, template_str: str, user_provided_variables: Dict[str, Any]) -> str:
        """Populate the template with given user_provided_variables."""
        try:
            template = self.env.from_string(template_str)
            populated = template.render(**user_provided_variables)
            # Ensure we return a string for mypy
            return str(populated)
        except jinja2.TemplateSyntaxError as e:
            raise ValueError(
                f"Invalid template syntax at line {e.lineno}: {str(e)}\n" f"Security level: {self.security_level}"
            ) from e
        except jinja2.UndefinedError as e:
            raise ValueError(
                f"Undefined variable in template: {str(e)}\n" "Make sure all required variables are provided"
            ) from e
        except Exception as e:
            raise ValueError(f"Error populating template: {str(e)}") from e

    def get_variable_names(self, template_str: str) -> Set[str]:
        """Extract variable names from template."""
        try:
            ast = self.env.parse(template_str)
            variables = meta.find_undeclared_variables(ast)
            # Ensure we return a set of strings for mypy
            return {str(var) for var in variables}
        except jinja2.TemplateSyntaxError as e:
            raise ValueError(f"Invalid template syntax: {str(e)}") from e
