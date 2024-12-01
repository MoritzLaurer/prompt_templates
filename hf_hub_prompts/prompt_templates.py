import json
import logging
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Match, Optional, Set, Union

import jinja2
import yaml
from jinja2 import Environment, meta

from .constants import Jinja2SecurityLevel, RendererType
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

    def __init__(
        self,
        prompt_data: Dict[str, Any],
        prompt_url: Optional[str] = None,
        renderer: Optional[RendererType] = None,
        jinja2_security_level: Jinja2SecurityLevel = "standard",
    ) -> None:
        # Track which renderer is being used
        self.renderer_type: RendererType
        self.renderer: TemplateRenderer

        if renderer is not None:
            # Use explicitly specified renderer
            if renderer == "jinja2":
                self.renderer = Jinja2TemplateRenderer(security_level=jinja2_security_level)
                self.renderer_type = "jinja2"
            elif renderer == "double_brace":
                self.renderer = DoubleBraceRenderer()
                self.renderer_type = "double_brace"
            elif renderer == "single_brace":
                self.renderer = SingleBraceRenderer()
                self.renderer_type = "single_brace"
            else:
                raise ValueError(
                    f"Unknown renderer type: {renderer}. Valid options are: double_brace, single_brace, jinja2"
                )
        else:
            # Auto-detect renderer
            if self._detect_jinja2_syntax(prompt_data):
                self.renderer = Jinja2TemplateRenderer(security_level=jinja2_security_level)
                self.renderer_type = "jinja2"
            elif self._detect_double_brace_syntax(prompt_data):
                self.renderer = DoubleBraceRenderer()
                self.renderer_type = "double_brace"
            else:
                self.renderer = SingleBraceRenderer()
                self.renderer_type = "single_brace"

        # Set template-specific required attributes
        self._set_required_attributes_for_template_type(prompt_data)

        # Set optional standard attributes that are the same across all templates
        self.input_variables = prompt_data.get("input_variables")
        self.metadata = prompt_data.get("metadata")

        # Validate alignment between template variables and input_variables
        if self.input_variables:
            self._validate_template_input_variables_alignment()

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

    def _fill_placeholders(self, template_part: Any, user_provided_variables: Dict[str, Any]) -> Any:
        """Recursively fill placeholders in strings or nested structures like dicts or lists."""
        if isinstance(template_part, str):
            # fill placeholders in strings
            return self.renderer.render(template_part, user_provided_variables)
        elif isinstance(template_part, dict):
            # Recursively handle dictionaries
            return {
                key: self._fill_placeholders(value, user_provided_variables) for key, value in template_part.items()
            }

        elif isinstance(template_part, list):
            # Recursively handle lists
            return [self._fill_placeholders(item, user_provided_variables) for item in template_part]

        return template_part  # For non-string, non-dict, non-list types, return as is

    def _validate_user_provided_variables(self, user_provided_variables: Dict[str, Any]) -> None:
        """Validate that all required variables are provided by the user.

        Args:
            user_provided_variables: Variables provided by user to populate template

        Raises:
            ValueError: If validation fails
        """
        # Since template_variables and input_variables are already aligned based on _validate_template_input_variables_alignment, we can validate against either
        required_variables = set(self.input_variables) if self.input_variables else self._get_template_variables()

        # Validate that user provided all required variables
        missing_vars = required_variables - set(user_provided_variables.keys())
        unexpected_vars = set(user_provided_variables.keys()) - required_variables

        if missing_vars or unexpected_vars:
            error_msg = []
            if missing_vars:
                error_msg.append(f"Missing required variables to fully populate the template: {list(missing_vars)}")
            if unexpected_vars:
                error_msg.append(f"Unexpected variables that are not used in the template: {list(unexpected_vars)}")
            if "prompt_url" in self.other_data:
                error_msg.append(f"Template URL: {self.other_data['prompt_url']}")
            raise ValueError("\n".join(error_msg))

    def _validate_template_input_variables_alignment(self) -> None:
        """Validate that declared input_variables match variables found in the template string.

        Raises:
            ValueError: If there's a mismatch between declared input_variables and template variables
        """
        # Get variables found in template
        template_variables = self._get_template_variables()

        # get input_variables (and handle None case for type checking)
        input_variables = self.input_variables if self.input_variables is not None else []

        # Check for mismatches between declared input_variables and template variables
        undeclared_template_vars = template_variables - set(input_variables)
        unused_input_vars = set(input_variables) - template_variables

        if undeclared_template_vars or unused_input_vars:
            error_msg = []
            if undeclared_template_vars:
                error_msg.append(
                    f"Template uses variables that are not declared in input_variables: {list(undeclared_template_vars)}"
                )
            if unused_input_vars:
                error_msg.append(
                    f"input_variables declares variables that are not used in template: {list(unused_input_vars)}"
                )
            if "prompt_url" in self.other_data:
                error_msg.append(f"Template URL: {self.other_data['prompt_url']}")
            raise ValueError("\n".join(error_msg))

    def _get_template_variables(self) -> Set[str]:
        """Get all variables used as placeholders in the template content.

        Returns:
            Set of variable names used as placeholders in the template
        """
        template_variables = set()
        if hasattr(self, "template"):
            template_variables = self.renderer.get_variable_names(self.template)
        elif hasattr(self, "messages"):
            for msg in self.messages:
                template_variables.update(self.renderer.get_variable_names(msg["content"]))
        return template_variables

    def _detect_double_brace_syntax(self, prompt_data: Dict[str, Any]) -> bool:
        """Detect if the template uses simple {{var}} syntax without Jinja2 features."""

        def contains_double_brace(text: str) -> bool:
            # Look for {{var}} pattern but exclude Jinja2-specific patterns
            basic_var = r"\{\{[^{}|.\[]+\}\}"  # Only match simple variables
            return bool(re.search(basic_var, text))

        if "template" in prompt_data:
            return contains_double_brace(prompt_data["template"])
        elif "messages" in prompt_data:
            return any(contains_double_brace(msg["content"]) for msg in prompt_data["messages"])
        return False

    def _detect_jinja2_syntax(self, prompt_data: Dict[str, Any]) -> bool:
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

        if "template" in prompt_data:
            return contains_jinja2(prompt_data["template"])
        elif "messages" in prompt_data:
            return any(contains_jinja2(msg["content"]) for msg in prompt_data["messages"])
        return False


class TextPromptTemplate(BasePromptTemplate):
    """A class representing a standard text prompt template.

    Examples:
        Download and use a text prompt template:
        >>> from hf_hub_prompts import PromptTemplateLoader
        >>> # Download example translation prompt
        >>> prompt_template = PromptTemplateLoader.from_hub(
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

    def populate_template(self, **user_provided_variables: Any) -> PopulatedPrompt:
        """Populate the prompt by replacing placeholders with provided values.

        Examples:
            >>> from hf_hub_prompts import PromptTemplateLoader
            >>> prompt_template = PromptTemplateLoader.from_hub(
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
        self._validate_user_provided_variables(user_provided_variables)
        populated_prompt = self._fill_placeholders(self.template, user_provided_variables)
        return PopulatedPrompt(content=populated_prompt)

    def to_langchain_template(self) -> "LC_PromptTemplate":
        """Convert the TextPromptTemplate to a LangChain PromptTemplate.

        Examples:
            >>> from hf_hub_prompts import PromptTemplateLoader
            >>> prompt_template = PromptTemplateLoader.from_hub(
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
        >>> from hf_hub_prompts import PromptTemplateLoader
        >>> # Download example code teaching prompt
        >>> prompt_template = PromptTemplateLoader.from_hub(
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

    def populate_template(self, **user_provided_variables: Any) -> PopulatedPrompt:
        """Populate the prompt messages by replacing placeholders with provided values.

        Examples:
            >>> from hf_hub_prompts import PromptTemplateLoader
            >>> prompt_template = PromptTemplateLoader.from_hub(
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
            **user_provided_variables: The values to fill placeholders in the messages.

        Returns:
            PopulatedPrompt: A PopulatedPrompt object containing the populated messages.
        """
        self._validate_user_provided_variables(user_provided_variables)

        messages_populated = [
            {**msg, "content": self._fill_placeholders(msg["content"], user_provided_variables)}
            for msg in self.messages
        ]
        return PopulatedPrompt(content=messages_populated)

    def create_messages(
        self, client: str = "openai", **user_provided_variables: Any
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Convenience method to populate a prompt template and format for client in one step.

        Examples:
            >>> from hf_hub_prompts import PromptTemplateLoader
            >>> prompt_template = PromptTemplateLoader.from_hub(
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
            **user_provided_variables: The variables to fill into the prompt template. For example, if your template
                expects variables like 'name' and 'age', pass them as keyword arguments:

        Returns:
            Union[List[Dict[str, Any]], Dict[str, Any]]: Populated and formatted messages.
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
            >>> prompt_template = PromptTemplateLoader.from_hub(
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


class TemplateRenderer(ABC):
    """Abstract base class for template rendering strategies."""

    @abstractmethod
    def render(self, template_str: str, user_provided_variables: Dict[str, Any]) -> str:
        """Render the template with given user_provided_variables."""
        pass

    @abstractmethod
    def get_variable_names(self, template_str: str) -> Set[str]:
        """Extract variable names from template."""
        pass


class SingleBraceRenderer(TemplateRenderer):
    """Template renderer using regex for basic {var} substitution."""

    def render(self, template_str: str, user_provided_variables: Dict[str, Any]) -> str:
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


class DoubleBraceRenderer(TemplateRenderer):
    """Template renderer using regex for {{var}} substitution."""

    def render(self, template_str: str, user_provided_variables: Dict[str, Any]) -> str:
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


class Jinja2TemplateRenderer(TemplateRenderer):
    """Jinja2 template renderer with configurable security levels.

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

    def render(self, template_str: str, user_provided_variables: Dict[str, Any]) -> str:
        """Render the template with given user_provided_variables."""
        try:
            template = self.env.from_string(template_str)
            rendered = template.render(**user_provided_variables)
            # Ensure we return a string for mypy
            return str(rendered)
        except jinja2.TemplateSyntaxError as e:
            raise ValueError(
                f"Invalid template syntax at line {e.lineno}: {str(e)}\n" f"Security level: {self.security_level}"
            ) from e
        except jinja2.UndefinedError as e:
            raise ValueError(
                f"Undefined variable in template: {str(e)}\n" "Make sure all required variables are provided"
            ) from e
        except Exception as e:
            raise ValueError(f"Error rendering template: {str(e)}") from e

    def get_variable_names(self, template_str: str) -> Set[str]:
        """Extract variable names from template."""
        try:
            ast = self.env.parse(template_str)
            variables = meta.find_undeclared_variables(ast)
            # Ensure we return a set of strings for mypy
            return {str(var) for var in variables}
        except jinja2.TemplateSyntaxError as e:
            raise ValueError(f"Invalid template syntax: {str(e)}") from e
