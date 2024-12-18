import io
import json
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Match, Optional, Set, Tuple, Union

import jinja2
import yaml
from huggingface_hub import HfApi, metadata_update
from huggingface_hub.hf_api import CommitInfo
from huggingface_hub.repocard import RepoCard
from huggingface_hub.utils import RepositoryNotFoundError
from jinja2 import Environment, meta

from .constants import Jinja2SecurityLevel, PopulatorType
from .populated_prompt import PopulatedPrompt
from .utils import create_yaml_handler, format_template_content


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
        template_variables: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        client_parameters: Optional[Dict[str, Any]] = None,
        custom_data: Optional[Dict[str, Any]] = None,
        populator: Optional[PopulatorType] = None,
        jinja2_security_level: Jinja2SecurityLevel = "standard",
    ) -> None:
        """Initialize a prompt template.

        Args:
            template: The template string or list of message dictionaries
            template_variables: List of variables required by the template
            metadata: Optional metadata about the prompt template
            client_parameters: Optional parameters for LLM client configuration (e.g., temperature, model)
            custom_data: Optional custom data which does not fit into the other categories
            populator: Optional template populator type. Choose from Literal["double_brace", "single_brace", "jinja2"]
            jinja2_security_level: Security level for Jinja2 populator. Choose from Literal["strict", "standard", "relaxed"]

        Raises:
            TypeError: If input types don't match expected types
            ValueError: If template format is invalid
        """
        # Type validation
        if template_variables is not None and not isinstance(template_variables, list):
            raise TypeError(f"template_variables must be a list, got {type(template_variables).__name__}")
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError(f"metadata must be a dict, got {type(metadata).__name__}")
        if client_parameters is not None and not isinstance(client_parameters, dict):
            raise TypeError(f"client_parameters must be a dict, got {type(client_parameters).__name__}")
        if custom_data is not None and not isinstance(custom_data, dict):
            raise TypeError(f"custom_data must be a dict, got {type(custom_data).__name__}")

        # Format validation
        self._validate_template_format(template)

        # Initialize attributes
        self.template = template
        self.template_variables = template_variables or []
        self.metadata = metadata or {}
        self.client_parameters = client_parameters or {}
        self.custom_data = custom_data or {}

        # set up the template populator
        self._set_up_populator(populator, jinja2_security_level)

        # Validate that variables provided in template and template_variables are equal
        if self.template_variables:
            self._validate_template_variables_equality()

    @abstractmethod
    def populate_template(self, **user_provided_variables: Any) -> PopulatedPrompt:
        """Abstract method to populate the prompt template with user-provided variables.

        Args:
            **user_provided_variables: The values to fill placeholders in the template.

        Returns:
            PopulatedPrompt: A PopulatedPrompt object containing the populated content.
        """
        pass

    def save_to_hub(
        self,
        repo_id: str,
        filename: str,
        repo_type: str = "dataset",
        format: Optional[Literal["yaml", "json"]] = None,
        yaml_library: str = "ruamel",
        prettify_template: bool = True,
        token: Optional[str] = None,
        create_repo: bool = False,
        private: bool = False,
        exist_ok: bool = True,
        resource_group_id: Optional[str] = None,
        revision: Optional[str] = None,
        create_pr: bool = False,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        parent_commit: Optional[str] = None,
    ) -> CommitInfo:
        """Save the prompt template to the Hugging Face Hub as a YAML or JSON file.

        Args:
            repo_id: The repository ID on the Hugging Face Hub (e.g., "username/repo-name")
            filename: Name of the file to save (e.g., "prompt.yaml" or "prompt.json")
            repo_type: Type of repository ("dataset", "model", or "space"). Defaults to "dataset"
            token: Hugging Face API token. If None, will use token from environment
            commit_message: Custom commit message. If None, uses default message
            create_repo: Whether to create the repository if it doesn't exist. Defaults to False
            format: Output format ("yaml" or "json"). If None, inferred from filename extension
            yaml_library: YAML library to use ("ruamel" or "pyyaml"). Defaults to "ruamel" for better formatting and format preservation.
            prettify_template: If true format the template content with literal block scalars, i.e. "|-" in yaml.
                This makes the string behave like a Python '''...''' block to make strings easier to read and edit.
                Defaults to True
            private: Whether to create a private repository. Defaults to False
            exist_ok: Don't error if repo already exists. Defaults to True
            resource_group_id: Optional resource group ID to associate with the repository
            revision: Optional branch/revision to push to. Defaults to main branch
            create_pr: Whether to create a Pull Request instead of pushing directly. Defaults to False
            commit_description: Optional commit description
            parent_commit: Optional parent commit to create PR from

        Returns:
            CommitInfo: Information about the commit/PR

        Examples:
            >>> from prompt_templates import ChatPromptTemplate
            >>> messages_template = [
            ...     {"role": "system", "content": "You are a coding assistant who explains concepts clearly and provides short examples."},
            ...     {"role": "user", "content": "Explain what {{concept}} is in {{programming_language}}."}
            ... ]
            >>> template_variables = ["concept", "programming_language"]
            >>> metadata = {
            ...     "name": "Code Teacher",
            ...     "description": "A simple chat prompt for explaining programming concepts with examples",
            ...     "tags": ["programming", "education"],
            ...     "version": "0.0.1",
            ...     "author": "My Awesome Company"
            ... }
            >>> prompt_template = ChatPromptTemplate(
            ...     template=messages_template,
            ...     template_variables=template_variables,
            ...     metadata=metadata,
            ... )
            >>> prompt_template.save_to_hub(
            ...     repo_id="MoritzLaurer/example_prompts_test",
            ...     filename="code_teacher_test.yaml",
            ...     #create_repo=True,  # if the repo does not exist, create it
            ...     #private=True,  # if you want to create a private repo
            ...     #token="hf_..."
            ... )
            'https://huggingface.co/MoritzLaurer/example_prompts_test/blob/main/code_teacher_test.yaml'
        """

        # Handle format inference and validation
        if format is None:
            # Infer format from extension
            extension = Path(filename).suffix.lstrip(".")
            if extension in ["yaml", "yml"]:
                format = "yaml"
            elif extension == "json":
                format = "json"
            else:
                format = "yaml"  # default if no extension
                filename += ".yaml"
        else:
            # Validate explicitly provided format matches file extension
            if format not in ["yaml", "yml", "json"]:
                raise ValueError(f"Unsupported format: {format}")

            file_extension = Path(filename).suffix.lstrip(".")
            if format in ["yaml", "yml"] and file_extension in ["yaml", "yml"]:
                # Both are YAML variants, so they match
                pass
            elif format != file_extension:
                raise ValueError(f"Provided format '{format}' does not match file extension '{filename}'")

        # Convert template to the specified format
        data = {
            "prompt": {
                "template": self.template,
                "template_variables": self.template_variables,
                "metadata": self.metadata,
                "client_parameters": self.client_parameters,
                "custom_data": self.custom_data,
            }
        }

        if prettify_template:
            data = format_template_content(data)

        if format == "json":
            content = json.dumps(data, indent=2, ensure_ascii=False)
            content_bytes = content.encode("utf-8")
        else:  # yaml
            yaml_handler = create_yaml_handler(yaml_library)
            string_stream = io.StringIO()
            yaml_handler.dump(data, string_stream)
            content = string_stream.getvalue()
            content_bytes = content.encode("utf-8")

        # Upload to Hub
        api = HfApi(token=token)

        if create_repo:
            api.create_repo(
                repo_id=repo_id,
                repo_type=repo_type,
                token=token,
                private=private,
                exist_ok=exist_ok,
                resource_group_id=resource_group_id,
            )

            repocard_text = (
                "---\n"
                "library_name: prompt-templates\n"
                "tags:\n"
                "- prompts\n"
                "- prompt-templates\n"
                "---\n"
                "This repository was created with the `prompt-templates` library and contains\n"
                "prompt templates in the `Files` tab.\n"
                "For easily reusing these templates, see the [prompt-templates documentation](https://github.com/MoritzLaurer/prompt-templates)."
            )
            card = RepoCard(repocard_text)
            card.push_to_hub(
                repo_id=repo_id,
                repo_type=repo_type,
                token=token,
                commit_message="Create repo card with prompt-templates library",
                create_pr=create_pr,
                parent_commit=parent_commit,
            )
        else:
            # Check if repo exists to provide better error message
            try:
                api.repo_info(repo_id=repo_id, repo_type=repo_type)
            except RepositoryNotFoundError as e:
                raise ValueError(f"Repository {repo_id} does not exist. Set create_repo=True to create it.") from e

        if not create_repo:
            # Update repo metadata to make prompt templates discoverable on the HF Hub
            metadata_update(
                repo_id=repo_id,
                metadata={"library_name": "prompt-templates", "tags": ["prompts", "prompt-templates"]},
                repo_type=repo_type,
                overwrite=False,
                token=token,
                commit_message=commit_message or "Update repo metadata with prompt-templates library",
                commit_description=commit_description,
                revision=revision,
                create_pr=create_pr,
                parent_commit=parent_commit,
            )

        # Upload file
        return api.upload_file(
            path_or_fileobj=io.BytesIO(content_bytes),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            commit_message=commit_message or f"Upload prompt template {filename}",
            commit_description=commit_description,
            revision=revision,
            create_pr=create_pr,
            parent_commit=parent_commit,
        )

    def save_to_local(
        self,
        path: Union[str, Path],
        format: Optional[Literal["yaml", "json"]] = None,
        yaml_library: str = "ruamel",
        prettify_template: bool = True,
    ) -> None:
        """Save the prompt template as a local YAML or JSON file.

        Args:
            path: Path where to save the file. Can be string or Path object
            format: Output format ("yaml" or "json"). If None, inferred from filename
            yaml_library: YAML library to use ("ruamel" or "pyyaml"). Defaults to "ruamel" for better formatting and format preservation.
            prettify_template: If true format the template content with literal block scalars, i.e. "|-" in yaml.
                This makes the string behave like a Python '''...''' block to make strings easier to read and edit.
                Defaults to True

        Examples:
            >>> from prompt_templates import ChatPromptTemplate
            >>> messages_template = [
            ...     {"role": "system", "content": "You are a coding assistant who explains concepts clearly and provides short examples."},
            ...     {"role": "user", "content": "Explain what {{concept}} is in {{programming_language}}."}
            ... ]
            >>> template_variables = ["concept", "programming_language"]
            >>> metadata = {
            ...     "name": "Code Teacher",
            ...     "description": "A simple chat prompt for explaining programming concepts with examples",
            ...     "tags": ["programming", "education"],
            ...     "version": "0.0.1",
            ...     "author": "My Awesome Company"
            ... }
            >>> prompt_template = ChatPromptTemplate(
            ...     template=messages_template,
            ...     template_variables=template_variables,
            ...     metadata=metadata,
            ... )
            >>> prompt_template.save_to_local("code_teacher_test.yaml")
        """

        path = Path(path)
        # Handle format inference and validation
        file_extension = path.suffix.lstrip(".")
        if format is None:
            # Infer format from extension
            if file_extension in ["yaml", "yml"]:
                format = "yaml"
            elif file_extension == "json":
                format = "json"
            else:
                raise ValueError(f"Cannot infer format from file extension: {path.suffix}")
        else:
            # Validate explicitly provided format matches file extension
            if format not in ["yaml", "yml", "json"]:
                raise ValueError(f"Unsupported format: {format}")
            if format in ["yaml", "yml"] and file_extension in ["yaml", "yml"]:
                # Both are YAML variants, so they match
                pass
            elif format != file_extension:
                raise ValueError(f"Provided format '{format}' does not match file extension '{path.suffix}'")

        data = {
            "prompt": {
                "template": self.template,
                "template_variables": self.template_variables,
                "metadata": self.metadata,
                "client_parameters": self.client_parameters,
                "custom_data": self.custom_data,
            }
        }

        if prettify_template:
            data = format_template_content(data)

        with open(path, "w", encoding="utf-8") as f:
            if format == "json":
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:  # yaml
                yaml_handler = create_yaml_handler(yaml_library)
                if yaml_library == "pyyaml":
                    yaml_handler.dump(data, f, sort_keys=False, allow_unicode=True)
                else:  # ruamel
                    yaml_handler.dump(data, f)

    def display(self, format: Literal["json", "yaml"] = "json") -> None:
        """Display the prompt configuration in the specified format.

        Examples:
            >>> from prompt_templates import PromptTemplateLoader
            >>> prompt_template = PromptTemplateLoader.from_hub(
            ...     repo_id="MoritzLaurer/example_prompts",
            ...     filename="translate.yaml"
            ... )
            >>> prompt_template.display(format="yaml")  # doctest: +NORMALIZE_WHITESPACE
            template: 'Translate the following text to {language}:
              {text}'
            template_variables:
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
        # Create a dict of all attributes except custom_data
        display_dict = self.__dict__.copy()
        display_dict.pop("custom_data", None)

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
        # We know that template variables and template_variables are equal based on _validate_template_variables_equality, so we can validate against either
        required_variables = (
            set(self.template_variables) if self.template_variables else self._get_variables_in_template()
        )
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

            raise ValueError("\n".join(error_parts))

    def _validate_template_variables_equality(self) -> None:
        """Validate that the declared template_variables and the actual variables in the template are identical."""
        variables_in_template = self._get_variables_in_template()
        template_variables = set(self.template_variables or [])

        # Check for mismatches
        undeclared_template_variables = variables_in_template - template_variables
        unused_template_variables = template_variables - variables_in_template

        if undeclared_template_variables or unused_template_variables:
            error_parts = []

            if undeclared_template_variables:
                error_parts.append(
                    f"template contains variables that are not declared in template_variables: {list(undeclared_template_variables)}"
                )
            if unused_template_variables:
                error_parts.append(
                    f"template_variables declares variables that are not used in template: {list(unused_template_variables)}"
                )

            template_extract = (
                str(self.template)[:100] + "..." if len(str(self.template)) > 100 else str(self.template)
            )
            error_parts.append(f"\nTemplate extract: {template_extract}")

            raise ValueError("\n".join(error_parts))

    def _get_variables_in_template(self) -> Set[str]:
        """Get all variables used as placeholders in the template string or messages dictionary."""
        variables_in_template = set()
        if isinstance(self.template, str):
            variables_in_template = self.populator.get_variable_names(self.template)
        elif isinstance(self.template, list) and any(isinstance(item, dict) for item in self.template):
            for message in self.template:
                content = message["content"]
                if isinstance(content, str):
                    variables_in_template.update(self.populator.get_variable_names(content))
                elif isinstance(content, list):
                    # Recursively search for variables in nested content
                    for item in content:
                        variables_in_template.update(self._get_variables_in_dict(item))
        return variables_in_template

    def _get_variables_in_dict(self, d: Dict[str, Any]) -> Set[str]:
        """Recursively extract variables from a dictionary structure."""
        variables = set()
        for value in d.values():
            if isinstance(value, str):
                variables.update(self.populator.get_variable_names(value))
            elif isinstance(value, dict):
                variables.update(self._get_variables_in_dict(value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        variables.update(self._get_variables_in_dict(item))
        return variables

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

                if not isinstance(msg["role"], str):
                    raise ValueError("Message 'role' must be a string")

                # Allow content to be either a string or a list of content items
                if not isinstance(msg["content"], (str, list)):
                    raise ValueError("Message 'content' must be either a string or a list")

                # If content is a list, validate each item
                # Can be list if passing images to OpenAI API
                if isinstance(msg["content"], list):
                    for item in msg["content"]:
                        if not isinstance(item, dict):
                            raise ValueError("Each content item in a list must be a dictionary")

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

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BasePromptTemplate):
            return False

        return (
            self.template == other.template
            and self.template_variables == other.template_variables
            and self.metadata == other.metadata
            and self.client_parameters == other.client_parameters
            and self.custom_data == other.custom_data
            and self.populator_type == other.populator_type
        )


class TextPromptTemplate(BasePromptTemplate):
    """A class representing a standard text prompt template.

    Examples:
        Instantiate a text prompt template:
        >>> from prompt_templates import TextPromptTemplate
        >>> template_text = "Translate the following text to {{language}}:\\n{{text}}"
        >>> template_variables = ["language", "text"]
        >>> metadata = {
        ...     "name": "Simple Translator",
        ...     "description": "A simple translation prompt for illustrating the standard prompt YAML format",
        ...     "tags": ["translation", "multilinguality"],
        ...     "version": "0.0.1",
        ...     "author": "Some Person"
        }
        >>> prompt_template = TextPromptTemplate(
        ...     template=template_text,
        ...     template_variables=template_variables,
        ...     metadata=metadata
        ... )
        >>> print(prompt_template)
        TextPromptTemplate(template='Translate the following text to {{language}}:\\n{{text}}', template_variables=['language', 'text'], metadata={'name': 'Simple Translator', 'description': 'A simple translation prompt for illustrating the standard prompt YAML format', 'tags': ['translation', 'multilinguality'], 'version': '0.0.1', 'author': 'Some Person'}, custom_data={}, populator_type='double_brace', populator=<prompt_templates.prompt_templates.DoubleBracePopulator object at 0x...>)

        >>> # Inspect template attributes
        >>> prompt_template.template
        'Translate the following text to {language}:\\n{text}'
        >>> prompt_template.template_variables
        ['language', 'text']
        >>> prompt_template.metadata['name']
        'Simple Translator'

        >>> # Populate the template
        >>> prompt = prompt_template.populate_template(
        ...     language="French",
        ...     text="Hello world!"
        ... )
        >>> print(prompt)
        'Translate the following text to French:\\nHello world!'

        Or download the same text prompt template from the Hub:
        >>> from prompt_templates import PromptTemplateLoader
        >>> prompt_template_downloaded = PromptTemplateLoader.from_hub(
        ...     repo_id="MoritzLaurer/example_prompts",
        ...     filename="translate.yaml"
        ... )
        >>> prompt_template_downloaded == prompt_template
        True
    """

    def __init__(
        self,
        template: str,
        template_variables: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        client_parameters: Optional[Dict[str, Any]] = None,
        custom_data: Optional[Dict[str, Any]] = None,
        populator: Optional[PopulatorType] = None,
        jinja2_security_level: Jinja2SecurityLevel = "standard",
    ) -> None:
        super().__init__(
            template=template,
            template_variables=template_variables,
            metadata=metadata,
            client_parameters=client_parameters,
            custom_data=custom_data,
            populator=populator,
            jinja2_security_level=jinja2_security_level,
        )

    def populate_template(self, **user_provided_variables: Any) -> PopulatedPrompt:
        """Populate the prompt by replacing placeholders with provided values.

        Examples:
            >>> from prompt_templates import PromptTemplateLoader
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
            >>> from prompt_templates import PromptTemplateLoader
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
            input_variables=self.template_variables,
            metadata=self.metadata,
        )


class ChatPromptTemplate(BasePromptTemplate):
    """A class representing a chat prompt template that can be formatted for and used with various LLM clients.

    Examples:
        Instantiate a chat prompt template:
        >>> from prompt_templates import ChatPromptTemplate
        >>> template_messages = [
        ...     {"role": "system", "content": "You are a coding assistant who explains concepts clearly and provides short examples."},
        ...     {"role": "user", "content": "Explain what {{concept}} is in {{programming_language}}."}
        ... ]
        >>> template_variables = ["concept", "programming_language"]
        >>> metadata = {
        ...     "name": "Code Teacher",
        ...     "description": "A simple chat prompt for explaining programming concepts with examples",
        ...     "tags": ["programming", "education"],
        ...     "version": "0.0.1",
        ...     "author": "My Awesome Company"
        ... }
        >>> prompt_template = ChatPromptTemplate(
        ...     template=template_messages,
        ...     template_variables=template_variables,
        ...     metadata=metadata
        ... )
        >>> print(prompt_template)
        ChatPromptTemplate(template=[{'role': 'system', 'content': 'You are a coding a..., template_variables=['concept', 'programming_language'], metadata={'name': 'Code Teacher', 'description': 'A simple ..., custom_data={}, populator_type='double_brace', populator=<prompt_templates.prompt_templates.DoubleBracePopula...)
        >>> # Inspect template attributes
        >>> prompt_template.template
        [{'role': 'system', 'content': 'You are a coding assistant who explains concepts clearly and provides short examples.'}, {'role': 'user', 'content': 'Explain what {concept} is in {programming_language}.'}]
        >>> prompt_template.template_variables
        ['concept', 'programming_language']

        >>> # Populate the template
        >>> messages = prompt_template.populate_template(
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
        >>> from prompt_templates import PromptTemplateLoader
        >>> prompt_template_downloaded = PromptTemplateLoader.from_hub(
        ...     repo_id="MoritzLaurer/example_prompts",
        ...     filename="code_teacher.yaml"
        ... )
        >>> prompt_template_downloaded == prompt_template
        True
    """

    template: List[Dict[str, str]]

    def __init__(
        self,
        template: List[Dict[str, Any]],
        template_variables: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        client_parameters: Optional[Dict[str, Any]] = None,
        custom_data: Optional[Dict[str, Any]] = None,
        populator: Optional[PopulatorType] = None,
        jinja2_security_level: Jinja2SecurityLevel = "standard",
    ) -> None:
        super().__init__(
            template=template,
            template_variables=template_variables,
            metadata=metadata,
            client_parameters=client_parameters,
            custom_data=custom_data,
            populator=populator,
            jinja2_security_level=jinja2_security_level,
        )

    def populate_template(self, **user_provided_variables: Any) -> PopulatedPrompt:
        """Populate the prompt template messages by replacing placeholders with provided values.

        Examples:
            >>> from prompt_templates import PromptTemplateLoader
            >>> prompt_template = PromptTemplateLoader.from_hub(
            ...     repo_id="MoritzLaurer/example_prompts",
            ...     filename="code_teacher.yaml"
            ... )
            >>> messages = prompt_template.populate_template(
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
            >>> from prompt_templates import PromptTemplateLoader
            >>> prompt_template = PromptTemplateLoader.from_hub(
            ...     repo_id="MoritzLaurer/example_prompts",
            ...     filename="code_teacher.yaml"
            ... )
            >>> # Format for OpenAI (default)
            >>> messages = prompt_template.create_messages(
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
            >>> from prompt_templates import PromptTemplateLoader
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

        # LangChain expects a list of tuples of the form (role, content)
        messages: List[Tuple[str, str]] = [
            (str(message["role"]), str(message["content"])) for message in self.template
        ]
        return LC_ChatPromptTemplate(
            messages=messages,
            input_variables=self.template_variables,
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
