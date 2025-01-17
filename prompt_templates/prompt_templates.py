import io
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Set, Tuple, Union

import yaml
from huggingface_hub import HfApi, hf_hub_download, metadata_update
from huggingface_hub.hf_api import CommitInfo
from huggingface_hub.repocard import RepoCard
from huggingface_hub.utils import RepositoryNotFoundError, validate_repo_id

from .constants import VALID_PROMPT_EXTENSIONS, ClientType, Jinja2SecurityLevel, PopulatorType
from .populators import DoubleBracePopulator, Jinja2TemplatePopulator, SingleBracePopulator, TemplatePopulator
from .utils import create_yaml_handler, format_for_client, format_template_content


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
        populator: PopulatorType = "jinja2",
        jinja2_security_level: Jinja2SecurityLevel = "standard",
    ) -> None:
        """Initialize a prompt template.

        Args:
            template: The template string or list of message dictionaries.
            template_variables: List of variables used in the template.
            metadata: Dictionary of metadata about the template.
            client_parameters: Dictionary of parameters for the inference client (e.g., temperature, model).
            custom_data: Dictionary of custom data which does not fit into the other categories.
            populator: The populator to use. Choose from Literal["jinja2", "double_brace_regex", "single_brace_regex"]. Defaults to "jinja2".
            jinja2_security_level: Security level for Jinja2 populator. Choose from Literal["strict", "standard", "relaxed"]. Defaults to "standard".
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

        # Initialize attributes
        self.template = template
        self.template_variables = template_variables or []
        self.metadata = metadata or {}
        self.client_parameters = client_parameters or {}
        self.custom_data = custom_data or {}
        self.populator = populator
        self.jinja2_security_level = jinja2_security_level

        # Validate template format
        self._validate_template_format(self.template)

        # Create populator instance
        self._create_populator_instance(self.populator, self.jinja2_security_level)

        # Validate that variables provided in template and template_variables are equal
        if self.template_variables:
            self._validate_template_variables_equality()

    @abstractmethod
    def populate_template(self, **user_provided_variables: Any) -> str | List[Dict[str, Any]]:
        """Abstract method to populate the prompt template with user-provided variables.

        Args:
            **user_provided_variables: The values to fill placeholders in the template.

        Returns:
            str | List[Dict[str, Any]]: The populated prompt content.
        """
        pass

    @classmethod
    def load_from_local(
        cls,
        path: Union[str, Path],
        populator: PopulatorType = "jinja2",
        jinja2_security_level: Literal["strict", "standard", "relaxed"] = "standard",
        yaml_library: str = "ruamel",
    ) -> Union["TextPromptTemplate", "ChatPromptTemplate"]:
        """Load a prompt template from a local YAML file.

        Args:
            path (Union[str, Path]): Path to the YAML file containing the prompt template
            populator ([PopulatorType]): The populator type to use among Literal["double_brace_regex", "single_brace_regex", "jinja2"]. Defaults to "jinja2".
            jinja2_security_level (Literal["strict", "standard", "relaxed"], optional): The security level for the Jinja2 populator. Defaults to "standard".
            yaml_library (str, optional): The YAML library to use ("ruamel" or "pyyaml"). Defaults to "ruamel".

        Returns:
            Union[TextPromptTemplate, ChatPromptTemplate]: The loaded template instance

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is not a .yaml/.yml file
            ValueError: If the YAML structure is invalid
            ValueError: If attempting to load a text template with ChatPromptTemplate or vice versa

        Examples:
            Download a text prompt template:
            >>> from prompt_templates import TextPromptTemplate
            >>> prompt_template = TextPromptTemplate.load_from_local("./tests/test_data/example_prompts/translate.yaml")
            >>> print(prompt_template)
            TextPromptTemplate(template='Translate the following text to {{language}}:\\n{{..., template_variables=['language', 'text'], metadata={'name': 'Simple Translator', 'description': 'A si..., client_parameters={}, custom_data={}, populator='jinja2', jinja2_security_level='standard')
            >>> prompt_template.template
            'Translate the following text to {{language}}:\\n{{text}}'
            >>> prompt_template.template_variables
            ['language', 'text']
            >>> prompt_template.metadata['name']
            'Simple Translator'

            Download a chat prompt template:
            >>> from prompt_templates import ChatPromptTemplate
            >>> prompt_template = ChatPromptTemplate.load_from_local("./tests/test_data/example_prompts/code_teacher.yaml")
            >>> print(prompt_template)
            ChatPromptTemplate(template=[{'role': 'system', 'content': 'You are a coding a...', template_variables=['concept', 'programming_language'], metadata={'name': 'Code Teacher', 'description': 'A simple ...', client_parameters={}, custom_data={}, populator='jinja2', jinja2_security_level='standard')
            >>> prompt_template.template
            [{'role': 'system', 'content': 'You are a coding assistant who explains concepts clearly and provides short examples.'}, {'role': 'user', 'content': 'Explain what {{concept}} is in {{programming_language}}.'}]
            >>> prompt_template.template_variables
            ['concept', 'programming_language']
            >>> prompt_template.metadata['version']
            '0.0.1'

        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Template file not found: {path}")
        if path.suffix not in VALID_PROMPT_EXTENSIONS:
            raise ValueError(f"Template file must be a .yaml or .yml file, got: {path}")

        yaml = create_yaml_handler(yaml_library)
        try:
            with open(path, "r") as file:
                if yaml_library == "ruamel":
                    prompt_file_dic = yaml.load(file)
                else:
                    prompt_file_dic = yaml.safe_load(file)
        except Exception as e:
            raise ValueError(
                f"Failed to parse '{path}' as a valid YAML file. "
                f"Please ensure the file is properly formatted.\n"
                f"Error details: {str(e)}"
            ) from e

        cls._validate_template_type(prompt_file_dic, str(path))

        return cls._load_template_from_dict(
            prompt_file_dic, populator=populator, jinja2_security_level=jinja2_security_level
        )

    @classmethod
    def load_from_hub(
        cls,
        repo_id: str,
        filename: str,
        repo_type: str = "dataset",
        revision: Optional[str] = None,
        populator: PopulatorType = "jinja2",
        jinja2_security_level: Literal["strict", "standard", "relaxed"] = "standard",
        yaml_library: str = "ruamel",
    ) -> Union["TextPromptTemplate", "ChatPromptTemplate"]:
        """Load a prompt template from the Hugging Face Hub.

        Downloads and loads a prompt template from a repository on the Hugging Face Hub.
        The template file should be a YAML file following the standardized format.

        Args:
            repo_id (str): The repository ID on Hugging Face Hub (e.g., 'username/repo')
            filename (str): Name of the YAML file containing the template
            repo_type (str, optional): Type of repository. Must be one of
                ['dataset', 'model', 'space']. Defaults to "dataset"
            revision (Optional[str], optional): Git revision to download from.
                Can be a branch name, tag, or commit hash. Defaults to None
            populator ([PopulatorType]): The populator type to use among Literal["double_brace_regex", "single_brace_regex", "jinja2"]. Defaults to "jinja2".
            jinja2_security_level (Literal["strict", "standard", "relaxed"], optional): The security level for the Jinja2 populator. Defaults to "standard".
            yaml_library (str, optional): The YAML library to use ("ruamel" or "pyyaml"). Defaults to "ruamel".


        Returns:
            Union[TextPromptTemplate, ChatPromptTemplate]: The loaded template instance

        Raises:
            ValueError: If repo_id format is invalid
            ValueError: If repo_type is invalid
            FileNotFoundError: If file cannot be downloaded from Hub
            ValueError: If the YAML structure is invalid
            ValueError: If attempting to load a text template with ChatPromptTemplate or vice versa

        Examples:
            Download a text prompt template:
            >>> from prompt_templates import TextPromptTemplate
            >>> prompt_template = TextPromptTemplate.load_from_hub(
            ...     repo_id="MoritzLaurer/example_prompts",
            ...     filename="translate.yaml"
            ... )
            >>> print(prompt_template)
            TextPromptTemplate(template='Translate the following text to {{language}}:\\n{{...', template_variables=['language', 'text'], metadata={'name': 'Simple Translator', 'description': 'A si...', client_parameters={}, custom_data={}, populator='jinja2', jinja2_security_level='standard')
            >>> prompt_template.template
            'Translate the following text to {{language}}:\\n{{text}}'
            >>> prompt_template.template_variables
            ['language', 'text']
            >>> prompt_template.metadata['name']
            'Simple Translator'

            Download a chat prompt template:
            >>> from prompt_templates import ChatPromptTemplate
            >>> prompt_template = ChatPromptTemplate.load_from_hub(
            ...     repo_id="MoritzLaurer/example_prompts",
            ...     filename="code_teacher.yaml"
            ... )
            >>> print(prompt_template)
            ChatPromptTemplate(template=[{'role': 'system', 'content': 'You are a coding a...', template_variables=['concept', 'programming_language'], metadata={'name': 'Code Teacher', 'description': 'A simple ...', client_parameters={}, custom_data={}, populator='jinja2', jinja2_security_level='standard')
            >>> prompt_template.template
            [{'role': 'system', 'content': 'You are a coding assistant who explains concepts clearly and provides short examples.'}, {'role': 'user', 'content': 'Explain what {{concept}} is in {{programming_language}}.'}]
            >>> prompt_template.template_variables
            ['concept', 'programming_language']
            >>> prompt_template.metadata['version']
            '0.0.1'
        """
        # Validate Hub parameters
        try:
            validate_repo_id(repo_id)
        except ValueError as e:
            raise ValueError(f"Invalid repo_id format: {str(e)}") from e

        if repo_type not in ["dataset", "model", "space"]:
            raise ValueError(f"repo_type must be one of ['dataset', 'model', 'space'], got {repo_type}")

        # Ensure .yaml extension
        if not filename.endswith(VALID_PROMPT_EXTENSIONS):
            filename += ".yaml"

        try:
            file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type, revision=revision)
        except Exception as e:
            raise FileNotFoundError(f"Failed to download template from Hub: {str(e)}") from e

        yaml = create_yaml_handler(yaml_library)
        try:
            with open(file_path, "r") as file:
                if yaml_library == "ruamel":
                    prompt_file_dic = yaml.load(file)
                else:
                    prompt_file_dic = yaml.safe_load(file)
        except Exception as e:
            raise ValueError(
                f"Failed to parse '{filename}' as a valid YAML file. "
                f"Please ensure the file is properly formatted.\n"
                f"Error details: {str(e)}"
            ) from e

        file_info = f"'{filename}' from '{repo_id}'"
        cls._validate_template_type(prompt_file_dic, file_info)

        return cls._load_template_from_dict(
            prompt_file_dic, populator=populator, jinja2_security_level=jinja2_security_level
        )

    @staticmethod
    def _load_template_from_dict(
        prompt_file_dic: Dict[str, Any],
        populator: PopulatorType = "jinja2",
        jinja2_security_level: Literal["strict", "standard", "relaxed"] = "standard",
    ) -> Union["TextPromptTemplate", "ChatPromptTemplate"]:
        """Internal method to load a template from parsed YAML data.

        Args:
            prompt_file_dic: Dictionary containing parsed YAML data
            populator: Optional template populator type
            jinja2_security_level: Security level for Jinja2 populator

        Returns:
            Union[TextPromptTemplate, ChatPromptTemplate]: Loaded template instance

        Raises:
            ValueError: If YAML structure is invalid
        """
        # Validate YAML structure
        if "prompt" not in prompt_file_dic:
            raise ValueError(
                f"Invalid YAML structure: The top-level keys are {list(prompt_file_dic.keys())}. "
                "The YAML file must contain the key 'prompt' as the top-level key."
            )

        prompt_data = prompt_file_dic["prompt"]

        # Check for standard "template" key
        if "template" not in prompt_data:
            if "messages" in prompt_data:
                template = prompt_data["messages"]
                del prompt_data["messages"]
                logger.info(
                    "The YAML file uses the 'messages' key for the chat prompt template following the LangChain format. "
                    "The 'messages' key is renamed to 'template' for simplicity and consistency in this library."
                )
            else:
                raise ValueError(
                    f"Invalid YAML structure under 'prompt' key: {list(prompt_data.keys())}. "
                    "The YAML file must contain a 'template' key under 'prompt'. "
                    "Please refer to the documentation for a compatible YAML example."
                )
        else:
            template = prompt_data["template"]

        # Extract fields
        template_variables = prompt_data.get("template_variables")
        metadata = prompt_data.get("metadata")
        client_parameters = prompt_data.get("client_parameters")
        custom_data = {
            k: v
            for k, v in prompt_data.items()
            if k not in ["template", "template_variables", "metadata", "client_parameters", "custom_data"]
        }
        custom_data = {**prompt_data.get("custom_data", {}), **custom_data}

        # Determine template type and create appropriate instance
        if isinstance(template, list) and any(isinstance(item, dict) for item in template):
            return ChatPromptTemplate(
                template=template,
                template_variables=template_variables,
                metadata=metadata,
                client_parameters=client_parameters,
                custom_data=custom_data,
                populator=populator,
                jinja2_security_level=jinja2_security_level,
            )
        elif isinstance(template, str):
            return TextPromptTemplate(
                template=template,
                template_variables=template_variables,
                metadata=metadata,
                client_parameters=client_parameters,
                custom_data=custom_data,
                populator=populator,
                jinja2_security_level=jinja2_security_level,
            )
        else:
            raise ValueError(
                f"Invalid template type: {type(template)}. "
                "Template must be either a string for text prompts or a list of dictionaries for chat prompts."
            )

    @classmethod
    def _validate_template_type(cls, prompt_file_dic: Dict[str, Any], file_info: str) -> None:
        """Validate that the template type matches the class it was called from.

        Args:
            prompt_file_dic: Dictionary containing parsed YAML data
            file_info: String describing the file location (e.g., file path or Hub location)
                for error messages

        Raises:
            ValueError: If template structure is invalid or type doesn't match the class
        """
        if not isinstance(prompt_file_dic, dict) or "prompt" not in prompt_file_dic:
            raise ValueError(f"File '{file_info}' must contain a top-level 'prompt' key")

        template = prompt_file_dic["prompt"].get("template")
        if template is None:
            raise ValueError(f"Template is missing in file '{file_info}'")

        is_chat_template = isinstance(template, list) and any(isinstance(item, dict) for item in template)
        is_text_template = isinstance(template, str)

        if cls.__name__ == "ChatPromptTemplate" and not is_chat_template:
            raise ValueError(
                f"Cannot load a text template using ChatPromptTemplate. The template in '{file_info}' "
                "appears to be a text template. Use TextPromptTemplate.load_from_local() or .load_from_hub() instead."
            )
        elif cls.__name__ == "TextPromptTemplate" and not is_text_template:
            raise ValueError(
                f"Cannot load a chat template using TextPromptTemplate. The template in {file_info} "
                "appears to be a chat template. Use ChatPromptTemplate.load_from_local() or .load_from_hub() instead."
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
            >>> prompt_template.save_to_local("./tests/test_data/example_prompts/code_teacher_test.yaml")  # doctest: +SKIP
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
            data["prompt"]["template"] = format_template_content(data["prompt"]["template"])

        with open(path, "w", encoding="utf-8") as f:
            if format == "json":
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:  # yaml
                yaml_handler = create_yaml_handler(yaml_library)
                if yaml_library == "ruamel":
                    yaml_handler.dump(data, f)
                elif yaml_library == "pyyaml":
                    yaml_handler.dump(data, f, sort_keys=False, allow_unicode=True)
                else:
                    raise ValueError(
                        f"Unknown yaml library: {yaml_library}. Valid options are: 'ruamel' (default) or 'pyyaml'."
                    )

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
            >>> prompt_template.save_to_hub(  # doctest: +SKIP
            ...     repo_id="MoritzLaurer/example_prompts_test",
            ...     filename="code_teacher_test.yaml",
            ...     #create_repo=True,  # if the repo does not exist, create it
            ...     #private=True,  # if you want to create a private repo
            ...     #token="hf_..."
            ... )
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
            data["prompt"]["template"] = format_template_content(data["prompt"]["template"])

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

        # Check if repo exists before attempting to create it to avoid overwriting repo card
        try:
            api.repo_info(repo_id=repo_id, repo_type=repo_type)
            repo_exists = True
        except RepositoryNotFoundError:
            repo_exists = False

        if create_repo and repo_exists:
            logger.info(
                f"You specified create_repo={create_repo}, but repository {repo_id} already exists. "
                "Skipping repo creation."
            )
        elif not create_repo and not repo_exists:
            raise ValueError(f"Repository {repo_id} does not exist. Set create_repo=True to create it.")
        elif create_repo and not repo_exists:
            logger.info(f"Creating/Updating HF Hub repository {repo_id}")
            api.create_repo(
                repo_id=repo_id,
                repo_type=repo_type,
                token=token,
                private=private,
                # exist_ok=exist_ok,  # not using this arg to avoid inconsistency
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
                "For easily reusing these templates, see the [documentation](https://github.com/MoritzLaurer/prompt-templates)."
            )
            card = RepoCard(repocard_text)
            card.push_to_hub(
                repo_id=repo_id,
                repo_type=repo_type,
                token=token,
                commit_message="Create/Update repo card with prompt-templates library",
                create_pr=create_pr,
                parent_commit=parent_commit,
            )
        elif not create_repo and repo_exists:
            # Update repo metadata to make prompt templates discoverable on the HF Hub
            logger.info(f"Updating HF Hub repository {repo_id} with prompt-templates library metadata.")
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
        logger.info(f"Uploading prompt template {filename} to HF Hub repository {repo_id}")
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

    def display(self, format: Literal["json", "yaml"] = "json") -> None:
        """Display the prompt configuration in the specified format.

        Examples:
            >>> from prompt_templates import TextPromptTemplate
            >>> prompt_template = TextPromptTemplate.load_from_hub(
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
        # Filter out private attributes (those starting with _)
        public_attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

        attributes = ", ".join(
            f"{key}={repr(value)[:50]}...'" if len(repr(value)) > 50 else f"{key}={repr(value)}"
            for key, value in public_attrs.items()
        )
        return f"{self.__class__.__name__}({attributes})"

    def _populate_placeholders(self, template_part: Any, user_provided_variables: Dict[str, Any]) -> Any:
        """Recursively fill placeholders in strings or nested structures like dicts or lists."""
        if isinstance(template_part, str):
            # fill placeholders in strings
            return self._populator_instance.populate(template_part, user_provided_variables)
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
            error_parts.append(f"Template extract: {template_extract}")

            raise ValueError("\n".join(error_parts))

    def _get_variables_in_template(self) -> Set[str]:
        """Get all variables used as placeholders in the template string or messages dictionary."""
        variables_in_template = set()
        if isinstance(self.template, str):
            variables_in_template = self._populator_instance.get_variable_names(self.template)
        elif isinstance(self.template, list) and any(isinstance(item, dict) for item in self.template):
            for message in self.template:
                content = message["content"]
                if isinstance(content, str):
                    variables_in_template.update(self._populator_instance.get_variable_names(content))
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
                variables.update(self._populator_instance.get_variable_names(value))
            elif isinstance(value, dict):
                variables.update(self._get_variables_in_dict(value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        variables.update(self._get_variables_in_dict(item))
        return variables

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

    def _create_populator_instance(self, populator: PopulatorType, jinja2_security_level: Jinja2SecurityLevel) -> None:
        """Create populator instance.

        Args:
            populator: Explicit populator type. Must be one of ('jinja2', 'double_brace_regex', 'single_brace_regex').
            jinja2_security_level: Security level for Jinja2 populator

        Raises:
            ValueError: If an unknown populator type is specified
        """
        self._populator_instance: TemplatePopulator

        if populator == "jinja2":
            self._populator_instance = Jinja2TemplatePopulator(security_level=jinja2_security_level)
        elif populator == "double_brace_regex":
            self._populator_instance = DoubleBracePopulator()
        elif populator == "single_brace_regex":
            self._populator_instance = SingleBracePopulator()
        else:
            raise ValueError(
                f"Unknown populator type: {populator}. Valid options are: jinja2, double_brace_regex, single_brace_regex"
            )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BasePromptTemplate):
            return False

        return (
            self.template == other.template
            and self.template_variables == other.template_variables
            and self.metadata == other.metadata
            and self.client_parameters == other.client_parameters
            and self.custom_data == other.custom_data
            and self.populator == other.populator
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
        ...     "author": "Guy van Babel"
        ... }
        >>> prompt_template = TextPromptTemplate(
        ...     template=template_text,
        ...     template_variables=template_variables,
        ...     metadata=metadata
        ... )
        >>> print(prompt_template)
        TextPromptTemplate(template='Translate the following text to {{language}}:\\n{{..., template_variables=['language', 'text'], metadata={'name': 'Simple Translator', 'description': 'A si..., client_parameters={}, custom_data={}, populator='jinja2', jinja2_security_level='standard')

        >>> # Inspect template attributes
        >>> prompt_template.template
        'Translate the following text to {{language}}:\\n{{text}}'
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
        Translate the following text to French:
        Hello world!

        Or download the same text prompt template from the Hub:
        >>> from prompt_templates import TextPromptTemplate
        >>> prompt_template_downloaded = TextPromptTemplate.load_from_hub(
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
        populator: PopulatorType = "jinja2",
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

    def populate_template(self, **user_provided_variables: Any) -> str:
        """Populate the prompt by replacing placeholders with provided values.

        Examples:
            >>> from prompt_templates import TextPromptTemplate
            >>> prompt_template = TextPromptTemplate.load_from_hub(
            ...     repo_id="MoritzLaurer/example_prompts",
            ...     filename="translate.yaml"
            ... )
            >>> prompt_template.template
            'Translate the following text to {{language}}:\\n{{text}}'
            >>> prompt = prompt_template.populate_template(
            ...     language="French",
            ...     text="Hello world!"
            ... )
            >>> print(prompt)
            Translate the following text to French:
            Hello world!

        Args:
            **user_provided_variables: The values to fill placeholders in the prompt template.

        Returns:
            str: The populated prompt string.
        """
        self._validate_user_provided_variables(user_provided_variables)
        populated_prompt = str(self._populate_placeholders(self.template, user_provided_variables))
        return populated_prompt

    def to_langchain_template(self) -> "LC_PromptTemplate":
        """Convert the TextPromptTemplate to a LangChain PromptTemplate.

        Examples:
            >>> from prompt_templates import TextPromptTemplate
            >>> prompt_template = TextPromptTemplate.load_from_hub(
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
        ...     "author": "Guido van Bossum"
        ... }
        >>> prompt_template = ChatPromptTemplate(
        ...     template=template_messages,
        ...     template_variables=template_variables,
        ...     metadata=metadata
        ... )
        >>> print(prompt_template)
        ChatPromptTemplate(template=[{'role': 'system', 'content': 'You are a coding a..., template_variables=['concept', 'programming_language'], metadata={'name': 'Code Teacher', 'description': 'A simple ..., client_parameters={}, custom_data={}, populator='jinja2', jinja2_security_level='standard')
        >>> # Inspect template attributes
        >>> prompt_template.template
        [{'role': 'system', 'content': 'You are a coding assistant who explains concepts clearly and provides short examples.'}, {'role': 'user', 'content': 'Explain what {{concept}} is in {{programming_language}}.'}]
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
        >>> # You can convert to formats used by other LLM clients like Anthropic's or Google Gemini's like this:
        >>> from prompt_templates import format_for_client
        >>> messages_anthropic = format_for_client(messages, "anthropic")
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
        >>> from prompt_templates import ChatPromptTemplate
        >>> prompt_template_downloaded = ChatPromptTemplate.load_from_hub(
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
        populator: PopulatorType = "jinja2",
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

    def populate_template(self, **user_provided_variables: Any) -> List[Dict[str, Any]]:
        """Populate the prompt template messages by replacing placeholders with provided values.

        Examples:
            >>> from prompt_templates import ChatPromptTemplate
            >>> prompt_template = ChatPromptTemplate.load_from_hub(
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
            List[Dict[str, Any]]: The populated prompt as a list in the OpenAI messages format.
        """
        self._validate_user_provided_variables(user_provided_variables)

        messages_template_populated: List[Dict[str, str]] = [
            {
                "role": str(message["role"]),
                "content": self._populate_placeholders(message["content"], user_provided_variables),
            }
            for message in self.template
        ]
        return messages_template_populated

    def create_messages(
        self, client: ClientType = "openai", **user_provided_variables: Any
    ) -> List[Dict[str, Any]] | Dict[str, Any]:
        """Convenience method that populates a prompt template and formats it for a client in one step.
        This method is only useful if your a client that does not use the OpenAI messages format, because
        populating a ChatPromptTemplate converts it into the OpenAI messages format by default.

        Examples:
            >>> from prompt_templates import ChatPromptTemplate
            >>> prompt_template = ChatPromptTemplate.load_from_hub(
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
            client (str): The client format to use ('openai', 'anthropic', 'google'). Defaults to 'openai'.
            **user_provided_variables: The variables to fill into the prompt template. For example, if your template
                expects variables like 'name' and 'age', pass them as keyword arguments.

        Returns:
            List[Dict[str, Any]] | Dict[str, Any]: A populated prompt formatted for the specified client.
        """
        if "client" in user_provided_variables:
            logger.warning(
                f"'client' was passed both as a parameter for the LLM inference client ('{client}') and in user_provided_variables "
                f"('{user_provided_variables['client']}'). The first parameter version will be used for formatting, "
                "while the second user_provided_variable version will be used in template population."
            )

        prompt = self.populate_template(**user_provided_variables)
        return format_for_client(prompt, client)

    def to_langchain_template(self) -> "LC_ChatPromptTemplate":
        """Convert the ChatPromptTemplate to a LangChain ChatPromptTemplate.

        Examples:
            >>> from prompt_templates import ChatPromptTemplate
            >>> prompt_template = ChatPromptTemplate.load_from_hub(
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
