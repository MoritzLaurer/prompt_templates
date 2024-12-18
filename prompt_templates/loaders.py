import importlib.util
import inspect
import logging
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Union

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import validate_repo_id

from .constants import VALID_PROMPT_EXTENSIONS, PopulatorType
from .prompt_templates import ChatPromptTemplate, TextPromptTemplate
from .tools import Tool
from .utils import create_yaml_handler


logger = logging.getLogger(__name__)


class PromptTemplateLoader:
    """Class for loading prompt templates from different sources.

    This class provides methods to load prompt templates from either local files or the
    Hugging Face Hub. Templates are expected to be YAML files that follow a standardized
    format for either text or chat prompts.

    Examples:
        Load a template from the Hub:
        >>> from prompt_templates import PromptTemplateLoader
        >>> prompt_template = PromptTemplateLoader.from_hub(
        ...     repo_id="MoritzLaurer/example_prompts",
        ...     filename="code_teacher.yaml"
        ... )
        >>> print(prompt_template)
        ChatPromptTemplate(template=[{'role': 'system', 'content': 'You are a coding a..., template_variables=['concept', 'programming_language'], metadata={'name': 'Code Teacher', 'description': 'A simple ..., custom_data={}, populator_type='double_brace', populator=<prompt_templates.prompt_templates.DoubleBracePopula...)
        >>> prompt_template.template
        [{'role': 'system', 'content': 'You are a coding assistant...'}, ...]
        >>> prompt_template.metadata["name"]
        'Code Teacher'

        Load a template from a local file:
        >>> prompt_template = PromptTemplateLoader.from_local("./tests/test_data/translate.yaml")
        >>> print(template)
        TextPromptTemplate(template='Translate the following text to {{language}}:\\n{{..., template_variables=['language', 'text'], metadata={'name': 'Simple Translator', 'description': 'A si..., custom_data={}, populator_type='double_brace', populator=<prompt_templates.prompt_templates.DoubleBracePopula...)
        >>> prompt_template.template
        'Translate the following text to {language}:\\n{text}'
        >>> prompt_template.template_variables
        ['language', 'text']
    """

    @classmethod
    def from_local(
        cls,
        path: Union[str, Path],
        populator: Optional[PopulatorType] = None,
        jinja2_security_level: Literal["strict", "standard", "relaxed"] = "standard",
        yaml_library: str = "ruamel",
    ) -> Union[TextPromptTemplate, ChatPromptTemplate]:
        """Load a prompt template from a local YAML file.

        Args:
            path (Union[str, Path]): Path to the YAML file containing the prompt template
            populator (Optional[PopulatorType], optional): The populator type to use among Literal["double_brace", "single_brace", "jinja2"]. Defaults to None.
            jinja2_security_level (Literal["strict", "standard", "relaxed"], optional): The security level for the Jinja2 populator. Defaults to "standard".
            yaml_library (str, optional): The YAML library to use ("ruamel" or "pyyaml"). Defaults to "ruamel".

        Returns:
            Union[TextPromptTemplate, ChatPromptTemplate]: The loaded template instance

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is not a .yaml/.yml file
            ValueError: If the YAML structure is invalid

        Examples:
            Download a text prompt template:
            >>> from prompt_templates import PromptTemplateLoader
            >>> prompt_template = PromptTemplateLoader.from_local("./tests/test_data/translate.yaml")
            >>> print(prompt_template)
            TextPromptTemplate(template='Translate the following text to {{language}}:\\n{{..., template_variables=['language', 'text'], metadata={'name': 'Simple Translator', 'description': 'A si..., custom_data={}, populator_type='double_brace', populator=<prompt_templates.prompt_templates.DoubleBracePopula...)
            >>> prompt_template.template
            'Translate the following text to {language}:\\n{text}'
            >>> prompt_template.template_variables
            ['language', 'text']
            >>> prompt_template.metadata['name']
            'Simple Translator'

            Download a chat prompt template:
            >>> prompt_template = PromptTemplateLoader.from_local("./tests/test_data/code_teacher.yaml")
            >>> print(prompt_template)
            ChatPromptTemplate(template=[{'role': 'system', 'content': 'You are a coding assistant who explains concepts clearly and provides short examples.'}, {'role': 'user', 'content': 'Explain what {concept} is in {programming_language}.'}], template_variables=['concept', 'programming_language'], metadata={'name': 'Code Teacher', 'description': 'A simple ..., custom_data={}, populator_type='double_brace', populator=<prompt_templates.prompt_templates.DoubleBracePopula...)
            >>> prompt_template.template
            [{'role': 'system', 'content': 'You are a coding assistant who explains concepts clearly and provides short examples.'}, {'role': 'user', 'content': 'Explain what {concept} is in {programming_language}.'}]
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
                    prompt_file = yaml.load(file)
                else:
                    prompt_file = yaml.safe_load(file)
        except Exception as e:
            raise ValueError(
                f"Failed to parse '{path}' as a valid YAML file. "
                f"Please ensure the file is properly formatted.\n"
                f"Error details: {str(e)}"
            ) from e

        return cls._load_template_from_yaml(
            prompt_file, populator=populator, jinja2_security_level=jinja2_security_level
        )

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        filename: str,
        repo_type: str = "dataset",
        revision: Optional[str] = None,
        populator: Optional[PopulatorType] = None,
        jinja2_security_level: Literal["strict", "standard", "relaxed"] = "standard",
        yaml_library: str = "ruamel",
    ) -> Union[TextPromptTemplate, ChatPromptTemplate]:
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
            populator (Optional[PopulatorType], optional): The populator type to use among Literal["double_brace", "single_brace", "jinja2"]. Defaults to None.
            jinja2_security_level (Literal["strict", "standard", "relaxed"], optional): The security level for the Jinja2 populator. Defaults to "standard".
            yaml_library (str, optional): The YAML library to use ("ruamel" or "pyyaml"). Defaults to "ruamel".


        Returns:
            Union[TextPromptTemplate, ChatPromptTemplate]: The loaded template instance

        Raises:
            ValueError: If repo_id format is invalid
            ValueError: If repo_type is invalid
            FileNotFoundError: If file cannot be downloaded from Hub
            ValueError: If the YAML structure is invalid

        Examples:
            Download a text prompt template:
            >>> from prompt_templates import PromptTemplateLoader
            >>> prompt_template = PromptTemplateLoader.from_hub(
            ...     repo_id="MoritzLaurer/example_prompts",
            ...     filename="translate.yaml"
            ... )
            >>> print(prompt_template)
            TextPromptTemplate(template='Translate the following text to {{language}}:\\n{{..., template_variables=['language', 'text'], metadata={'name': 'Simple Translator', 'description': 'A si..., custom_data={}, populator_type='double_brace', populator=<prompt_templates.prompt_templates.DoubleBracePopula...)
            >>> prompt_template.template
            'Translate the following text to {language}:\\n{text}'
            >>> prompt_template.template_variables
            ['language', 'text']
            >>> prompt_template.metadata['name']
            'Simple Translator'

            Download a chat prompt template:
            >>> prompt_template = PromptTemplateLoader.from_hub(
            ...     repo_id="MoritzLaurer/example_prompts",
            ...     filename="code_teacher.yaml"
            ... )
            >>> print(prompt_template)
            ChatPromptTemplate(template=[{'role': 'system', 'content': 'You are a coding assistant who explains concepts clearly and provides short examples.'}, {'role': 'user', 'content': 'Explain what {concept} is in {programming_language}.'}], template_variables=['concept', 'programming_language'], metadata={'name': 'Code Teacher', 'description': 'A simple ..., custom_data={}, populator_type='double_brace', populator=<prompt_templates.prompt_templates.DoubleBracePopula...)
            >>> prompt_template.template
            [{'role': 'system', 'content': 'You are a coding assistant who explains concepts clearly and provides short examples.'}, {'role': 'user', 'content': 'Explain what {concept} is in {programming_language}.'}]
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
                    prompt_file = yaml.load(file)
                else:
                    prompt_file = yaml.safe_load(file)
        except Exception as e:
            raise ValueError(
                f"Failed to parse '{filename}' as a valid YAML file. "
                f"Please ensure the file is properly formatted.\n"
                f"Error details: {str(e)}"
            ) from e

        return cls._load_template_from_yaml(
            prompt_file, populator=populator, jinja2_security_level=jinja2_security_level
        )

    @staticmethod
    def _load_template_from_yaml(
        prompt_file: Dict[str, Any],
        populator: Optional[PopulatorType] = None,
        jinja2_security_level: Literal["strict", "standard", "relaxed"] = "standard",
    ) -> Union[TextPromptTemplate, ChatPromptTemplate]:
        """Internal method to load a template from parsed YAML data.

        Args:
            prompt_file: Dictionary containing parsed YAML data
            populator: Optional template populator type
            jinja2_security_level: Security level for Jinja2 populator

        Returns:
            Union[TextPromptTemplate, ChatPromptTemplate]: Loaded template instance

        Raises:
            ValueError: If YAML structure is invalid
        """
        # Validate YAML structure
        if "prompt" not in prompt_file:
            raise ValueError(
                f"Invalid YAML structure: The top-level keys are {list(prompt_file.keys())}. "
                "The YAML file must contain the key 'prompt' as the top-level key."
            )

        prompt_data = prompt_file["prompt"]

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


class ToolLoader:
    """Class for loading tools from different sources.

    This class provides methods to load tool functions from either local files or the Hugging Face Hub.
    Tools are expected to be single Python functions with Google-style docstrings that specify their
    functionality, inputs, outputs, and metadata.

    Note:
        The ToolLoader class and related functionalities for working with tools is still highly experimental.

    Examples:
        Load a tool from a local file:
        >>> tool = ToolLoader.from_local("./tests/test_data/get_stock_price.py")
        >>> tool.name
        'get_stock_price'

        Load a tool from the Hub:
        >>> tool = ToolLoader.from_hub(
        ...     repo_id="MoritzLaurer/example_tools",
        ...     filename="get_stock_price.py"
        ... )
    """

    @classmethod
    def from_local(cls, path: Union[str, Path]) -> Tool:
        """Load a tool from a local Python file.
        The Python file should contain exactly one function with a Google-style docstring.

        Args:
            path (Union[str, Path]): Path to the Python file containing the tool function

        Returns:
            Tool: The loaded tool instance

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is not a .py file
            ImportError: If the module cannot be loaded
            ValueError: If file doesn't contain exactly one function or is missing docstring

        Examples:
            >>> tool = ToolLoader.from_local("./tests/test_data/get_stock_price.py")
            >>> print(tool.description)
            Retrieve stock price data for a given ticker symbol.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tool file not found: {path}")
        if not path.suffix == ".py":
            raise ValueError(f"Tool file must be a .py file, got: {path}")

        tool = cls._load_tool_from_file(path)
        cls._check_dependencies(tool)
        return tool

    @classmethod
    def from_hub(cls, repo_id: str, filename: str, repo_type: str = "dataset", revision: Optional[str] = None) -> Tool:
        """Load a tool from the Hugging Face Hub.

        Downloads and loads a tool function from a repository on the Hugging Face Hub.
        The tool file should contain exactly one function with a Google-style docstring.

        Args:
            repo_id (str): The repository ID on Hugging Face Hub (e.g., 'username/repo')
            filename (str): Name of the Python file containing the tool
            repo_type (str, optional): Type of repository. Must be one of
                ['dataset', 'model', 'space']. Defaults to "dataset"
            revision (Optional[str], optional): Git revision to download from.
                Can be a branch name, tag, or commit hash. Defaults to None

        Returns:
            Tool: The loaded tool instance

        Raises:
            ValueError: If repo_id format is invalid
            ValueError: If repo_type is invalid
            FileNotFoundError: If file cannot be downloaded from Hub
            ImportError: If the module cannot be loaded
            ValueError: If file doesn't contain exactly one function or is missing docstring

        Examples:
            >>> tool = ToolLoader.from_hub(
            ...     repo_id="MoritzLaurer/example_tools",
            ...     filename="get_stock_price.py"
            ... )
            >>> print(tool.metadata)
            {'version': '0.0.1', 'author': 'John Doe', 'requires_gpu': 'False', 'requires_api_key': 'False'}
        """
        # Validate Hub parameters
        try:
            validate_repo_id(repo_id)
        except ValueError as e:
            raise ValueError(f"Invalid repo_id format: {str(e)}") from e

        if repo_type not in ["dataset", "model", "space"]:
            raise ValueError(f"repo_type must be one of ['dataset', 'model', 'space'], got {repo_type}")

        # Ensure .py extension
        if not filename.endswith(".py"):
            filename += ".py"

        try:
            file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type, revision=revision)
        except Exception as e:
            raise FileNotFoundError(f"Failed to download tool from Hub: {str(e)}") from e

        tool = cls._load_tool_from_file(Path(file_path))
        cls._check_dependencies(tool)
        return tool

    @classmethod
    def _load_tool_from_file(cls, file_path: Path) -> Tool:
        """Internal method to load a tool from a .py file containing a single function.

        Args:
            file_path: Path to the Python file containing the tool function

        Returns:
            Tool: Loaded tool instance

        Raises:
            ImportError: If module cannot be loaded
            ValueError: If file doesn't contain exactly one function or missing docstring
        """
        # Load the module
        spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
        if not spec or not spec.loader:
            raise ImportError(f"Could not load module from {file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find the function (assuming only one function in file)
        functions = [
            obj for name, obj in inspect.getmembers(module) if inspect.isfunction(obj) and not name.startswith("_")
        ]
        if len(functions) != 1:
            raise ValueError(f"Expected exactly one function in {file_path}, found {len(functions)}")
        func = functions[0]

        # Parse the docstring
        if not func.__doc__:
            raise ValueError(f"Function {func.__name__} must have a docstring")

        # Use the function name as the tool name
        name = func.__name__

        # Extract main description (first paragraph)
        docstring = inspect.cleandoc(func.__doc__)  # Normalize indentation
        doc_parts = docstring.split("\n\n")
        description = doc_parts[0].strip()

        # Parse Google-style sections
        sections = cls._parse_docstring_sections(func.__doc__)

        # Extract dependencies from imports
        dependencies = cls._extract_tool_dependencies(file_path)

        return Tool(
            func=func,
            name=name,
            description=description,
            args_description=sections.get("args", {}),
            return_description=sections.get("returns", ""),
            raises_description=sections.get("raises", {}),
            metadata=sections.get("metadata", {}),
            dependencies=dependencies,
        )

    @staticmethod
    def _parse_docstring_sections(docstring: str) -> Dict[str, Any]:
        """Parse Google-style docstring sections.

        Args:
            docstring: The function's docstring

        Returns:
            Dict containing parsed sections (args, returns, raises, metadata)
        """
        sections: Dict[str, Any] = {"args": {}, "returns": "", "raises": {}, "metadata": {}}

        current_section = None
        for line in docstring.split("\n"):
            line = line.strip()

            # Detect section headers
            if line.startswith("Args:"):
                current_section = "args"
                continue
            elif line.startswith("Returns:"):
                current_section = "returns"
                continue
            elif line.startswith("Raises:"):
                current_section = "raises"
                continue
            elif line.startswith("Metadata:"):
                current_section = "metadata"
                continue

            # Parse section content
            if current_section == "args" and line and not line.startswith("-"):
                match = re.match(r"(\w+)\s*\(([\w\[\],\s]+)\):\s*(.+)", line)
                if match:
                    sections["args"][match.group(1)] = match.group(3).strip()

            elif current_section == "returns" and line:
                sections["returns"] += line + " "

            elif current_section == "raises" and line:
                match = re.match(r"(\w+):\s*(.+)", line)
                if match:
                    sections["raises"][match.group(1)] = match.group(2).strip()

            elif current_section == "metadata" and line.startswith("-"):
                match = re.match(r"-\s*(\w+):\s*(.+)", line)
                if match:
                    sections["metadata"][match.group(1)] = match.group(2).strip()

        # Clean up returns section
        sections["returns"] = sections["returns"].strip()

        return sections

    @staticmethod
    def _check_dependencies(tool: Tool) -> None:
        """Check if all tool dependencies are installed and warn if not."""
        uninstalled = tool.return_uninstalled_dependencies()
        if uninstalled:
            warnings.warn(
                f"Tool '{tool.name}' has uninstalled dependencies: {uninstalled}. "
                "The tool may not work correctly until these packages are installed.",
                stacklevel=2,
            )

    @staticmethod
    def _extract_tool_dependencies(file_path: Path) -> Set[str]:
        """Extract Python package dependencies from import statements.

        Args:
            file_path: Path to the Python file

        Returns:
            Set of package names that are imported
        """
        dependencies = set()
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(("import ", "from ")):
                    # Extract package name (first part of import)
                    package = line.split()[1].split(".")[0]
                    # Ignore stdlib modules
                    if package not in sys.stdlib_module_names:
                        dependencies.add(package)
        return dependencies


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


def list_tools(repo_id: str, repo_type: str = "dataset", token: Optional[str] = None) -> List[str]:
    """List available tool Python files in a Hugging Face Hub repository.

    Args:
        repo_id (str): The repository ID on Hugging Face Hub
        repo_type (str, optional): The type of repository. Defaults to "dataset"
        token (Optional[str], optional): An optional authentication token. Defaults to None

    Returns:
        List[str]: A list of Python filenames in the repository sorted alphabetically

    Examples:
        List all tools in a repository:
        >>> from prompt_templates import list_tools
        >>> files = list_tools("MoritzLaurer/example_tools")
        >>> files
        ['get_stock_price.py']

    Note:
        This function simply returns all .py file names in the repository.
        It does not validate if the files contain valid tools, which would require downloading them.
    """
    api = HfApi(token=token)
    py_files = [file for file in api.list_repo_files(repo_id, repo_type=repo_type) if file.endswith(".py")]
    return sorted(py_files)
