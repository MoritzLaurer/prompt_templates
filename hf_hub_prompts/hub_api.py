import logging
from typing import List, Optional, Union

import yaml
from huggingface_hub import HfApi, hf_hub_download

from .prompt_templates import ChatPromptTemplate, TextPromptTemplate


logger = logging.getLogger(__name__)


def download_prompt_template(
    repo_id: str, filename: str, repo_type: Optional[str] = "model"
) -> Union[TextPromptTemplate, ChatPromptTemplate]:
    """Download a prompt template from the Hugging Face Hub.

    Examples:
        Download and use a text prompt template:
        >>> from hf_hub_prompts import download_prompt_template
        >>> # Download translation prompt
        >>> prompt_template = download_prompt_template(
        ...     repo_id="MoritzLaurer/example_prompts",
        ...     filename="translate.yaml"
        ... )
        >>> # Inspect the template
        >>> prompt_template.template
        'Translate the following text to {language}:\\n{text}'
        >>> prompt_template.input_variables
        ['language', 'text']
        >>> prompt_template.metadata['name']
        'Simple Translator'

        Download and use a chat prompt template:
        >>> # Downloadas code teaching prompt
        >>> prompt_template = download_prompt_template(
        ...     repo_id="MoritzLaurer/example_prompts",
        ...     filename="code_teacher.yaml"
        ... )
        >>> # Inspect the template
        >>> prompt_template.messages
        [{'role': 'system', 'content': 'You are a coding assistant who explains concepts clearly and provides short examples.'}, {'role': 'user', 'content': 'Explain what {concept} is in {programming_language}.'}]
        >>> prompt_template.input_variables
        ['concept', 'programming_language']
        >>> prompt_template.metadata['version']
        '0.0.1'

    Args:
        repo_id (str): The repository ID on Hugging Face Hub (e.g., 'username/repo_name').
        filename (str): The filename of the prompt YAML file.
        repo_type (Optional[str]): The type of repository to download from. Defaults to "model".

    Returns:
        BasePromptTemplate: The appropriate template type based on YAML content:
            - TextPromptTemplate: If YAML contains a 'template' key
            - ChatPromptTemplate: If YAML contains a 'messages' key

    Raises:
        ValueError: If the YAML file cannot be parsed or does not meet the expected structure.
    """
    if not filename.endswith((".yaml", ".yml")):
        filename += ".yaml"

    file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)

    try:
        with open(file_path, "r") as file:
            prompt_file = yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise ValueError(
            f"Failed to parse the file '{filename}' as a valid YAML file. "
            f"Please ensure the file is properly formatted.\n"
            f"Error details: {str(e)}"
        ) from e

    # Validate YAML keys to enforce minimal common standard structure
    if "prompt" not in prompt_file:
        raise ValueError(
            f"Invalid YAML structure: The top-level keys are {list(prompt_file.keys())}. "
            "The YAML file must contain the key 'prompt' as the top-level key."
        )

    prompt_data = prompt_file["prompt"]
    prompt_url = f"https://huggingface.co/{repo_id}/blob/main/{filename}"

    # Determine which PromptTemplate class to instantiate
    if "messages" in prompt_data:
        return ChatPromptTemplate(prompt_data=prompt_data, prompt_url=prompt_url)
    elif "template" in prompt_data:
        return TextPromptTemplate(prompt_data=prompt_data, prompt_url=prompt_url)
    else:
        raise ValueError(
            f"Invalid YAML structure under 'prompt' key: {list(prompt_data.keys())}. "
            "The YAML file must contain either 'messages' or 'template' key under 'prompt'. "
            "Please refer to the documentation for a compatible YAML example."
        )


def list_prompt_templates(repo_id: str, repo_type: Optional[str] = "model", token: Optional[str] = None) -> List[str]:
    """List available prompt template YAML files in a Hugging Face Hub repository.

    Examples:
        List all prompt templates in a repository:
        >>> from hf_hub_prompts import list_prompt_templates
        >>> files = list_prompt_templates("MoritzLaurer/example_prompts")
        >>> files
        ['code_teacher.yaml', 'translate.yaml']

    Note:
        This function simply returns all YAML file names in the repository.
        It does not validate if the files contain valid prompt templates, which would require downloading them.

    Args:
        repo_id (str): The repository ID on Hugging Face Hub.
        repo_type (Optional[str]): The type of repository. Defaults to "model".
        token (Optional[str]): An optional authentication token. Defaults to None.

    Returns:
        List[str]: A list of YAML filenames in the repository sorted alphabetically.
    """
    logger.info(
        "This function simply returns all YAML file names in the repository. "
        "It does not validate if the files contain valid prompt templates, which would require downloading them."
    )
    api = HfApi(token=token)
    yaml_files = [
        file for file in api.list_repo_files(repo_id, repo_type=repo_type) if file.endswith((".yaml", ".yml"))
    ]
    return sorted(yaml_files)
