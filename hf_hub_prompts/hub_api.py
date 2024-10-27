import logging
from typing import List, Optional, Union

import yaml
from huggingface_hub import HfApi, hf_hub_download

from .prompt_templates import ChatPromptTemplate, TextPromptTemplate


logger = logging.getLogger(__name__)


def download_prompt(
    repo_id: str, filename: str, repo_type: Optional[str] = "model"
) -> Union[TextPromptTemplate, ChatPromptTemplate]:
    """Download a prompt from the Hugging Face Hub and create a PromptTemplate or ChatPromptTemplate.

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
        # It's a ChatPromptTemplate
        return ChatPromptTemplate(**prompt_data, full_yaml_content=prompt_file, prompt_url=prompt_url)
    elif "template" in prompt_data:
        # It's a PromptTemplate
        return TextPromptTemplate(**prompt_data, full_yaml_content=prompt_file, prompt_url=prompt_url)
    else:
        raise ValueError(
            f"Invalid YAML structure under 'prompt': {list(prompt_data.keys())}. "
            "The YAML file must contain either 'messages' or 'template' key under 'prompt'. "
            "Please refer to the documentation for a compatible YAML example."
        )


def list_prompts(repo_id: str, repo_type: Optional[str] = "model", token: Optional[str] = None) -> List[str]:
    """List available prompt YAML files in a Hugging Face repository.

    Note:
        This function simply returns all YAML file names in the repository.
        It does not check if a file is a valid prompt, which would require downloading it.

    Args:
        repo_id (str): The repository ID on Hugging Face Hub.
        token (Optional[str]): An optional authentication token. Defaults to None.

    Returns:
        List[str]: A list of YAML filenames in the repository.
    """
    logger.info(
        "This function simply returns all YAML file names in the repository."
        "It does not check if a file is a valid prompt, which would require downloading it."
    )
    api = HfApi(token=token)
    yaml_files = [
        file for file in api.list_repo_files(repo_id, repo_type=repo_type) if file.endswith((".yaml", ".yml"))
    ]
    return yaml_files
