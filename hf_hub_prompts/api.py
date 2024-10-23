from .prompt_template import PromptTemplate
from huggingface_hub import hf_hub_download, HfApi
import yaml
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def download_prompt(repo_id: str, filename: str, repo_type: Optional[str] = "model") -> PromptTemplate:
    """Download a prompt from the Hugging Face Hub and create a PromptTemplate.

    Args:
        repo_id (str): The repository ID on Hugging Face Hub (e.g., 'username/repo_name').
        filename (str): The filename of the prompt YAML file.
        repo_type (Optional[str]): The type of repository to download from. Defaults to "model".

    Returns:
        PromptTemplate: A PromptTemplate object created from the downloaded prompt.

    Raises:
        ValueError: If the YAML file cannot be parsed or does not meet the expected structure.
    """
    
    if not filename.endswith((".yaml", ".yml")):
        filename += ".yaml"
    
    file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)

    try:
        with open(file_path, 'r') as file:
            prompt_file = yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse the file '{filename}' as a valid YAML file. Please ensure the file is properly formatted.\nError details: {str(e)}")

    # Validate YAML keys to enforce minimal common standard structure
    if "prompt" not in prompt_file:
        raise ValueError(
            f"Invalid YAML structure: The top-level keys are {list(prompt_file.keys())}. "
            "The YAML file must contain the key 'prompt' as the top-level key."
        )
    if "messages" not in prompt_file["prompt"]:
        raise ValueError(
            f"Invalid YAML structure: 'messages' key is missing under the 'prompt' top-level key. "
            "The library expects a 'messages' key following the OpenAI messages structure. "
            f"The current keys under 'prompt' are: {list(prompt_file['prompt'].keys())}. "
            "Please refer to the documentation for a compatible YAML example."
        )

    prompt_url = f"https://huggingface.co/{repo_id}/blob/main/{filename}"
    
    return PromptTemplate(**prompt_file["prompt"], full_yaml_content=prompt_file, prompt_url=prompt_url)
    
    
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
    yaml_files = [file for file in api.list_repo_files(repo_id, repo_type=repo_type) if file.endswith((".yaml", ".yml"))]
    return yaml_files

