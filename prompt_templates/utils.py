from typing import Any, Union

import yaml as pyyaml
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString


def create_yaml_handler(library: str = "ruamel") -> Union[YAML, Any]:
    """Create a YAML handler with the specified configuration.
    Ruamel is the default, because it allows for better format preservation and defaults to the newer YAML 1.2.
    Pyyaml can also be used, as it can be faster and is more widely used.

    Args:
        library: The YAML library to use ("ruamel" or "pyyaml"). Defaults to "ruamel".

    Returns:
        A configured YAML handler

    Raises:
        ValueError: If an unsupported YAML library is specified
    """
    if library == "ruamel":
        yaml = YAML(typ="rt")
        yaml.preserve_quotes = True
        yaml.default_flow_style = False
        yaml.width = 120
        yaml.indent(mapping=2, sequence=4, offset=2)
        return yaml
    elif library == "pyyaml":
        return pyyaml
    else:
        raise ValueError(f"Unsupported YAML library: {library}")


def format_template_content(node: Any) -> Any:
    '''Recursively format content strings to use YAML literal block scalars.
    This is used to make the string outputs in a yaml file contain "|-",
    which makes the string behave like a """...""" block in python
    to make strings easier to read and edit.

    Args:
        node: The prompt template content to format

    Returns:
        The formatted content with literal block scalars for multiline strings
    '''
    if isinstance(node, dict):
        for key, value in node.items():
            node[key] = LiteralScalarString(value.strip()) if key in ["content", "text"] else value
        return node
    elif isinstance(node, str):
        if "\n" in node or len(node) > 80:
            return LiteralScalarString(node.strip())
        else:
            return node
    else:
        return node
