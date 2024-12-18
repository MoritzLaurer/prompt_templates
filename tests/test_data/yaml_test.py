"""
poetry run python tests/test_data/yaml_test.py
"""

import os
from pathlib import Path

import yaml


def read_write_yaml(input_path, output_path=None):
    # Custom representer for multiline strings
    def str_presenter(dumper, data):
        if "\n" in data:
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    # Register the custom representer
    yaml.add_representer(str, str_presenter)

    # Configure yaml settings
    yaml.Dumper.ignore_aliases = lambda *args: True

    # Read the input file
    input_path = Path(input_path)
    with input_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    # If no output path specified, use input path
    if output_path is None:
        output_path = input_path
    else:
        output_path = Path(output_path)

    # Write back to file with custom settings
    with output_path.open("w", encoding="utf-8") as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False, allow_unicode=True, indent=2)

    return data


# Usage example:
if __name__ == "__main__":
    print(os.getcwd())
    data = read_write_yaml(
        "./tests/test_data/dataset_prompts/fineweb-edu-prompt.yaml", "./tests/test_data/dataset_prompts/output.yaml"
    )
