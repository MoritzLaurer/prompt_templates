# HF Hub Prompts

A Python library for sharing and downloading prompts on the Hugging Face Hub.

## Overview

HF Hub Prompts is a Python library that makes it easy to share and download prompts using the Hugging Face Hub infrastructure. It supports both standard text prompts and chat-based prompts, with features for template variables and formatting for different LLM clients.

## Features

- Download prompts from the Hugging Face Hub
- Support for both text and chat-based prompts
- Template variable system
- Format prompts for different LLM clients (OpenAI, Anthropic)
- LangChain compatibility
- YAML-based prompt storage

## Quick Start

```python
from hf_hub_prompts import download_prompt

# Download a prompt template
prompt = download_prompt(
    repo_id="username/repo_name",
    filename="my-prompt.yaml"
)

# Use the template
populated = prompt.populate_template(
    variable1="value1",
    variable2="value2"
)

# Get the final text
text = populated.content
```