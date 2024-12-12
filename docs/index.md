# Prompt templates on the Hugging Face Hub

Prompt templates have become key artifacts for researchers and practitioners working with AI. There is, however, no standardized way of sharing prompt templates. Prompts and prompt templates are shared on the HF Hub in [.txt files](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier/blob/main/utils/prompt.txt), in [HF datasets](https://huggingface.co/datasets/fka/awesome-chatgpt-prompts), as strings in [model cards](https://huggingface.co/OpenGVLab/InternVL2-8B#grounding-benchmarks), or on GitHub as [python strings](https://github.com/huggingface/cosmopedia/tree/main/prompts) embedded in scripts, in [JSON, YAML](https://github.com/hwchase17/langchain-hub/blob/master/prompts/README.md), or in [Jinja2](https://github.com/argilla-io/distilabel/tree/main/src/distilabel/steps/tasks/templates) files. 



## Objectives and non-objectives of this library
### Objectives
1. Provide a Python library that simplifies and standardises the sharing of prompt templates locally or on the Hugging Face Hub.
2. Start an open discussion on the best way of standardizing and encouraging the sharing of prompt templates
### Non-Objectives: 
- Compete with full-featured prompting libraries like [LangChain](https://github.com/langchain-ai/langchain), [ell](https://docs.ell.so/reference/index.html), etc. The objective is, instead, a simple solution for sharing prompt templates locally or on the HF Hub, which is interoperable with other libraries and which the community can build upon. 


## Quick start
Install the package:

```bash
pip install hf-hub-prompts
```


### Basic usage


1. List available prompts in a HF Hub repository (e.g this [repo](https://huggingface.co/MoritzLaurer/closed_system_prompts)):
```python
>>> from hf_hub_prompts import list_prompt_templates
>>> files = list_prompt_templates("MoritzLaurer/closed_system_prompts")
>>> files
['claude-3-5-artifacts-leak-210624.yaml', 'claude-3-5-sonnet-text-090924.yaml', 'claude-3-5-sonnet-text-image-090924.yaml', 'openai-metaprompt-audio.yaml', 'openai-metaprompt-text.yaml']
```

2. Download and inspect a prompt template
```python
>>> from hf_hub_prompts import PromptTemplateLoader
>>> prompt_template = PromptTemplateLoader.from_hub(
...     repo_id="MoritzLaurer/closed_system_prompts",
...     filename="claude-3-5-artifacts-leak-210624.yaml"
... )
>>> # Inspect template
>>> prompt_template.template
[{'role': 'system',
  'content': '<artifacts_info>\nThe assistant can create and reference artifacts ...'},
 {'role': 'user', 'content': '{{user_message}}'}]
>>> # Check required template variables
>>> prompt_template.template_variables
['current_date', 'user_message']
>>> prompt_template.metadata
{'source': 'https://gist.github.com/dedlim/6bf6d81f77c19e20cd40594aa09e3ecd'}
```


3. Populate the template with variables
```python
>>> messages = prompt_template.create_messages(
...     user_message="Create a tic-tac-toe game for me in Python",
...     current_date="Python"
... )
>>> # By default, the populated prompt is in the OpenAI messages format
>>> # which is compatible with many open-source LLM clients
>>> messages
[{'role': 'system', 'content': 'You are a coding assistant who explains concepts clearly and provides short examples.'}, {'role': 'user', 'content': 'Explain what list comprehension is in Python.'}]

>>> # You can also format for other clients, e.g. Anthropic
>>> messages_anthropic = prompt_template.create_messages(
...     client="anthropic",
...     concept="list comprehension",
...     programming_language="Python"
... )
>>> messages_anthropic
{'system': 'You are a coding assistant who explains concepts clearly and provides short examples.', 'messages': [{'role': 'user', 'content': 'Explain what list comprehension is in Python.'}]}
```

