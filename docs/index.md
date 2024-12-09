# Prompt templates on the Hugging Face Hub

Prompt templates have become key artifacts for researchers and practitioners working with AI. There is, however, no standardized way of sharing prompt templates. Prompts and prompt templates are shared on the HF Hub in [.txt files](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier/blob/main/utils/prompt.txt), in [HF datasets](https://huggingface.co/datasets/fka/awesome-chatgpt-prompts), as strings in [model cards](https://huggingface.co/OpenGVLab/InternVL2-8B#grounding-benchmarks), or on GitHub as [python strings](https://github.com/huggingface/cosmopedia/tree/main/prompts), in [JSON, YAML](https://github.com/hwchase17/langchain-hub/blob/master/prompts/README.md), or in [Jinja2](https://github.com/argilla-io/distilabel/tree/main/src/distilabel/steps/tasks/templates). 



## Objectives and non-objectives of this library
### Objectives
1. Provide a Python library that simplifies and standardises the sharing of prompt templates on the Hugging Face Hub.
2. Start an open discussion on the best way of standardizing and encouraging the sharing of prompt templates on the HF Hub, building upon the HF Hub's existing repository types and ensuring interoperability with other prompt-related libraries.
### Non-Objectives: 
- Compete with full-featured prompting libraries like [LangChain](https://github.com/langchain-ai/langchain), [ell](https://docs.ell.so/reference/index.html), etc. The objective is, instead, a simple solution for sharing prompt templates on the HF Hub, which is compatible with other libraries and which the community can build upon. 


## Quick start
Install the package:

```bash
pip install hf-hub-prompts
```


### Basic usage

```python
>>> # 1. List available prompts in a Hub repository:
>>> from hf_hub_prompts import list_prompt_templates
>>> files = list_prompt_templates("MoritzLaurer/example_prompts")
>>> files
['code_teacher.yaml', 'translate.yaml']

>>> # 2. Download a prompt template:
>>> from hf_hub_prompts import PromptTemplateLoader
>>> prompt_template = PromptTemplateLoader.from_hub(
...     repo_id="MoritzLaurer/example_prompts",
...     filename="code_teacher.yaml"
... )

>>> # 3. Inspect the template:
>>> prompt_template.template
[{'role': 'system', 'content': 'You are a coding assistant who explains concepts clearly and provides short examples.'}, {'role': 'user', 'content': 'Explain what {concept} is in {programming_language}.'}]
>>> # Check required template variables
>>> prompt_template.template_variables
['concept', 'programming_language']

>>> # 4. Populate the template with variables
>>> messages = prompt_template.create_messages(
...     concept="list comprehension",
...     programming_language="Python"
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

