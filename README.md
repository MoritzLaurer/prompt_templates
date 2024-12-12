# Prompt Templates

Prompt templates have become key artifacts for researchers and practitioners working with AI. There is, however, no standardized way of sharing prompt templates. Prompts and prompt templates are shared on the Hugging Face Hub in [.txt files](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier/blob/main/utils/prompt.txt), in [HF datasets](https://huggingface.co/datasets/fka/awesome-chatgpt-prompts), as strings in [model cards](https://huggingface.co/OpenGVLab/InternVL2-8B#grounding-benchmarks), or on GitHub as [python strings](https://github.com/huggingface/cosmopedia/tree/main/prompts) embedded in scripts, in [JSON and YAML](https://github.com/hwchase17/langchain-hub/blob/master/prompts/README.md) files, or in [Jinja2](https://github.com/argilla-io/distilabel/tree/main/src/distilabel/steps/tasks/templates) files. 



## Objectives and non-objectives of this library
### Objectives
- Provide functionality for working with prompt templates locally and sharing them on the Hugging Face Hub. 
- Propose a prompt template standard through .yaml and .json files that enables modular development of complex LLM systems and is interoperable with other libraries
### Non-Objective 
- Compete with full-featured prompting libraries like [LangChain](https://github.com/langchain-ai/langchain), [ell](https://docs.ell.so/reference/index.html), etc. The objective is, instead, a simple solution for working with prompt templates locally or on the HF Hub, which is interoperable with other libraries and which the community can build upon.


## Documentation

A discussion of the standard prompt format, usage examples, the API reference etc. are available in the [docs](https://moritzlaurer.github.io/prompt_templates/).


## Quick start

Let's use this [closed_system_prompts repo](https://huggingface.co/MoritzLaurer/closed_system_prompts) of official prompts from OpenAI and Anthropic. These prompt templates have either been leaked or were shared by these LLM providers, but were originally in a non-machine-readable, non-standardized format.


#### 1. Install the library:

```bash
pip install prompt-templates
```


#### 2. List available prompts in a HF Hub repository. 

```python
from prompt_templates import list_prompt_templates
files = list_prompt_templates("MoritzLaurer/closed_system_prompts")
print(files)
# ['claude-3-5-artifacts-leak-210624.yaml', 'claude-3-5-sonnet-text-090924.yaml', 'claude-3-5-sonnet-text-image-090924.yaml', 'openai-metaprompt-audio.yaml', 'openai-metaprompt-text.yaml']
```

#### 3. Download and inspect a prompt template

```python
from prompt_templates import PromptTemplateLoader
prompt_template = PromptTemplateLoader.from_hub(
    repo_id="MoritzLaurer/closed_system_prompts",
    filename="claude-3-5-artifacts-leak-210624.yaml"
)
# Inspect template
print(prompt_template.template)
#[{'role': 'system',
#  'content': '<artifacts_info>\nThe assistant can create and reference artifacts ...'},
# {'role': 'user', 'content': '{{user_message}}'}]
# Check required template variables
print(prompt_template.template_variables)
# ['current_date', 'user_message']
print(prompt_template.metadata)
# {'source': 'https://gist.github.com/dedlim/6bf6d81f77c19e20cd40594aa09e3ecd'}
```


#### 4. Populate the template with variables
By default, the populated prompt is returned in the OpenAI messages format, which is compatible with most open-source LLM clients.

```python
messages = prompt_template.populate_template(
    user_message="Create a tic-tac-toe game for me in Python",
    current_date="Wednesday, 11 December 2024"
)
print(messages)
# PopulatedPrompt([{'role': 'system', 'content': '<artifacts_info>\nThe assistant can create and reference artifacts during conversations. Artifacts are ...'}, {'role': 'user', 'content': 'Create a tic-tac-toe game for me in Python'}])
```

#### 5. Use the populated template with any LLM client

```python
from openai import OpenAI
import os
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)
print(response.choices[0].message.content[:100], "...")
# Here's a simple text-based Tic-Tac-Toe game in Python. This code allows two players to take turns pl ...
```

```python
from huggingface_hub import InferenceClient
client = InferenceClient(api_key=os.environ.get("HF_TOKEN"))
response = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct", 
    messages=messages.to_dict(),
    max_tokens=500
)
print(response.choices[0].message.content[:100], "...")
# <antThinking>Creating a tic-tac-toe game in Python is a good candidate for an artifact. It's a self- ...
```

If you use an LLM client that expects a format different to the OpenAI messages standard, you can easily reformat the prompt for this client.

```python
from anthropic import Anthropic

messages_anthropic = messages.format_for_client(client="anthropic")

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
response = client.messages.create(
    model="claude-3-sonnet-20240229",
    system=messages_anthropic["system"],
    messages=messages_anthropic["messages"],
    max_tokens=1000
)
print(response.content[0].text[:100], "...")
# Sure, I can create a tic-tac-toe game for you in Python. Here's a simple implementation: ...
```


#### 6. Create your own prompt templates

```python
from prompt_templates import ChatPromptTemplate
messages_template = [
    {"role": "system", "content": "You are a coding assistant who explains concepts clearly and provides short examples."},
    {"role": "user", "content": "Explain what {{concept}} is in {{programming_language}}."}
]
template_variables = ["concept", "programming_language"]
metadata = {
    "name": "Code Teacher",
    "description": "A simple chat prompt for explaining programming concepts with examples",
    "tags": ["programming", "education"],
    "version": "0.0.1",
    "author": "Guido van Bossum"
}
prompt_template = ChatPromptTemplate(
    template=messages_template,
    template_variables=template_variables,
    metadata=metadata,
)

print(prompt_template)
# ChatPromptTemplate(template=[{'role': 'system', 'content': 'You are a coding a..., template_variables=['concept', 'programming_language'], metadata={'name': 'Code Teacher', 'description': 'A simple ..., client_parameters={}, custom_data={}, populator_type='double_brace', populator=<prompt_templates.prompt_templates.DoubleBracePopu...)
```

#### 7. Store or share your prompt templates
You can then store your prompt template locally or share it on the HF Hub.

```python
# save locally
prompt_template.save_to_local("./tests/test_data/code_teacher_test.yaml")
# or save it on the HF Hub
prompt_template.save_to_hub(repo_id="MoritzLaurer/example_prompts_test", filename="code_teacher_test.yaml", create_repo=True)
# CommitInfo(commit_url='https://huggingface.co/MoritzLaurer/example_prompts_test/commit/4cefd2c94f684f9bf419382f96b36692cd175e84', commit_message='Upload prompt template code_teacher_test.yaml', commit_description='', oid='4cefd2c94f684f9bf419382f96b36692cd175e84', pr_url=None, repo_url=RepoUrl('https://huggingface.co/MoritzLaurer/example_prompts_test', endpoint='https://huggingface.co', repo_type='model', repo_id='MoritzLaurer/example_prompts_test'), pr_revision=None, pr_num=None)
```


## TODO
- [ ] many things ...

