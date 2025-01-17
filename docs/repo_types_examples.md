
# Prompt templates on the HF Hub

The HF Hub is currently organized around three main repository types:

- Model repositories: Repos with model weights, tokenizers, and model configs.
- Dataset repositories: Repos with tabular datasets (mostly in parquet format). 
- Spaces repositories: Repos with hosted applications (often with code and data, which is then visualized in the Space).

Prompt templates can be saved into any of these repository types as .yaml or .json files. We recommend saving prompt templates in dataset repos by default. 



## 1. Saving collections of prompt templates as independent artifacts in dataset repos
Many prompt templates can be reused in different projects and with different models. We recommend sharing collections of reusable prompt templates in HF dataset repos, where the dataset card provides a description and usage instructions and the templates are shared as .yaml or .json files in the same repository.


<details>
  <summary>1. Example: using the <a href="https://gist.github.com/dedlim/6bf6d81f77c19e20cd40594aa09e3ecd">leaked Claude Artifacts prompt</a></summary>

#### List all prompt templates stored in a HF dataset repo
This [example HF repository](https://huggingface.co/MoritzLaurer/closed_system_prompts) 
contains leaked or released prompts from Anthropic and OpenAI. 

```python
from prompt_templates import list_prompt_templates
list_prompt_templates(repo_id="MoritzLaurer/closed_system_prompts")
# ['claude-3-5-artifacts-leak-210624.yaml', 'claude-3-5-sonnet-text-090924.yaml', 'claude-3-5-sonnet-text-image-090924.yaml', 'openai-metaprompt-audio.yaml', 'openai-metaprompt-text.yaml']
```

#### Download a specific prompt template
Here, we download the leaked prompt for Claude-3.5 Sonnet for creating Artifacts. 

```python
from prompt_templates import ChatPromptTemplate
prompt_template = ChatPromptTemplate.load_from_hub(
    repo_id="MoritzLaurer/closed_system_prompts",
    filename="claude-3-5-artifacts-leak-210624.yaml"
)

print(prompt_template)
# ChatPromptTemplate(template=[{'role': 'system', 'content': '<artifacts_info> The assistant can create and reference artifacts during conversations. Artifacts are ... Claude is now being connected with a human.'}, {'role': 'user', 'content': '{user_message}'}], template_variables=['current_date', 'user_message'], metadata=[{'source': 'https://gist.github.com/dedlim/6bf6d81f77c19e20cd40594aa09e3ecd'}])
```

Prompt templates are downloaded as either `ChatPromptTemplate` or `TextPromptTemplate` classes. This class makes it easy to populate a prompt template and convert it into a format that's compatible with different LLM clients. The type is automatically determined based on whether the YAML contains a simple string (TextPromptTemplate) or a list of dictionaries following the OpenAI messages format (ChatPromptTemplate).

#### Populate and use the prompt template
We can then populate the prompt template for a specific use-case.

```python
# Check which variables the prompt template requires
print(prompt_template.template_variables)
# ['current_date', 'user_message']

user_message = "Create a simple calculator web application"
messages = prompt_template.populate_template(
    user_message=user_message, 
    current_date="Monday 21st October 2024", 
)

# The default output is in the OpenAI messages format. We can easily reformat it for another client.
from prompt_templates import format_for_client

messages_anthropic = format_for_client(messages, client="anthropic")

```

The output is a list or a dictionary in the format expected by the specified LLM client. For example, OpenAI expects a list of message dictionaries, while Anthropic expects a dictionary with "system" and "messages" keys.

```python
#!pip install anthropic
from anthropic import Anthropic
client_anthropic = Anthropic()

response = client_anthropic.messages.create(
    model="claude-3-5-sonnet-20240620",
    system=messages_anthropic["system"],
    messages=messages_anthropic["messages"],
    max_tokens=4096,
)
```

</details>


<details>
  <summary>2. Example: <a href="https://arxiv.org/pdf/2410.12784">JudgeBench paper</a> prompts</summary>
The paper "JudgeBench: A Benchmark for Evaluating LLM-Based Judges" (<a href="https://arxiv.org/pdf/2410.12784">paper</a>) collects several prompts for using LLMs to evaluate unstructured LLM outputs. After copying them into a HF Hub dataset repo in the standardized YAML format, they can be directly loaded and populated.

```python
from prompt_templates import ChatPromptTemplate
prompt_template = ChatPromptTemplate.load_from_hub(
  repo_id="MoritzLaurer/prompts_from_papers", 
  filename="judgebench-vanilla-prompt.yaml"
)

```
</details>


<details>
  <summary>3. Example: Sharing <a href="https://huggingface.co/MoritzLaurer/closed_system_prompts">closed system prompts</a></summary>
The community has extracted system prompts from closed API providers like OpenAI or Anthropic and these prompts are unsystematically shared via GitHub, Reddit etc. (e.g. <a href="https://gist.github.com/dedlim/6bf6d81f77c19e20cd40594aa09e3ecd">Anthropic Artifacts prompt</a>). Some API providers have also started sharing their system prompts on their websites in non-standardized HTML (<a href="https://docs.anthropic.com/en/release-notes/system-prompts#sept-9th-2024">Anthropic</a>, <a href="https://platform.openai.com/docs/guides/prompt-generation">OpenAI</a>). To simplify to use of these prompts, they can be shared in a <a href="https://huggingface.co/MoritzLaurer/closed_system_prompts">HF Hub dataset repo</a> as standardized YAML files.  


```python
from prompt_templates import list_prompt_templates, ChatPromptTemplate
list_prompt_templates(repo_id="MoritzLaurer/closed_system_prompts")
# out: ['claude-3-5-artifacts-leak-210624.yaml', 'claude-3-5-sonnet-text-090924.yaml', 'claude-3-5-sonnet-text-image-090924.yaml', 'openai-metaprompt-audio.yaml', 'openai-metaprompt-text.yaml']

prompt_template = ChatPromptTemplate.load_from_hub(
  repo_id="MoritzLaurer/closed_system_prompts", 
  filename="openai-metaprompt-text.yaml"
)
```
</details>



## 2. Attaching prompt templates to models
Some open-weight LLMs have been trained to exhibit specific behaviours with specific prompt templates.
The vision language model [InternVL2](https://huggingface.co/collections/OpenGVLab/internvl-20-667d3961ab5eb12c7ed1463e) was trained to predict bounding boxes for manually specified areas with a special prompt template; 
the VLM [Molmo](https://huggingface.co/collections/allenai/molmo-66f379e6fe3b8ef090a8ca19) was trained to predict point coordinates of objects of images with a special prompt template; etc.

These prompt templates are currently either mentioned unsystematically in model cards or need to be tracked down on github or paper appendices by users. 

`prompt_templates` proposes to share these types of prompt templates in YAML or JSON files in the model repository together with the model weights. 

<details>
  <summary>1. Example: Sharing the <a href="https://huggingface.co/MoritzLaurer/open_models_special_prompts">InternVL2 special task prompt templates</a></summary>

```python
# download image prompt template
from prompt_templates import ChatPromptTemplate
prompt_template = ChatPromptTemplate.load_from_hub(
  repo_id="MoritzLaurer/open_models_special_prompts", 
  filename="internvl2-bbox-prompt.yaml"
)

# populate prompt
image_url = "https://unsplash.com/photos/ZVw3HmHRhv0/download?ixid=M3wxMjA3fDB8MXxhbGx8NHx8fHx8fDJ8fDE3MjQ1NjAzNjl8&force=true&w=1920"
region_to_detect = "the bird"
messages = prompt_template.populate_template(image_url=image_url, region_to_detect=region_to_detect)

print(messages)
#[{'role': 'user',
#  'content': [{'type': 'image_url',
#    'image_url': {'url': 'https://unsplash.com/photos/ZVw3HmHRhv0/download?ixid=M3wxMjA3fDB8MXxhbGx8NHx8fHx8fDJ8fDE3MjQ1NjAzNjl8&force=true&w=1920'}},
#   {'type': 'text',
#    'text': 'Please provide the bounding box coordinate of the region this sentence describes: <ref>the bird</ref>'}]}]
```

This populated prompt can then directly be used in a vLLM container, e.g. hosted on HF Inference Endpoints, using the OpenAI messages format and client.

```py
from openai import OpenAI
import os

ENDPOINT_URL = "https://tkuaxiztuv9pl4po.us-east-1.aws.endpoints.huggingface.cloud" + "/v1/" 

# initialize the OpenAI client but point it to an endpoint running vLLM or TGI
client = OpenAI(
    base_url=ENDPOINT_URL, 
    api_key=os.getenv("HF_TOKEN")
)

response = client.chat.completions.create(
    model="/repository", # with vLLM deployed on HF endpoint, this needs to be /repository since there are the model artifacts stored
    messages=messages,
)

response.choices[0].message.content
# out: 'the bird[[54, 402, 515, 933]]'
```
</details>



## 3. Attaching prompt templates to datasets
LLMs are increasingly used to help create datasets, for example for quality filtering or synthetic text generation.
The prompt templates used for creating a dataset are currently unsystematically shared on GitHub ([example](https://github.com/huggingface/cosmopedia/tree/main/prompts)), 
referenced in dataset cards ([example](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu#annotation)), or stored in .txt files ([example](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier/blob/main/utils/prompt.txt)), 
hidden in paper appendices or not shared at all. 
This makes reproducibility unnecessarily difficult.

To facilitate reproduction, these dataset prompt templates can be shared in YAML files in HF dataset repositories together with metadata on generation parameters, model_ids etc. 


<details>
  <summary>1. Example: the <a href="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu">FineWeb-edu</a> prompt</summary>
The FineWeb-Edu dataset was created by prompting `Meta-Llama-3-70B-Instruct` to score the educational value of web texts.
The authors <a href="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu#annotation">provide the prompt template</a> in a <a href="https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier/blob/main/utils/prompt.txt">.txt</a> file.

When provided in a YAML/JSON file in the dataset repo, the prompt template can easily be loaded and supplemented with client_parameters like the model_id or generation parameters for easy reproducibility. 
See this <a href="https://huggingface.co/datasets/MoritzLaurer/dataset_prompts">example dataset repository</a>


```python
from prompt_templates import ChatPromptTemplate
import torch
from transformers import pipeline

prompt_template = ChatPromptTemplate.load_from_hub(
  repo_id="MoritzLaurer/dataset_prompts", 
  filename="fineweb-edu-prompt.yaml", 
)

# populate the prompt
text_to_score = "The quick brown fox jumps over the lazy dog"
messages = prompt_template.populate_template(text_to_score=text_to_score)

# test prompt with local llama
model_id = "meta-llama/Llama-3.2-1B-Instruct"  # prompt was original created for meta-llama/Meta-Llama-3-70B-Instruct

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

outputs = pipe(
    messages,
    max_new_tokens=512,
)

print(outputs[0]["generated_text"][-1])
```

</details>


<details>
  <summary>2. Example: the <a href="https://huggingface.co/collections/HuggingFaceTB/cosmopedia-65d4e44c693d9451ce4344d6">Cosmopedia dataset</a></summary>
Cosmopedia is a dataset of synthetic textbooks, blogposts, stories, posts and WikiHow articles generated by Mixtral-8x7B-Instruct-v0.1.
The dataset shares it's prompt templates on <a href="https://github.com/huggingface/cosmopedia/tree/main/prompts">GitHub</a>
with a <a href="https://github.com/huggingface/cosmopedia/blob/main/prompts/auto_math_text/build_science_prompts.py">custom build logic</a>.
The prompts are not available in the <a href="https://huggingface.co/datasets/HuggingFaceTB/cosmopedia/tree/main">HF dataset repo</a>

The prompts could be directly added to the dataset repository in the standardized YAML/JSON format. 

</details>



## 4. Attaching prompt templates to HF Spaces

[TODO: create example]




