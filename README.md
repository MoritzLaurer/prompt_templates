
# HF Hub Prompts 
Prompts have become a key artifact for researchers and practitioners working with AI. `hf_hub_prompts` is a light-weight repo on sharing prompts on the Hugging Face Hub.

## Objectives and Non-Objectives of this repo
### Objectives
1. Provide a Python library that simplifies and standardises the sharing of prompts on the Hugging Face Hub.
2. Start an open discussion on the best way of standardizing and encouraging the sharing of prompts on the HF Hub, building upon the HF Hub's existing repository abstractions and ensuring interoperability with other prompt-related libraries.
### Non-Objectives: 
- Compete with full-featured prompting libraries like [LangChain](https://github.com/langchain-ai/langchain), [ell](https://docs.ell.so/reference/index.html), etc. The objective is, instead, a simple solution which is compatible with other libraries and which the community can build upon. 


## Installation and Example Usage

`pip install hf_hub_prompts`

### List all prompt templates stored in a HF Hub repo
This [example HF repository](https://huggingface.co/MoritzLaurer/closed_system_prompts) contains leaked or released prompts from Anthropic and OpenAI. 

```python
from hf_hub_prompts import list_prompts
list_prompts(repo_id="MoritzLaurer/closed_system_prompts")
# ['claude-3-5-artifacts-leak-210624.yaml', 'claude-3-5-sonnet-text-090924.yaml', 'claude-3-5-sonnet-text-image-090924.yaml', 'jokes-prompt.yaml', 'openai-metaprompt-audio.yaml', 'openai-metaprompt-text.yaml']
```

### Download a specific prompt template
Here, we download the leaked prompt for Claude-3.5 Sonnet for creating Artifacts. 

```python
from hf_hub_prompts import download_prompt
prompt_template = download_prompt(
    repo_id="MoritzLaurer/closed_system_prompts",
    filename="claude-3-5-artifacts-leak-210624.yaml"
)
print(prompt_template)
# PromptTemplate(messages=[{'role': 'system', 'content': '<artifacts_info> The assistant can create and reference artifacts during conversations. Artifacts are ... Claude is now being connected with a human.'}, {'role': 'user', 'content': '{user_message}'}], 'input_variables': ['current_date', 'user_message'], 'metadata': [{'source': 'https://gist.github.com/dedlim/6bf6d81f77c19e20cd40594aa09e3ecd'}]}})
```

Prompts are downloaded as `PromptTemplate` classes. This class makes it easy to populate a prompt template and convert it into a format that's compatible with different LLM clients. The class can also be handled like a simple Python dictionary, e.g. `prompt_template["prompt"]["messages"]`.

### Populate and use the prompt template
With the `format_messages` method, we can then populate the prompt template for a specific use-case.

```python
user_message = "Create a simple calculator web application"
messages_anthropic = prompt_template.format_messages(
    user_message=user_message, 
    current_date="Monday 21st October 2024", 
    client="anthropic"
)
```

The output is always a simple Python dictionary in the format expected by the specified LLM client.  
```python
client_anthropic = Anthropic()

response = client_anthropic.messages.create(
    model="claude-3-5-sonnet-20240620",
    system=messages_anthropic["system"],
    messages=messages_anthropic["messages"],
    max_tokens=4096,
)
```



## Main use-case scenarios
Prompts have become a key artifact produced by researchers and practitioners working with AI. 
The main use-cases for sharing and reusing these artifacts on the HF Hub are:

### 1. Sharing prompts as independent research artifacts (without model weights)
Many prompts can be reused with several different models and are not linked to specific model weights.
These prompts can be shared via a HF Model repository, where the model card provides a description and usage instructions, and prompts are shared via .yaml files in the same repository.

<details>
  <summary>1. Example: <a href="https://arxiv.org/pdf/2410.12784">JudgeBench paper</a> prompts</summary>
The paper "JudgeBench: A Benchmark for Evaluating LLM-Based Judges" (<a href="https://arxiv.org/pdf/2410.12784">paper</a>) collects several prompts for using LLMs to evaluate unstructured LLM outputs. After copying them into a <a href="https://huggingface.co/MoritzLaurer/judgebench-prompts">HF Hub model repo</a> in the standardized YAML format, they can be directly loaded and populated.

```python
prompt_template = download_prompt(repo_id="MoritzLaurer/judgebench-prompts", filename="vanilla-prompt.yaml")
```
</details>


<details>
  <summary>2. Example: Sharing <a href="https://huggingface.co/MoritzLaurer/closed_system_prompts">closed system prompts</a></summary>
The community has extracted system prompts from closed API providers like OpenAI or Anthropic and these prompts are unsystematically shared via GitHub, Reddit etc. (e.g. <a href="https://gist.github.com/dedlim/6bf6d81f77c19e20cd40594aa09e3ecd">Anthropic Artifacts prompt</a>). Some API providers have also started sharing their system prompts on their websites in non-standardized HTML (<a href="https://docs.anthropic.com/en/release-notes/system-prompts#sept-9th-2024">Anthropic</a>, <a href="https://platform.openai.com/docs/guides/prompt-generation">OpenAI</a>). To simplify to use of these prompts, they can be shared in a <a href="https://huggingface.co/MoritzLaurer/closed_system_prompts">HF Hub model repo</a> as standardized YAML files.  


```python
from hf_hub_prompts import list_prompts, download_prompt
list_prompts(repo_id="MoritzLaurer/closed_system_prompts")
# out: ['claude-3-5-artifacts-leak-210624.yaml', 'claude-3-5-sonnet-text-090924.yaml', 'claude-3-5-sonnet-text-image-090924.yaml', 'jokes-prompt.yaml', 'openai-metaprompt-audio.yaml', 'openai-metaprompt-text.yaml']

prompt_template = download_prompt(repo_id="MoritzLaurer/closed_system_prompts", filename="openai-metaprompt-text.yaml")
```
</details>



### 2. Sharing prompts together with model weights
Some open-weight LLMs have been trained to exhibit specific behaviours with specific prompts.
The vision language model [InternVL2](https://huggingface.co/collections/OpenGVLab/internvl-20-667d3961ab5eb12c7ed1463e) was trained to predict bounding boxes for manually specified areas with a special prompt; 
the VLM [Molmo](https://huggingface.co/collections/allenai/molmo-66f379e6fe3b8ef090a8ca19) was trained to predict point coordinates of objects of images with a special prompt; etc.

These prompts are currently either mentioned unsystematically in model cards or need to be tracked down on github or paper appendices by users. 

`hf_hub_prompts` proposes to share these types of prompts in YAML files in the model repository together with the model weights. 

<details>
  <summary>1. Example: Sharing the <a href="https://huggingface.co/MoritzLaurer/open_models_special_prompts">InternVL2 special task prompts</a></summary>

```python
from hf_hub_prompts import download_prompt

# download image prompt template
prompt_template = download_prompt(repo_id="MoritzLaurer/open_models_special_prompts", filename="internvl2-bbox-prompt.yaml")

# populate prompt
image_url = "https://unsplash.com/photos/ZVw3HmHRhv0/download?ixid=M3wxMjA3fDB8MXxhbGx8NHx8fHx8fDJ8fDE3MjQ1NjAzNjl8&force=true&w=1920"
region_to_detect = "the bird"
messages = prompt_template.format_messages(image_url=image_url, region_to_detect=region_to_detect, client="openai")

print(messages)
#[{'role': 'user',
#  'content': [{'type': 'image_url',
#    'image_url': {'url': 'https://unsplash.com/photos/ZVw3HmHRhv0/download?ixid=M3wxMjA3fDB8MXxhbGx8NHx8fHx8fDJ8fDE3MjQ1NjAzNjl8&force=true&w=1920'}},
#   {'type': 'text',
#    'text': 'Please provide the bounding box coordinate of the region this sentence describes: <ref>the bird</ref>'}]}]
```

This prompt can then directly be used in a vLLM container, e.g. hosted on HF Inference Endpoints, using the OpenAI messages format and client.

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



### 3. Attaching prompts to datasets
LLMs are increasingly used to help create datasets, for example for quality filtering or synthetic text generation.
The prompts used for creating a dataset are currently unsystematically shared on GitHub ([example](https://github.com/huggingface/cosmopedia/tree/main/prompts)), 
referenced in dataset cards ([example](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu#annotation)), or stored in .txt files ([example](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier/blob/main/utils/prompt.txt)), 
hidden in paper appendices or not shared at all. 
This makes reproducibility unnecessarily difficult.

To facilitate reproduction, these dataset prompts can be shared in YAML files in HF dataset repositories together with metadata on generation parameters, model_ids etc. 


<details>
  <summary>1. Example: the <a href="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu">FineWeb-edu</a> prompt</summary>
The FineWeb-Edu dataset was created by prompting `Meta-Llama-3-70B-Instruct` to score the educational value of web texts.
The authors <a href="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu#annotation">provide the prompt</a> in a .txt file.

When provided in a YAML file in the dataset repo, the prompt can easily be loaded and supplemented with metadata
like the model_id or generation parameters for easy reproducibility. 
See this <a href="https://huggingface.co/datasets/MoritzLaurer/dataset_prompts">example dataset repository</a>


```python
from hf_hub_prompts import download_prompt
import torch
from transformers import pipeline

prompt_template = download_prompt(repo_id="MoritzLaurer/dataset_prompts", filename="fineweb-edu-prompt.yaml", repo_type="dataset")

# populate the prompt
text_to_score = "The quick brown fox jumps over the lazy dog"
messages = prompt_template.format_messages(text_to_score=text_to_score)

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
The dataset shares it's prompts on <a href="https://github.com/huggingface/cosmopedia/tree/main/prompts">GitHub</a>
with a <a href="https://github.com/huggingface/cosmopedia/blob/main/prompts/auto_math_text/build_science_prompts.py">custom build logic</a>.
The prompts are not available in the <a href="https://huggingface.co/datasets/HuggingFaceTB/cosmopedia/tree/main">HF dataset repo</a>

The prompts could be directly added to the dataset repository in the standardized YAML format. 

</details>



## The standardized YAML prompt template format
The library expects prompts to be stored in YAML files in any HF Hub repository. See the `Files` tab in these repos for [open-weight model prompts](https://huggingface.co/MoritzLaurer/open_models_special_prompts), [closed-model prompts](https://huggingface.co/MoritzLaurer/closed_system_prompts), or [dataset prompts](https://huggingface.co/datasets/MoritzLaurer/dataset_prompts).

The YAML files must follow the following structure:
- Top-level key (required): `prompt`. 
- Second-level key (required): *Either* `messages` *or* `template`. If `messages`, the prompt template must be provded as a list of dictionaries following the OpenAI messages format. This format is recommended for use with LLM APIs or inference containers.  If `template`, the prompt must be provided as a single string. 
- Second-level keys (optional): (1) `input_variables`: an optional list of variables for populating the prompt template. This is also used for input validation; (2) `metadata`: Other information, such as the source, date, author etc.; (3) Any other key of relevance, such as `client_settings` with parameters for reproducibility with a specific inference client, or `metrics` form evaluations on specific datasets.

This structure is inspired by the LangChain [PromptTemplate](https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.prompt.PromptTemplate.html) and [ChatPromptTemplate](https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html). 



## Reflection: Sharing prompts as YAML files, or jinja2 templates, or as HF datasets?

### Pro/Con prompts as YAML files
- Existing prompt hubs use YAML: [LangChain Hub](https://smith.langchain.com/hub); [Haystack Prompt Hub](https://haystack.deepset.ai/blog/share-and-use-prompt-with-prompthub)
- YAML is the standard for working with prompts in production settings in my experience with practitioners. See also [this discussion](https://github.com/langchain-ai/langchain/discussions/21672).
- Managing individual prompts in separate YAML files makes each prompt a separate file unit. 
    - This makes it e.g. easier to add metadata and production-relevant information in the respective prompt YAML file.
    - Prompts in individual YAML files also enables users to add individual prompts into any HF repo abstraction (Model, Space, Dataset), while datasets always have to be their own abstraction. 

### Pro/Con Jinja2 files
- has more rich functionality for populating templates than YAML
- allows arbitrary code execution and is less safe
- can be harder to read for beginners

### Pro/Con prompts as datasets
- Some prompt datasets like [awesome-chatgpt-prompts](https://huggingface.co/datasets/fka/awesome-chatgpt-prompts) have received many likes on HF
- The dataset viewer allows for easy and quick visualization
- tabular format is not ideal for prompts and is not standard among practitioners


## Standardized loading of YAML files
- Challenges: 
    - good prompt template classes e.g. from langchain have additional functionality like async formatting. The objective is not to reproduce this. People can use the langchain chat template if they want. 
    - How to handle tools?
    - how to handle multi-agent systems?




### Notes on compatibility with `transformers`
- `transformers` provides partial prompt input standardization via chat_templates following the OpenAI messages format:
    - The [simplest use](https://huggingface.co/docs/transformers/en/conversations) is via the text-generation pipeline
    - See also details on [chat_templates](https://huggingface.co/docs/transformers/main/en/chat_templating): 
- Limitations: 
    - VLMs require special pre-processors that are not directly compatible with the standardized messages format (?). And new VLMs like [InternVL](https://huggingface.co/OpenGVLab/InternVL2-1B/blob/main/tokenizer_config.json) or [Molmo](https://huggingface.co/allenai/Molmo-7B-D-0924) often require non-standardized remote code for image preprocessing. 
    - LLMs like [command-r](https://huggingface.co/CohereForAI/c4ai-command-r-plus-08-2024) have cool special prompts e.g. for grounded generation, but they provide their own custom remote code for preparing prompts/functionalities properly for these special prompts.


## TODO
- [ ] implement PromptTemplate with "template" key instead of "message" key. 
- [ ] get broader feedback
- [ ] ...


## Other notes 

### How to handle tools?
Potential standard ways of storing tools: 
- JSON files: Tool use and function calling is often handled via JSON strings and different libraries then provide different abstractions on top of this. 
 - .py file: libraries like `LangChain` or `Transformers.Agents` enable the use of tools/functions via normal python functions with doc strings and a decorator. This would be less universally compatible/interoperable though. 

`Transformers.Agents` currently has [Tool.push_to_hub](https://huggingface.co/docs/transformers/v4.45.2/en/main_classes/agent#transformers.Tool.push_to_hub) which pushes tools to the hub as a Space. This makes sense if users want a hosted tool with compute, but it is not interoperable with API client libraries. Some tools & prompts are stored [here](https://huggingface.co/huggingface-tools).


### How to handle agents?
TBD. Out of scope?

### Existing prompt template repos:
- distilabel: https://github.com/argilla-io/distilabel/tree/main/src/distilabel/steps/tasks/templates; https://distilabel.argilla.io/latest/components-gallery/tasks/
- langchain hub for prompts: https://smith.langchain.com/hub, old public oss repo: https://github.com/hwchase17/langchain-hub
- langgraph templates for agents: https://blog.langchain.dev/launching-langgraph-templates/
- deepset prompt hub: https://github.com/deepset-ai/prompthub
- promptify (not maintained anymore)  https://github.com/promptslab/Promptify/tree/27a53fa8e8f2a4d90f887d06ece65a44466f873a/promptify/prompts

### Other resources on working with prompts
- langfuse prompt management: https://langfuse.com/docs/prompts/example-langchain





<!-- 
3. Example: Agent Model Repo
    - (maybe:) OAI MLEBench Agents/Dataset: https://github.com/openai/mle-bench (Seems like no nice tabular dataset provided.)
    - Or Aymeric's GAIA prompts
-->
