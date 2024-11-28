# Standardizing prompt templates

The library expects prompt templates to be stored as modular YAML or JSON files. They can be part of any HF repository, see for example the `Files` tab in these repos for [open-weight model prompts](https://huggingface.co/MoritzLaurer/open_models_special_prompts), [closed-model prompts](https://huggingface.co/MoritzLaurer/closed_system_prompts), or [dataset prompts](https://huggingface.co/datasets/MoritzLaurer/dataset_prompts).

A prompt template YAML or JSON file must follow the following standardized structure:

- Top-level key (required): `prompt`. This top-level key signals to the parser that the content of the file is a prompt template.
- Second-level key (required): *Either* `messages` *or* `template`. If `messages`, the prompt template must be provided as a list of dictionaries following the OpenAI messages format. This format is recommended for use with LLM APIs or inference containers. If `template`, the prompt template should be provided as a single string. Input variable placeholders for populating the prompt template are denoted with curly brackets, similar to Python f-strings.
- Second-level keys (optional): (1) `input_variables`: an optional list of variables for populating the prompt template. This is also used for input validation; (2) `metadata`: Other information, such as the source, date, author etc.; (3) Any other key of relevance, such as `client_settings` with parameters for reproducibility with a specific inference client, or `metrics` form evaluations on specific datasets.

This structure is inspired by the LangChain [PromptTemplate](https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.prompt.PromptTemplate.html) 
and [ChatPromptTemplate](https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html).

Example prompt template in YAML: 
```yaml
prompt:
  messages:
    - role: "system"
      content: "You are a coding assistant who explains concepts clearly and provides short examples."
    - role: "user"
      content: "Explain what {concept} is in {programming_language}."
  input_variables:
    - concept
    - programming_language
  metadata:
    name: "Code Teacher"
    description: "A simple chat prompt for explaining programming concepts with examples"
    tags:
      - programming
      - education
    version: "0.0.1"
    author: "Karl Marx"
```

**Naming convention:** We call a file a *"prompt template"*, when it has placeholders ({...}) for dynamically populating the template like an f-string. This makes files more useful and reusable by others for different use-cases. Once the placeholders in the template are populated with specific input variables, we call it a *"prompt"*. 

The following example illustrates how the prompt template becomes a prompt. 

```python
>>> # 1. Download a prompt template:
>>> from hf_hub_prompts import PromptTemplateLoader
>>> prompt_template = PromptTemplateLoader.from_hub(
...     repo_id="MoritzLaurer/example_prompts",
...     filename="code_teacher.yaml"
... )

>>> # 2. Inspect the template and it's input variables:
>>> prompt_template.messages
[{'role': 'system', 'content': 'You are a coding assistant who explains concepts clearly and provides short examples.'}, {'role': 'user', 'content': 'Explain what {concept} is in {programming_language}.'}]
>>> prompt_template.input_variables
['concept', 'programming_language']

>>> # 3. Populate the template with its input variables
>>> prompt = prompt_template.populate_template(
...     concept="list comprehension",
...     programming_language="Python"
... )
>>> prompt.content
[{'role': 'system', 'content': 'You are a coding assistant who explains concepts clearly and provides short examples.'}, {'role': 'user', 'content': 'Explain what list comprehension is in Python.'}]
```



## Pros/Cons for different file formats for sharing prompt templates

### Pro/Con prompts as YAML files
- Existing prompt hubs use YAML (or JSON): [LangChain Hub](https://smith.langchain.com/hub) (see also [this](https://github.com/hwchase17/langchain-hub/blob/master/prompts/README.md)); 
[Haystack Prompt Hub](https://haystack.deepset.ai/blog/share-and-use-prompt-with-prompthub)
- YAML (or JSON) is the standard for working with prompts in production settings in my experience with practitioners. See also [this discussion](https://github.com/langchain-ai/langchain/discussions/21672).
- Managing individual prompt templates in separate YAML files makes each prompt template an independent modular unit. 
    - This makes it e.g. easier to add metadata and production-relevant information in the respective prompt YAML file.
    - Prompt templates in individual YAML files also enables users to add individual prompts into any HF repo abstraction (Model, Space, Dataset), while datasets always have to be their own abstraction.

### Pro/Con JSON files
- The same pro arguments of YAML also apply to JSON. 
- Directly parsable as Python dict, similar to YAML
- More verbose to type and less pretty than YAML, but probably more familiar to some users

### Pro/Con Jinja2 files
- Has more rich functionality for populating prompt templates
- Can be directly integrated into YAML or JSON, so can always be added to the common YAML/JSON standard
- Issue: allows arbitrary code execution and is less safe
- Harder to read for beginners

### Pro/Con prompts as datasets
- Some prompt datasets like [awesome-chatgpt-prompts](https://huggingface.co/datasets/fka/awesome-chatgpt-prompts) have received many likes on HF
- The dataset viewer allows for easy and quick visualization
- Main cons: the tabular data format is not well suited for reusing prompts and is not standard among practitioners
    - Prompt templates are independent modular units that can be used in different applications, which supports the good practice of modular development, into one tabular file. 
    - Having multiple prompts in the same dataset forces different prompts to have the same column structure
    - Datasets on the HF hub are in parquet files, which is not easily editable and interoperable. Editing a prompt in JSON or YAML is much easier than editing a (parquet) dataset and JSON/YAML is much easier to load. 
    - Extracting a single prompt from a dataset with dataset/pandas-like operations is unnecessarily complicated
    - Data viewers for tabular data are bad for visualizing the structure of long prompt templates (where e.g. line breaks have an important substantive meaning)


### Compatibility with LangChain
LangChain is a great library for creating interoperability between different LLM clients.
It also standardises the use of prompts with its [PromptTemplate](https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.prompt.PromptTemplate.html) 
and [ChatPromptTemplate](https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html) classes. The objective of this library is not to reproduce the full functionality of these LangChain classes. 

A `PromptTemplate` from `hf_hub_prompts` can be easily converted to a langchain template: 

```py
from hf_hub_prompts import PromptTemplateLoader
prompt_template = PromptTemplateLoader.from_hub(
    repo_id="MoritzLaurer/closed_system_prompts",
    filename="jokes-prompt.yaml"
)
prompt_template_langchain = prompt_template.to_langchain_template()
```


### Notes on compatibility with `transformers`
- `transformers` provides partial prompt input standardization via chat_templates following the OpenAI messages format:
    - The [simplest use](https://huggingface.co/docs/transformers/en/conversations) is via the text-generation pipeline
    - See also details on [chat_templates](https://huggingface.co/docs/transformers/main/en/chat_templating).
- Limitations: 
    - The original purpose of these chat_templates is to easily add special tokens that a specific open-source model requires under the hood. The `hub_hub_prompts` library is designed for prompt templates for any LLM, not just open-source LLMs.   
    - VLMs require special pre-processors that are not directly compatible with the standardized messages format (?). And new VLMs like [InternVL](https://huggingface.co/OpenGVLab/InternVL2-1B/blob/main/tokenizer_config.json) or [Molmo](https://huggingface.co/allenai/Molmo-7B-D-0924) often require non-standardized remote code for image preprocessing. 
    - LLMs like [command-r](https://huggingface.co/CohereForAI/c4ai-command-r-plus-08-2024) have cool special prompts e.g. for grounded generation, but they provide their own custom remote code for preparing prompts/functionalities properly for these special prompts.



### Existing prompt template repos:
- [LangChain Hub](https://smith.langchain.com/hub) for prompts (main hub is proprietary. See the old public oss [repo](https://github.com/hwchase17/langchain-hub), using JSON or YAML, with {...} for input variables)
- [LangGraph Templates](https://blog.langchain.dev/launching-langgraph-templates/) (underlying data structure unclear, does not seem to have a collaborative way of sharing templates)
- [LlamaHub](https://llamahub.ai/) (seems to use GitHub as backend)
- [Deepset Prompt Hub](https://github.com/deepset-ai/prompthub) (seems not maintained anymore, used YAML with {...} for input variables)
- distilabel [templates](https://github.com/argilla-io/distilabel/tree/main/src/distilabel/steps/tasks/templates) and [tasks](https://distilabel.argilla.io/latest/components-gallery/tasks/) ([source](https://github.com/argilla-io/distilabel/tree/main/src/distilabel/steps/tasks)) (using pure jinja2 with {{ ... }} for input variables)
- [Langfuse](https://langfuse.com/docs/prompts/get-started), see also [example here](https://langfuse.com/guides/cookbook/prompt_management_langchain) (no public prompt repo, using JSON internally with {{...}} for input variables)
- [Promptify](https://github.com/promptslab/Promptify/tree/27a53fa8e8f2a4d90f887d06ece65a44466f873a/promptify/prompts) (not maintained anymore, used jinja1 and {{ ... }} for input variables)
