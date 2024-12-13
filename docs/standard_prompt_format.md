# Standardizing prompt templates

The library expects prompt templates to be stored as modular YAML or JSON files. They can be stored locally or in an HF repository, see for example the `Files` tab in these repos for [open-weight model prompts](https://huggingface.co/MoritzLaurer/open_models_special_prompts), [closed-model prompts](https://huggingface.co/MoritzLaurer/closed_system_prompts), or [dataset prompts](https://huggingface.co/datasets/MoritzLaurer/dataset_prompts).

A prompt template YAML or JSON file must follow the following standardized structure:

- Top-level key (required): `prompt`. This top-level key signals to the parser that the content of the file is a prompt template.
- Second-level key (required): `template`. This can be either a simple _string_, or a _list of dictionaries_ following the OpenAI messages format. The messages format is recommended for use with LLM APIs or inference containers. Variable placeholders for populating the prompt template string are denoted with double curly brackets _{{...}}_.
- Second-level keys (optional): (1) `template_variables` (_list_): variables for populating the prompt template. This is used for input validation and to make the required variables for long templates easily accessible; (2) `metadata` (_dict_): information about the template such as the source, date, author etc.; (3) `client_parameters` (_dict_): parameters for the inference client (e.g. temperature, model).

Example prompt template following the standard in YAML: 
```yaml
prompt:
  template:
    - role: "system"
      content: "You are a coding assistant who explains concepts clearly and provides short examples."
    - role: "user"
      content: "Explain what {{concept}} is in {{programming_language}}."
  template_variables:
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

**Naming convention:** We call a file a *"prompt template"*, when it has placeholders ({{...}}) for dynamically populating the template similar to an f-string. This makes files more useful and reusable by others for different use-cases. Once the placeholders in the template are populated with specific variables, we call it a *"prompt"*. 

The following example illustrates how the prompt template becomes a prompt. 

```python
>>> # 1. Download a prompt template:
>>> from prompt_templates import PromptTemplateLoader
>>> prompt_template = PromptTemplateLoader.from_hub(
...     repo_id="MoritzLaurer/example_prompts",
...     filename="code_teacher.yaml"
... )

>>> # 2. Inspect the template and it's variables:
>>> prompt_template.template
[{'role': 'system', 'content': 'You are a coding assistant who explains concepts clearly and provides short examples.'}, {'role': 'user', 'content': 'Explain what {concept} is in {programming_language}.'}]
>>> prompt_template.template_variables
['concept', 'programming_language']

>>> # 3. Populate the template with its variables
>>> prompt = prompt_template.populate_template(
...     concept="list comprehension",
...     programming_language="Python"
... )
>>> print(prompt)
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
- Main cons: the tabular data format is not well suited for reusing prompt templates 
and is not standard among practitioners
  - Extracting a single prompt from a tabular dataset with dataset/pandas-like operations is unnecessarily complicated.
  - In industry practice, prompt templates are independent modular units that can be reused for different use-cases. Having multiple templates in the same dataset forces different templates to have the same column structure and prevents proper modular development.  
  - Datasets on the HF hub are in parquet files, which are not easily editable. Editing a prompt in JSON or YAML is much easier than editing a (parquet) dataset. 
  - Data viewers for tabular data are bad for visualizing the structure of long prompt templates (where e.g. line breaks have an important substantive meaning). Viewing and editing prompt templates in markdown-like editors is more standard in the ecosystem.
  - Saving prompt templates as datasets prevents them from being modular components of model or space repos (see [example use-cases](repo_types_examples.md) for this) 



### Compatibility with LangChain
LangChain is a great library for creating interoperability between different LLM clients.
This library is inspired by LangChain's [PromptTemplate](https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.prompt.PromptTemplate.html) 
and [ChatPromptTemplate](https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html) classes. One difference is that the LangChain ChatPromptTemplate expects a "messages" key instead of a "template" key for the prompt template in the messages format. This HF library uses the "template" key both for HF [TextPromptTemplate][prompt_templates.prompt_templates.TextPromptTemplate] and for HF [ChatPromptTemplate][prompt_templates.prompt_templates.ChatPromptTemplate] for simplicity. If you still load a YAML/JSON file with a "messages" key, it will be automatically renamed to "template". You can also always convert a HF PromptTemplate to a LangChain template with [.to_langchain_template()][prompt_templates.prompt_templates.ChatPromptTemplate.to_langchain_template]. The objective of this library is not to reproduce the full functionality of a library like LangChain, but to enable the community to share prompts on the HF Hub and load and reuse them with any of their favourite libraries. 


A `PromptTemplate` from `prompt_templates` can be easily converted to a langchain template: 

```py
from prompt_templates import PromptTemplateLoader
prompt_template = PromptTemplateLoader.from_hub(
    repo_id="MoritzLaurer/example_prompts",
    filename="code_teacher.yaml"
)
prompt_template_langchain = prompt_template.to_langchain_template()
```


### Notes on compatibility with `transformers`
- `transformers` provides partial prompt input standardization via chat_templates following the OpenAI messages format:
    - The [simplest use](https://huggingface.co/docs/transformers/en/conversations) is via the text-generation pipeline
    - See also details on [chat_templates](https://huggingface.co/docs/transformers/main/en/chat_templating).
- Limitations: 
    - The original purpose of these chat_templates is to easily add special tokens that a specific open-source model requires under the hood. The `prompt_templates` library is designed for prompt templates for any LLM, not just open-source LLMs.   
    - VLMs require special pre-processors that are not directly compatible with the standardized messages format (?). And new VLMs like [InternVL](https://huggingface.co/OpenGVLab/InternVL2-1B/blob/main/tokenizer_config.json) or [Molmo](https://huggingface.co/allenai/Molmo-7B-D-0924) often require non-standardized remote code for image preprocessing. 
    - LLMs like [command-r](https://huggingface.co/CohereForAI/c4ai-command-r-plus-08-2024) have cool special prompts e.g. for grounded generation, but they provide their own custom remote code for preparing prompts/functionalities properly for these special prompts.



### Existing prompt template repos:
- [LangChain Hub](https://smith.langchain.com/hub) for prompts (main hub is proprietary. See the old public oss [repo](https://github.com/hwchase17/langchain-hub), using JSON or YAML, with {...} for template variables)
- [LangGraph Templates](https://blog.langchain.dev/launching-langgraph-templates/) (underlying data structure unclear, does not seem to have a collaborative way of sharing templates)
- [LlamaHub](https://llamahub.ai/) (seems to use GitHub as backend)
- [Deepset Prompt Hub](https://github.com/deepset-ai/prompthub) (seems not maintained anymore, used YAML with {...} for template variables)
- distilabel [templates](https://github.com/argilla-io/distilabel/tree/main/src/distilabel/steps/tasks/templates) and [tasks](https://distilabel.argilla.io/latest/components-gallery/tasks/) ([source](https://github.com/argilla-io/distilabel/tree/main/src/distilabel/steps/tasks)) (using pure jinja2 with {{ ... }} for template variables)
- [Langfuse](https://langfuse.com/docs/prompts/get-started), see also [example here](https://langfuse.com/guides/cookbook/prompt_management_langchain) (no public prompt repo, using JSON internally with {{...}} for template variables)
- [Promptify](https://github.com/promptslab/Promptify/tree/27a53fa8e8f2a4d90f887d06ece65a44466f873a/promptify/prompts) (not maintained anymore, used jinja1 and {{ ... }} for template variables)
