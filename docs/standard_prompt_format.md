# Standardizing prompt templates

The library expects prompts to be stored as YAML or JSON files in any HF Hub repository. See the `Files` tab in these repos for [open-weight model prompts](https://huggingface.co/MoritzLaurer/open_models_special_prompts), [closed-model prompts](https://huggingface.co/MoritzLaurer/closed_system_prompts), or [dataset prompts](https://huggingface.co/datasets/MoritzLaurer/dataset_prompts).

The YAML files must follow the following structure:

- Top-level key (required): `prompt`. 
- Second-level key (required): *Either* `messages` *or* `template`. If `messages`, the prompt template must be provided as a list of dictionaries following the OpenAI messages format. This format is recommended for use with LLM APIs or inference containers.  If `template`, the prompt should be provided as a single string. 
- Second-level keys (optional): (1) `input_variables`: an optional list of variables for populating the prompt template. This is also used for input validation; (2) `metadata`: Other information, such as the source, date, author etc.; (3) Any other key of relevance, such as `client_settings` with parameters for reproducibility with a specific inference client, or `metrics` form evaluations on specific datasets.

This structure is inspired by the LangChain [PromptTemplate](https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.prompt.PromptTemplate.html) 
and [ChatPromptTemplate](https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html). 



## Pros/Cons of sharing prompts as YAML files, or jinja2 templates, or as HF datasets

### Pro/Con prompts as YAML files
- Existing prompt hubs use YAML: [LangChain Hub](https://smith.langchain.com/hub) (see also [this](https://github.com/hwchase17/langchain-hub/blob/master/prompts/README.md)); 
[Haystack Prompt Hub](https://haystack.deepset.ai/blog/share-and-use-prompt-with-prompthub)
- YAML is the standard for working with prompts in production settings in my experience with practitioners. See also [this discussion](https://github.com/langchain-ai/langchain/discussions/21672).
- Managing individual prompts in separate YAML files makes each prompt a separate file unit. 
    - This makes it e.g. easier to add metadata and production-relevant information in the respective prompt YAML file.
    - Prompts in individual YAML files also enables users to add individual prompts into any HF repo abstraction (Model, Space, Dataset), while datasets always have to be their own abstraction.

### Pro/Con JSON files
- Directly parsable as python dict, similar to YAML
- More verbose to type and less pretty than YAML 

### Pro/Con Jinja2 files
- Has more rich functionality for populating templates
- Can be directly integrated into YAML or JSON
- Issue: allows arbitrary code execution and is less safe
- Harder to read for beginners

### Pro/Con prompts as datasets
- Some prompt datasets like [awesome-chatgpt-prompts](https://huggingface.co/datasets/fka/awesome-chatgpt-prompts) have received many likes on HF
- The dataset viewer allows for easy and quick visualization
- Main cons: the tabular data format is not well suited for reusing prompts and is not standard among practitioners
    - Prompts are independent units that can be used in different applications, while a dataset forces different prompts into one parquet file. 
    - Having multiple prompts in the same dataset forces different prompts to have the same column structure
    - Extracting a single prompt from a dataset with dataset/pandas-like operations is unnecessarily complicated
    - Editing a prompt in JSON or YAML is much easier than editing a (parquet) dataset
    - Data viewers for tabular data are bad for visualizing the structure of long prompts (with line breaks etc.)


### Compatibility with LangChain
LangChain is a great library for creating interoperability between different LLM clients.
It also standardises the use of prompts with its [PromptTemplate](https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.prompt.PromptTemplate.html) 
and [ChatPromptTemplate](https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html) classes. The objective of this library is not to reproduce the full functionality of these LangChain classes. 

A `PromptTemplate` from `hf_hub_prompts` can be easily converted to a langchain template: 

```py
prompt_template = download_prompt_template(
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
    - VLMs require special pre-processors that are not directly compatible with the standardized messages format (?). And new VLMs like [InternVL](https://huggingface.co/OpenGVLab/InternVL2-1B/blob/main/tokenizer_config.json) or [Molmo](https://huggingface.co/allenai/Molmo-7B-D-0924) often require non-standardized remote code for image preprocessing. 
    - LLMs like [command-r](https://huggingface.co/CohereForAI/c4ai-command-r-plus-08-2024) have cool special prompts e.g. for grounded generation, but they provide their own custom remote code for preparing prompts/functionalities properly for these special prompts.



### Existing prompt template repos:
- [LangChain Hub](https://smith.langchain.com/hub) for prompts (main hub is proprietary. See the old public oss [repo](https://github.com/hwchase17/langchain-hub), using JSON or YAML, with {...} for input variables)
- [LangGraph Templates](https://blog.langchain.dev/launching-langgraph-templates/) (underlying data structure unclear, does not seem to have a collaborative way of sharing templates)
- [LlamaHub](https://llamahub.ai/) (seems to use GitHub as backend)
- [Deepset Prompt Hub](https://github.com/deepset-ai/prompthub) (seems not maintained anymore, used YAML with {...} for input variables)
- distilabel [templates](https://github.com/argilla-io/distilabel/tree/main/src/distilabel/steps/tasks/templates) and [tasks](https://distilabel.argilla.io/latest/components-gallery/tasks/) (using pure jinja2 with {{ ... }} for input variables)
- [Langfuse](https://langfuse.com/docs/prompts/get-started), see also [example here](https://langfuse.com/guides/cookbook/prompt_management_langchain) (no public prompt repo, using JSON internally with {{...}} for input variables)
- [Promptify](https://github.com/promptslab/Promptify/tree/27a53fa8e8f2a4d90f887d06ece65a44466f873a/promptify/prompts) (not maintained anymore, used jinja1 and {{ ... }} for input variables)
