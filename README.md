# HF Hub Prompts 
Prompts have become a key artifact for researchers and practitioners working with AI. 
There is, however, no standardized way of sharing prompts.
Prompts are shared on the HF Hub in [.txt files](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier/blob/main/utils/prompt.txt),
in [HF datasets](https://huggingface.co/datasets/fka/awesome-chatgpt-prompts),
as strings in [model cards](https://huggingface.co/OpenGVLab/InternVL2-8B#grounding-benchmarks),
or on GitHub as [python strings](https://github.com/huggingface/cosmopedia/tree/main/prompts), 
in [JSON, YAML](https://github.com/hwchase17/langchain-hub/blob/master/prompts/README.md),
or in [Jinja2](https://github.com/argilla-io/distilabel/tree/main/src/distilabel/steps/tasks/templates). 



### Objectives and Non-Objectives of this library
#### Objectives
1. Provide a Python library that simplifies and standardises the sharing of prompts on the Hugging Face Hub.
2. Start an open discussion on the best way of standardizing and 
encouraging the sharing of prompts on the HF Hub, building upon the HF Hub's existing repository types and ensuring interoperability with other prompt-related libraries.
#### Non-Objectives: 
- Compete with full-featured prompting libraries like [LangChain](https://github.com/langchain-ai/langchain), 
[ell](https://docs.ell.so/reference/index.html), etc. The objective is, instead, a simple solution for sharing prompts on the HF Hub, which is compatible with other libraries and which the community can build upon. 



## Quick Start

```bash
pip install hf-hub-prompts
```

For examples of the core functionality, see the [docs](https://moritzlaurer.github.io/hf_hub_prompts/).


## Main use-case scenarios on the HF Hub
For use-case examples with all repository types on the Hugging Face Hub, see the [docs](https://moritzlaurer.github.io/hf_hub_prompts/).


## The standardized YAML or JSON prompt template format
For a discussion of the standardized YAML or JSON prompt template format, see the [docs](https://moritzlaurer.github.io/hf_hub_prompts/).


## TODO
- [ ] many things ...




