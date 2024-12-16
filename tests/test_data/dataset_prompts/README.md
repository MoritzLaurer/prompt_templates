---
license: mit
library_name: prompt-templates
tags:
- prompts
---

## Sharing prompts linked to datasets
This repo illustrates how you can use the `prompt_templates` library to load prompts from YAML files in dataset repositories.

LLMs are increasingly used to help create datasets, for example for quality filtering or synthetic text generation.
The prompts used for creating a dataset are currently unsystematically shared on GitHub ([example](https://github.com/huggingface/cosmopedia/tree/main/prompts)), 
referenced in dataset cards ([example](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu#annotation)), or stored in .txt files ([example](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier/blob/main/utils/prompt.txt)), 
hidden in paper appendices or not shared at all. 
This makes reproducibility unnecessarily difficult.

To facilitate reproduction, these dataset prompts can be shared in YAML files in HF dataset repositories together with metadata on generation parameters, model_ids etc. 


### Example: FineWeb-Edu

The FineWeb-Edu dataset was created by prompting `Meta-Llama-3-70B-Instruct` to score the educational value of web texts. The authors <a href="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu#annotation">provide the prompt</a> in a .txt file.

When provided in a YAML file in the dataset repo, the prompt can easily be loaded and supplemented with metadata like the model_id or generation parameters for easy reproducibility. 

```python
#!pip install hf_hub_prompts
from hf_hub_prompts import download_prompt
import torch
from transformers import pipeline

prompt_template = download_prompt(repo_id="MoritzLaurer/dataset_prompts", filename="fineweb-edu-prompt.yaml", repo_type="dataset")

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


