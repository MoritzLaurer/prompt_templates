---
license: mit
library_name: prompt-templates
tags:
- prompts
- model-prompts
---

## Sharing special prompts of open-weight models

This repo illustrates how you can use the `prompt_templates` library to load prompts from YAML files in open-weight model repositories.
Several open-weight models have been tuned on specific tasks with specific prompts. 
For example, the InternVL2 vision language models are one of the very few VLMs that have been trained for zeroshot bounding box prediction for any object.
To elicit this capability, users need to use this special prompt: `Please provide the bounding box coordinate of the region this sentence describes: <ref>{region_to_detect}</ref>'`

These these kinds of task-specific special prompts are currently unsystematically reported in model cards, github repos, .txt files etc. 

The prompt_templates library standardises the sharing of prompts in YAML files.
I recommend sharing these special prompts directly in the model repository of the respective model. 

Below is an example for the InternVL2 model. 

Note that this model card is not actively maintained and the latest documentation for `prompt_templates` is available at https://github.com/MoritzLaurer/prompt_templates 

#### Prompt for extracting bounding boxes of specific objects of interest with InternVL2
```py
#!pip install prompt_templates
from prompt_templates import PromptTemplateLoader

# download image prompt template
prompt_template = PromptTemplateLoader.from_hub(repo_id="MoritzLaurer/open_models_special_prompts", filename="internvl2-bbox-prompt.yaml")

# populate prompt
image_url = "https://unsplash.com/photos/ZVw3HmHRhv0/download?ixid=M3wxMjA3fDB8MXxhbGx8NHx8fHx8fDJ8fDE3MjQ1NjAzNjl8&force=true&w=1920"
region_to_detect = "the bird"
messages = prompt_template.populate_template(image_url=image_url, region_to_detect=region_to_detect)

print(messages)
# out: [{'role': 'user'
#        'content': [{'type': 'image_url',
#                      'image_url': {'url': 'https://unsplash.com/photos/ZVw3HmHRhv0/download?ixid=M3wxMjA3fDB8MXxhbGx8NHx8fHx8fDJ8fDE3MjQ1NjAzNjl8&force=true&w=1920'}},
#                    {'type': 'text',
#                     'text': 'Please provide the bounding box coordinate of the region this sentence describes: <ref>the bird</ref>'}]
#      }]
```

#### Prompt for extracting bounding boxes of any object in an image with InternVL2
```py
# download image prompt template
prompt_template = PromptTemplateLoader.from_hub(repo_id="MoritzLaurer/open_models_special_prompts", filename="internvl2-objectdetection-prompt.yaml")

# populate prompt
image_url = "https://unsplash.com/photos/ZVw3HmHRhv0/download?ixid=M3wxMjA3fDB8MXxhbGx8NHx8fHx8fDJ8fDE3MjQ1NjAzNjl8&force=true&w=1920"
messages = prompt_template.populate_template(image_url=image_url)

print(messages)
# [{'role': 'user',
#  'content': [{'type': 'image_url',
#    'image_url': {'url': 'https://unsplash.com/photos/ZVw3HmHRhv0/download?ixid=M3wxMjA3fDB8MXxhbGx8NHx8fHx8fDJ8fDE3MjQ1NjAzNjl8&force=true&w=1920'}},
#   {'type': 'text',
#    'text': 'Please detect and label all objects in the following image and mark their positions.'}]}]
```

#### Using the prompt with an open inference container like vLLM or TGI

These populated prompts in the OpenAI messages format are then directly compatible with vLLM or TGI containers. 
When you host one of these containers on a HF Endpoint, for example, you can call on the model with the OpenAI client or with the HF Interence Client. 

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


