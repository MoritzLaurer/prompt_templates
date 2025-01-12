# Create a prompt template

You can create and save a prompt template in a few simple lines of code. 

- Create a `ChatPromptTemplate` if your template is composed of multiple messages. Messages must follow the  OpenAI chat message format for standardization (they can always be converted to formats for other LLM clients in a later step via the [utils](reference/utils.md#prompt_templates.utils.format_for_client)). 
- Create a `TextPromptTemplate` if your template is a single string and end users should insert it into the adequate message role themselves.

## TextPromptTemplates

```python
from prompt_templates import TextPromptTemplate

template = """\
Translate the following text to {{language}}:
{{text}}
"""

template_variables = ["language", "text"]  # for input validation to avoid hidden errors
metadata = {
    "name": "Simple Translator",
    "description": "A simple translation prompt for illustrating the standard prompt YAML format",
    "tags": ["translation", "multilinguality"],
    "version": "0.0.1",
    "author": "Guy van Babel",
}
client_parameters = {"temperature": 0}

prompt_template = TextPromptTemplate(
    template=template,
    template_variables=template_variables,
    metadata=metadata,
    client_parameters=client_parameters,
)
```


## ChatPromptTemplates

```python
from prompt_templates import ChatPromptTemplate

template_messages = [
    {"role": "system", "content": "You are a coding assistant who explains concepts clearly and provides short examples."},
    {"role": "user", "content": "Explain what {{concept}} is in {{programming_language}}."},
]
template_variables = ["concept", "programming_language"]  # for input validation to avoid hidden errors
metadata = {
    "name": "Code Teacher",
    "description": "A simple chat prompt for explaining programming concepts with examples",
    "tags": ["programming", "education"],
    "version": "0.0.1",
    "author": "Guido van Bossum",
}
client_parameters = {"temperature": 0}

prompt_template = ChatPromptTemplate(
    template=template_messages,
    template_variables=template_variables,
    metadata=metadata,
    client_parameters=client_parameters,
)
```

## Save the prompt template in a separate file

```python
# save template locally or on the HF Hub
filename = "code_teacher.yaml"
prompt_template.save_to_local(f"./example_prompts/{filename}")
prompt_template.save_to_hub(
    repo_id="MoritzLaurer/example_prompts", 
    filename=filename, 
    create_repo=True,
)
```


## When NOT to create modular prompt template files

When you just started testing prompts for a new task it might be overkill to create prompt templates in separate files. In the earily stage of testing you can keep keep your prompts embedded in your scripts for simplicity. 


## When to create modular prompt template files

After initial vibe-tests with a prompt template, it is good practice to test the template more systematically. When you reach this stage of more systematic testing, you should start storing and versioning your prompt templates in separate .yaml files.

Disentangling your code from your prompt templates enables you to:

- compare which formulation of your template works best on test data
- invite colleagues to try and create their own versions and compare them

Think of your prompt templates as the key **hyperparameters** of your LLM system, where the prompt formulation on the client parameters are the key factors that determine the performance of the system. 

Making prompt templates modular makes them systematically testable, sharable and reusable. 





