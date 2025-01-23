# PromptTemplateDictionaries

!!! note
    This feature is highly experimental and will change in the coming days.

Complex LLM systems often depend on multiple interdependent prompt templates instead of a single template. A good example for this are agents, where the general system logic, planning steps and different tasks are defined in separate templates. It can be easier to define, read and change the interdependent templates in a single file, as opposed to separate files.

The `PromptTemplateDictionary` is designed for these use-cases. A `PromptTemplateDictionary` is simply a dictionary of `ChatPromptTemplate`s or `TextPromptTemplate`s which are loaded as a single Python object and stored in a single YAML file. You can load and use them like this:

```py
>>> from prompt_templates import PromptTemplateDictionary

>>> template_dictionary = PromptTemplateDictionary.load_from_local(
...     file_path="./tests/test_data/example_prompts/agent_example_1.yaml"
... )

>>> print(template_dictionary.template_dictionary)  # TODO: rename attribute
# {'agent_system_prompt': ChatPromptTemplate(template=[{'role': 'system', 'content': 'You are a code age...', template_variables=['tool_descriptions', 'task'], metadata={}, client_parameters={}, custom_data={}, populator='jinja2', jinja2_security_level='standard'),
#  'agent_planning_prompt': TextPromptTemplate(template='Here is your task:\n\nTask:\n```\n{{task}}\n```\n...', template_variables=['task', 'tool_descriptions', 'managed_agents_desc...', metadata={}, client_parameters={}, custom_data={}, populator='jinja2', jinja2_security_level='standard')}
```

When integrating the `PromptTemplateDictionary` into your agent code, you can access and populate the respective template as follows. Once populated, the template becomes a list of message dicts (for `ChatPromptTemplate`) or a single string (for `TextPromptTemplate`) which can be directly passed to an LLM client.

```py
agent_system_prompt = template_dictionary["agent_system_prompt"].populate(
    tool_descriptions="... some tool descriptions ...",
    task="... some task ...",
)
print(agent_system_prompt)
# [{'role': 'system',
#   'content': 'You are a code agent and you have the following tools at your disposal:\n<tools>\n... some tool descriptions ...\n</tools>'},
#  {'role': 'user',
#   'content': 'Here is the task:\n<task>\n... some task ...\n</task>\nNow begin!'}]
```


A `PromptTemplateDictionary` is defined like this in a yaml file:

```yaml
prompt:
  template_dictionary:
    agent_system_prompt:
      template:
        - role: "system"
          content: |-
            You are a code agent and you have the following tools at your disposal:
            <tools>
            {{tool_descriptions}}
            </tools>
        - role: "user"
          content: |-
            Here is the task:
            <task>
            {{task}}
            </task>
            Now begin!
      template_variables:
        - tool_descriptions
        - task
    agent_planning_prompt:
      template: |-
        Here is your task:

        Task:
        </task>
        {{task}}
        <task>

        Your plan can leverage any of these tools:
        {{tool_descriptions}}

        {{managed_agents_descriptions}}

        List of facts that you know:
        <facts>
        {{answer_facts}}
        </facts>

        Now begin! Write your plan below.
      template_variables:
        - task
        - tool_descriptions
        - managed_agents_descriptions
        - answer_facts
  metadata:
    name: "Example Code Agent"
    description: "A simple code agent example"
    tags:
      - agent
    version: "0.0.1"
    author: "Guido van Bossum"
  client_parameters: {}
  custom_data: {}
```


You can either create and edit these templates directly in YAML.

Alternatively, you can create a `PromptTemplateDictionary` programmatically like this: 

```py
from prompt_templates import PromptTemplateDictionary, ChatPromptTemplate, TextPromptTemplate

agent_system_prompt_template = ChatPromptTemplate(
    template=[
        {'role': 'system', 'content': 'You are a code agent and you have the following tools at your disposal:\n<tools>\n{{tool_descriptions}}\n</tools>'},
        {'role': 'user', 'content': 'Here is the task:\n<task>\n{{task}}\n</task>\nNow begin!'},
    ],
    template_variables=['tool_descriptions', 'task'],
)

agent_planning_prompt_template = TextPromptTemplate(
    template='Here is your task:\n\nTask:\n```\n{{task}}\n```\n\nYour plan can leverage any of these tools:\n{{tool_descriptions}}\n\n{{managed_agents_descriptions}}\n\nList of facts that you know:\n```\n{{answer_facts}}\n```\n\nNow begin! Write your plan below.',
    template_variables=['task', 'tool_descriptions', 'managed_agents_descriptions', 'answer_facts'],
)

template_dictionary = PromptTemplateDictionary(
    template_dictionary={
        "agent_system_prompt": agent_system_prompt_template,
        "agent_planning_prompt": agent_planning_prompt_template,
    }
)

# not implemented yet
template_dictionary.save_to_local(file_path="./tests/test_data/example_prompts/agent_example_test.yaml")
template_dictionary.save_to_hub(repo_id="moritzlaurer/agent_example_test", filename="agent_example_test.yaml", create_repo=True)
```


