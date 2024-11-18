# Agents and Tools

Sharing tools and agents on the HF Hub in a standardized way is not implemented yet.
This page contains some initial thoughts on this. 


### How to handle tools?
Potential standard ways of storing tools: 

- JSON files: Tool use and function calling is often handled via JSON strings and different libraries then provide different abstractions on top of this. 
- .py file: libraries like `LangChain` or `Transformers.Agents` enable the use of tools/functions via normal python functions with doc strings and a decorator. This would be less universally compatible/interoperable though. 

`Transformers.Agents` currently has [Tool.push_to_hub](https://huggingface.co/docs/transformers/v4.45.2/en/main_classes/agent#transformers.Tool.push_to_hub) which pushes tools to the hub as a Space. Some tools & prompts have been stored like this [here](https://huggingface.co/huggingface-tools) on the Hub. This makes sense if users want a hosted tool with compute, but it is not interoperable with API client libraries.


### How to handle agents?
Agents have several main components:

- A glue library like [autogen](https://github.com/microsoft/autogen), [CrewAI](https://github.com/crewAIInc/crewAI), [langchain](https://github.com/langchain-ai/langchain), [Transformers.Agents](https://huggingface.co/docs/transformers/en/agents), which orchestrate a series of LLM calls and tool use.
- A set of prompt templates that define different tasks and agent personas.
- A set of tools in JSON format.
- A compute environment to run the agent code invoking the prompts and tools.

HF Space repos provide a suitable compute environment for agent code and Space repos can also host the YAML/JSON files for prompt templates and tools.

[TODO: add example of a HF Space repo with an agent.]


<!-- 
1. Example: Agent Model Repo
    - (maybe:) OAI MLEBench Agents/Dataset: https://github.com/openai/mle-bench (Seems like no nice tabular dataset provided.)
    - Or Aymeric's GAIA prompts
-->
