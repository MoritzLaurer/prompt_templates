# Agents

Sharing tools and agents on the HF Hub in a standardized way is not implemented yet.
This page contains some initial thoughts on this. 

## Main components of agents
Agents have several main components:

- A glue library like [autogen](https://github.com/microsoft/autogen), [CrewAI](https://github.com/crewAIInc/crewAI), [langchain](https://github.com/langchain-ai/langchain), or [transformers.agents](https://huggingface.co/docs/transformers/en/agents), which orchestrate a series of LLM calls and tool use.
- A set of prompt templates that define different tasks and agent personas.
- A set of tools, either as JSON strings or Python functions.
- A compute environment to run the agent code invoking the prompts and tools.


## Sharing agents on the HF Hub

HF Space repos provide a suitable compute environment for agent code and Space repos can also host the YAML/JSON files for prompt templates and tools.

[TODO: add example of a HF Space repo with an agent.]


<!-- 
1. Example: Agent Model Repo
    - (maybe:) OAI MLEBench Agents/Dataset: https://github.com/openai/mle-bench (Seems like no nice tabular dataset provided.)
    - Or Aymeric's GAIA prompts
-->
