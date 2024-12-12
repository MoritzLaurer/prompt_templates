# Agents

!!! note
    Standardizing the sharing of tools and agents is in an early experimental stage.

How could the sharing of agents be standardized on the HF Hub?

A good standard for sharing agents should be: modular, open, and interoperable. 

## Modularity: Main components of agents
Agents have four main components:

1. An orchestration library such as [autogen](https://github.com/microsoft/autogen), [CrewAI](https://github.com/crewAIInc/crewAI), [langchain](https://github.com/langchain-ai/langchain), or [transformers.agents](https://huggingface.co/docs/transformers/en/agents), which implements prompt formatting, tool parsing, API calls, agent interaction etc.
2. A set of prompt templates that define different tasks and agent personas.
3. A set of tools, which are essentially a prompt template + code.
4. A compute environment to run the agent code, invoking the prompts and tools.

**Modularity** is a fundamental principle in software engineering. It enables maintainability, reusability, scalability, and testability. In practice, however, the code for LLM systems and agents often combines prompt strings, tool functions and the agent orchestration code in the same files. This means that changes in prompts are hard to test and version and it is harder for others to reuse prompt templates or tools for their own systems. 

Following the principle of modularity, agents should be shared in a directory of modular .yaml/.json files for prompt templates; .py files for tools; and a single agent.py file for the orchestration code. 


## Openness: Sharing and running agents on the HF Hub

HF Space repositories provide a suitable unit for storing the files for prompt templates (.json or .yaml files), tools (.py files) and orchestration code (single agent.py file) in a single directory, combined with attached compute for executing the agent. One Space can contain one agent, which can be executed on a free CPU, or with high-end GPUs if required. HF Spaces can be public, private or shared with a group of people in a specific organization.

[TODO: add example of a HF Space repo with an agent.]

Open question: How can individual prompts and tools be made easily findable and likeable for the community, if they are only files within a repository? 


## Interoperability

Prompts and tools can be made interoperable by breaking them down into the basic file format most libraries use: prompts in .json/.yaml files following the OAI messages format and tools in .py files with functions and doc strings. Only the orchestration code in the agent.py file is non-standardized and can use the code of any orchestration framework, calling on the modular and standardized tools and prompts. 




<!-- 
1. Example: Agent Model Repo
    - (maybe:) OAI MLEBench Agents/Dataset: https://github.com/openai/mle-bench (Seems like no nice tabular dataset provided.)
    - Or Aymeric's GAIA prompts
-->
