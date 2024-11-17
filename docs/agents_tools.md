
## Working with tools and agents
Sharing tools and agents on the HF Hub in a standardized way is not implemented yet.
This page contains some initial thoughts on this. 


### How to handle tools?
Potential standard ways of storing tools: 
- JSON files: Tool use and function calling is often handled via JSON strings and different libraries then provide different abstractions on top of this. 
- .py file: libraries like `LangChain` or `Transformers.Agents` enable the use of tools/functions via normal python functions with doc strings and a decorator. This would be less universally compatible/interoperable though. 

`Transformers.Agents` currently has [Tool.push_to_hub](https://huggingface.co/docs/transformers/v4.45.2/en/main_classes/agent#transformers.Tool.push_to_hub) which pushes tools to the hub as a Space. This makes sense if users want a hosted tool with compute, but it is not interoperable with API client libraries. Some tools & prompts are stored [here](https://huggingface.co/huggingface-tools).


### How to handle agents?
TBD


<!-- 
3. Example: Agent Model Repo
    - (maybe:) OAI MLEBench Agents/Dataset: https://github.com/openai/mle-bench (Seems like no nice tabular dataset provided.)
    - Or Aymeric's GAIA prompts
-->
