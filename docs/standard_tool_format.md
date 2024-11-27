# Standardizing and Sharing Tools


## What are LLM tools?

Imagine you want to build a financial chatbot. For a good chatbot, it is not enough to just generate convincing text, you might also want it to be able to fetch recent financial information or do calculations. While LLM can only generate text, their text output can be used as input to external code, which does some useful action. This external code is called a "function" or a "tool".

Different companies use slightly different language and implementations for this same idea: OpenAI uses the term "[function calling](https://platform.openai.com/docs/guides/function-calling)" when text input for a single function is produced and the term "[tool use](https://platform.openai.com/docs/assistants/tools)" when an LLM assistant has some autonomy to produce text input for one out of several functions; Anthropic primarily uses the term "[tool use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)" (and function calling as a synonym); similar to [Mistral](https://docs.mistral.ai/capabilities/function_calling/); similar to open-source inference engines like [TGI](https://huggingface.co/docs/text-generation-inference/basic_tutorials/using_guidance) or [vLLM](https://docs.vllm.ai/en/v0.6.0/serving/openai_compatible_server.html#tool-calling-in-the-chat-completion-api), which have converged on OpenAI's API specification. (Note that these APIs all follow the JsonAgent paradigm, which is slightly different to the CodeAgent paradigm)


## Main components of tools

LLM tools have the following main components: 

1. A textual description of the tool, including its inputs and outputs. This description is passed to the LLM's prompt, to enable it to produce text outputs that fit to the tool's description. For closed-source LLM, this integration of the tool description into the prompt is hidden. 
2. Code that implements the tool. For example a simple Python function taking a search query text as input, does an API call, and returns ranked search results as output. 
3. A compute environment in which the tool's code is executed. This can e.g. be your local computers' development environment, or docker container running on a cloud CPU. 


## Current formats for sharing tools

The tutorials of **LLM API providers** format tools either as Python dictionaries or JSON strings ([OpenAI](https://platform.openai.com/docs/guides/function-calling), [Anthropic](https://docs.anthropic.com/en/docs/build-with-claude/tool-use), [Mistral](https://docs.mistral.ai/capabilities/function_calling/), [TGI](https://huggingface.co/docs/text-generation-inference/basic_tutorials/using_guidance#tools-and-functions-), [vLLM](https://docs.vllm.ai/en/stable/getting_started/examples/offline_chat_with_tools.html)), which are integrated into example scripts.

**LLM agent libraries** all have their own implementations of tools for their library: [LangChain Tools](https://github.com/langchain-ai/langchain/tree/master/libs/langchain/langchain/tools), LangChain Community [Tools](https://github.com/langchain-ai/langchain/tree/master/libs/community/langchain_community/tools) or [Agent Toolkits](https://github.com/langchain-ai/langchain/tree/a83357dc5ab5fcbed8c2dd7606e9ce763e48d194/libs/community/langchain_community/agent_toolkits) ([docs](https://python.langchain.com/docs/how_to/#tools)); [LlamaHub](https://llamahub.ai/?tab=tools) ([docs](https://docs.llamaindex.ai/en/stable/understanding/agent/tools/), [docs](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/tools/)); [CrewAI Tools](https://github.com/crewAIInc/crewAI-tools) ([docs](https://docs.crewai.com/concepts/tools), including wrapper for using LangChain and LlamaHub tools); [AutoGen](https://github.com/microsoft/autogen/tree/main/python/packages/autogen-core/src/autogen_core/components/tools) ([docs](https://microsoft.github.io/autogen/dev//user-guide/core-user-guide/framework/tools.html), including a LangChain [tool wrapper](https://github.com/microsoft/autogen/tree/main/python/packages/autogen-ext/src/autogen_ext/tools)); [Transformers Agents](https://github.com/huggingface/transformers/tree/main/src/transformers/agents) etc.

As all of these libraries and their tool collections are hosten on GitHub, GitHub has indirectly become the main platform for sharing LLM tools today, although it has not been designed for this purpose. 

The main standardizing force for LLM tools are the API specifications and the expected JSON input format of LLM API providers. As OpenAI is the main standard setter, most libraries are compatible with the JSON input format specified in the OpenAI function/tool calling [guide](https://platform.openai.com/docs/guides/function-calling) and [docs](https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools). In the field of agents, this has lead to the json agent paradigm. (Note that this requirement of LLM API compatibility is unnecessary in the code agent paradigm, where the LLM writes executable code itself, instead of only writing the structured input for existing code.)


## Reflections on the best formats for standardizing tools

The most elegant and universal way of creating a tool is probably a .py file with a function and a doc string (used e.g. by [CrewAI](https://docs.crewai.com/concepts/tools#creating-your-own-tools), [AutoGen](https://microsoft.github.io/autogen/0.2/docs/tutorial/tool-use/#tool-schema), [LangChain](https://python.langchain.com/docs/how_to/custom_tools/) and [Transformers Agents](https://huggingface.co/docs/transformers/en/agents#create-a-new-tool)). This combines the executable function code with the textual description of the tool via the doc string in an standardized way. 

For JsonAgents, the function's docstring can be parsed to construct the expected input for the LLM API and the API then resturns the required inputs for the .py file.
For CodeAgents, the function can directly be passed to the LLM's prompt and the .py file is directly executable.  

Alternatively, tools could be shared as .json files, but this would decouple the tool's description (in the .json file) from its code (e.g. in a .py file)


## Current implementation in transformers.agents

`transformers.agents` currently has [Tool.push_to_hub](https://huggingface.co/docs/transformers/v4.45.2/en/main_classes/agent#transformers.Tool.push_to_hub) which pushes tools to the hub as a Space. Some tools & prompts have been stored like this [here](https://huggingface.co/huggingface-tools) on the Hub. This makes sense if users want a hosted tool with compute. The modularity and interoperability of this approach, however, can probably be improved. Tools as single functions in .py files would be independent units that can be reuse more easily by others and would be more interoperable with other libraries. 






<!--

## illustration of a mapping/translation from a Python function with docstring to JSON input in OpenAI format
(for JSON agents)

### Schema
```json
{
  "type": "function",
  "function": {
    "name": "get_stock_price",
    "parameters": {
        "type": "object", 
        "properties": {
            "ticker": {
                "type": "string", 
                "description": "The stock ticker symbol for the company whose current stock price you want to retrieve. For example, 'AAPL' for Apple Inc."
            }
        },
        "required": ["ticker"],
        "additionalProperties": false
    }
  }
}
```

### Implementation
```python
def get_stock_price(ticker: str) -> float:
    """
    Retrieve the current stock price for a given ticker symbol.
    
    Args:
        ticker (str): The stock ticker symbol for the company whose current stock price you want to retrieve. For example, 'AAPL' for Apple Inc.
        
    Returns:
        float: The current stock price
    """
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        current_price = stock.info['regularMarketPrice']
        return current_price
    except Exception as e:
        raise ValueError(f"Could not retrieve stock price for {ticker}: {str(e)}")
```

### Schema-Implementation Mapping

| Schema Element | Implementation Element |
|----------------|------------------------|
| `"function": "get_stock_price"` | `def get_stock_price()` |
| `"type": "string"` | `ticker: str` |
| `"description"` | Function docstring |


-->

