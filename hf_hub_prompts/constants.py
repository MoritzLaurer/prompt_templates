from typing import Literal


# File extensions
VALID_PROMPT_EXTENSIONS = (".yaml", ".yml")
VALID_TOOL_EXTENSIONS = (".py",)

# Template types
PopulatorType = Literal["double_brace", "single_brace", "jinja2"]
Jinja2SecurityLevel = Literal["strict", "standard", "relaxed"]

# Later: could maybe introduce messages types for stricter input validation
# Would need to check expectations of different clients
# ChatMessage = Dict[Literal["role", "content"], str]
# ChatMessages = List[ChatMessage]
