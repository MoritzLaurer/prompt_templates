from typing import Literal


# File extensions
VALID_PROMPT_EXTENSIONS = (".yaml", ".yml")
VALID_TOOL_EXTENSIONS = (".py",)

# Template types
RendererType = Literal["double_brace", "single_brace", "jinja2"]
Jinja2SecurityLevel = Literal["strict", "standard", "relaxed"]
