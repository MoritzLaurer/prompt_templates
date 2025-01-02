from typing import Literal


# File extensions
VALID_PROMPT_EXTENSIONS = (".yaml", ".yml")
VALID_TOOL_EXTENSIONS = (".py",)

# Template types
PopulatorType = Literal["double_brace", "single_brace", "jinja2", "autodetect"]
Jinja2SecurityLevel = Literal["strict", "standard", "relaxed"]
