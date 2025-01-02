from typing import Literal


# File extensions
VALID_PROMPT_EXTENSIONS = (".yaml", ".yml")
VALID_TOOL_EXTENSIONS = (".py",)

# Template types
PopulatorType = Literal["jinja2", "double_brace_regex", "single_brace_regex"]
Jinja2SecurityLevel = Literal["strict", "standard", "relaxed"]
