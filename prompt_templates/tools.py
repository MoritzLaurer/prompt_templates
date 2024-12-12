import inspect
from typing import Any, Callable, Dict, List, Optional, Set

import pkg_resources


class Tool:
    """A standardized tool that can be converted to various agent framework formats."""

    def __init__(
        self,
        func: Callable[..., Any],
        name: str,
        description: str,
        args_description: Dict[str, str],
        return_description: Optional[str] = None,
        raises_description: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Set[str]] = None,
    ):
        self.func = func
        self.name = name
        self.description = description
        self.args_description = args_description
        self.return_description = return_description
        self.raises_description = raises_description
        self.metadata = metadata
        self.dependencies = dependencies if dependencies is not None else set()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Make the tool directly callable with the function's arguments.

        This method allows you to use the tool instance directly as a function,
        passing through any arguments to the underlying function.

        Args:
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Any: The result of calling the underlying function

        Examples:
            Note that the output here always depends on the specific tool function you load.
            >>> from prompt_templates import ToolLoader
            >>> tool = ToolLoader.from_hub(repo_id="MoritzLaurer/example_tools", filename="get_stock_price.py")
            >>> result = tool(ticker="AAPL", days="5d")
            >>> # This specific tool always returns a dictionary with the following keys
            >>> isinstance(result, dict)
            True
            >>> sorted(result.keys())
            ['currency', 'prices', 'timestamps']
            >>> result['currency']
            'USD'
        """
        return self.func(*args, **kwargs)

    def return_uninstalled_dependencies(self) -> List[str]:
        """Check if all required dependencies are installed.

        Returns:
            List[str]: List of uninstalled dependencies that need to be installed for the function to work.

        Examples:
            Check if there are any uninstalled dependencies:
            >>> from prompt_templates import ToolLoader
            >>> tool = ToolLoader.from_hub(repo_id="MoritzLaurer/example_tools", filename="get_stock_price.py")
            >>> uninstalled = tool.return_uninstalled_dependencies()
            >>> if uninstalled:
            ...     print(f"Please install these packages: {uninstalled}")
        """
        uninstalled: List[str] = []
        for dep in self.dependencies:
            try:
                pkg_resources.get_distribution(dep)
            except pkg_resources.DistributionNotFound:
                uninstalled.append(dep)
        return uninstalled

    def to_openai_function(self) -> Dict[str, Any]:
        """Convert to OpenAI function format."""
        # Extract parameter types from function signature
        sig = inspect.signature(self.func)

        parameters: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}

        for name, param in sig.parameters.items():
            param_type = "string"  # default
            if param.annotation != inspect.Parameter.empty:
                if issubclass(param.annotation, str):
                    param_type = "string"
                elif issubclass(param.annotation, int):
                    param_type = "integer"
                elif issubclass(param.annotation, float):
                    param_type = "number"
                elif issubclass(param.annotation, bool):
                    param_type = "boolean"

            parameters["properties"][name] = {"type": param_type, "description": self.args_description.get(name, "")}

            if param.default == inspect.Parameter.empty:
                parameters["required"].append(name)

        return {"name": self.name, "description": self.description, "parameters": parameters}
