from typing import Any, Dict, List, Optional, Literal, Union
import re
import json
import yaml
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

SUPPORTED_CLIENT_FORMATS = ["openai", "anthropic"]  # TODO: add more clients


class BasePromptTemplate(ABC):
    """An abstract base class for prompt templates."""

    def __init__(self, full_yaml_content: Optional[Dict[str, Any]] = None, prompt_url: Optional[str] = None, **kwargs: Any) -> None:
        """Initialize a BasePromptTemplate instance.

        Args:
            full_yaml_content (Optional[Dict[str, Any]]): The full content of the YAML prompt file.
            **kwargs: Arbitrary keyword arguments corresponding to the keys in the YAML file.
        """
        # Set all YAML file keys as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.full_yaml_content = full_yaml_content
        self.prompt_url = prompt_url  

    @abstractmethod
    def populate_template(self, **kwargs: Any) -> Union[str, List[Dict[str, Any]]]:
        """Abstract method to populate the prompt template with the given variables.
        
        Args:
            **kwargs: The values to replace placeholders in the template.
        
        Returns:
            Union[str, List[Dict[str, Any]]]: The populated template, either as:
                - str: For standard prompt templates
                - List[Dict[str, Any]]: For chat prompt templates in OpenAI format
        """
        pass

    def display(self, format: Literal['json', 'yaml'] = 'json') -> None:
        """Display the full prompt YAML file content in the specified format.

        Args:
            format (Literal['json', 'yaml']): The format to display ('json' or 'yaml'). Defaults to 'json'.

        Raises:
            ValueError: If an unsupported format is specified.
        """

        if format == 'json':
            print(json.dumps(self.full_yaml_content, indent=2))
        elif format == 'yaml':
            print(yaml.dump(self.full_yaml_content, default_flow_style=False, sort_keys=False))
        else:
            raise ValueError(f"Unsupported format: {format}")

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]

    # TODO: remove self.full_yaml_content from this or elsewhere? Very long and duplicative
    def __repr__(self) -> str:
        attributes = ', '.join(f"{key}={repr(value)[:50]}..." if len(repr(value)) > 50 else f"{key}={repr(value)}"
                            for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}({attributes})"

    def _replace_placeholders(self, obj: Any, kwargs: Dict[str, Any]) -> Any:
        """Recursively replace placeholders in strings or nested structures like dicts or lists."""
        pattern = re.compile(r'\{([^{}]+)\}')

        if isinstance(obj, str):
            # Replace placeholders in strings
            def replacer(match):
                key = match.group(1).strip()
                return str(kwargs.get(key, match.group(0)))
            return pattern.sub(replacer, obj)

        elif isinstance(obj, dict):
            # Recursively handle dictionaries
            return {key: self._replace_placeholders(value, kwargs) for key, value in obj.items()}

        elif isinstance(obj, list):
            # Recursively handle lists
            return [self._replace_placeholders(item, kwargs) for item in obj]

        return obj  # For non-string, non-dict, non-list types, return as is

    def _validate_input_variables(self, kwargs: Dict[str, Any]) -> None:
        """Validate that the provided input variables match the expected ones."""
        if hasattr(self, 'input_variables'):
            missing_vars = set(self.input_variables) - set(kwargs.keys())
            extra_vars = set(kwargs.keys()) - set(self.input_variables)
            
            if missing_vars or extra_vars:
                error_msg = []
                error_msg.append(f"Expected input_variables from the prompt template: {self.input_variables}")
                if missing_vars:
                    error_msg.append(f"Missing variables: {list(missing_vars)}")
                if extra_vars:
                    error_msg.append(f"Unexpected variables: {list(extra_vars)}")
                error_msg.append(f"Provided variables: {list(kwargs.keys())}")
                if hasattr(self, 'prompt_url'):
                    error_msg.append(f"Template URL: {self.prompt_url}")
                
                raise ValueError("\n".join(error_msg))
        else:
            logger.warning(
                "No input_variables specified in template. "
                "Input validation is disabled."
            )


class PromptTemplate(BasePromptTemplate):
    """A class representing a standard prompt template."""

    def __init__(self, full_yaml_content: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Initialize a PromptTemplate instance.

        Args:
            full_yaml_content (Optional[Dict[str, Any]]): The full content of YAML prompt file.
            **kwargs: Arbitrary keyword arguments corresponding to the keys in the YAML file.

        Raises:
            ValueError: If 'template' key is not provided in kwargs.
        """
        if "template" not in kwargs:
            raise ValueError("You must always provide 'template' to PromptTemplate.")

        super().__init__(full_yaml_content=full_yaml_content, **kwargs)

    def populate_template(self, **kwargs: Any) -> str:
        """Populate the prompt by replacing placeholders with provided values.

        Args:
            **kwargs: The values to replace placeholders in the template.

        Returns:
            str: The populated prompt string.
        """
        self._validate_input_variables(kwargs)
        populated_prompt = self._replace_placeholders(self.template, kwargs)
        return populated_prompt

    def to_langchain_template(self) -> "PromptTemplate":
        """Convert the PromptTemplate to a LangChain PromptTemplate.

        Returns:
            PromptTemplate: A LangChain PromptTemplate object.

        Raises:
            ImportError: If LangChain is not installed.
        """
        try:
            from langchain.prompts import PromptTemplate as LC_PromptTemplate
        except ImportError:
            raise ImportError("LangChain is not installed. Please install it with 'pip install langchain' to use this feature.")

        lc_prompt_template = LC_PromptTemplate(
            template=self.template,
            input_variables=self.input_variables if hasattr(self, 'input_variables') else None,
            metadata=self.metadata if hasattr(self, 'metadata') else None,
        )
        return lc_prompt_template


class ChatPromptTemplate(BasePromptTemplate):
    """A class representing a chat prompt template that can be formatted and used with various LLM clients."""

    def __init__(self, full_yaml_content: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Initialize a ChatPromptTemplate instance.

        Args:
            full_yaml_content (Optional[Dict[str, Any]]): The full content of YAML prompt file.
            **kwargs: Arbitrary keyword arguments corresponding to the keys in the YAML file.

        Raises:
            ValueError: If 'messages' key is not provided in kwargs.
        """
        if "messages" not in kwargs:
            raise ValueError("You must always provide 'messages' to ChatPromptTemplate.")

        super().__init__(full_yaml_content=full_yaml_content, **kwargs)


    def populate_template(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """Populate the prompt messages by replacing placeholders with provided values.

        Args:
            **kwargs: The values to replace placeholders in the messages.

        Returns:
            List[Dict[str, Any]]: List of populated message dictionaries in OpenAI format.
        """
        self._validate_input_variables(kwargs)

        messages_populated = [
            {**msg, "content": self._replace_placeholders(msg["content"], kwargs)}
            for msg in self.messages
        ]
        return messages_populated


    def format_for_client(self, messages: List[Dict[str, Any]], client: str = "openai") -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Format populated messages for a specific client.

        Args:
            messages (List[Dict[str, Any]]): List of populated message dictionaries.
            client (str): The client format to use ('openai', 'anthropic'). Defaults to 'openai'.

        Returns:
            Union[List[Dict[str, Any]], Dict[str, Any]]: Messages formatted for the specified client.

        Raises:
            ValueError: If an unsupported client format is specified.
        """
        if client == "openai":
            return messages
        elif client == "anthropic":
            return self._template_to_anthropic(messages)
        else:
            raise ValueError(
                f"Unsupported client format: {client}. "
                f"Supported formats are: {SUPPORTED_CLIENT_FORMATS}"
            )
        

    def create_messages(self, client: str = "openai", **kwargs: Any) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Convenience method to populate template and format for client in one step.

        Args:
            client (str): The client format to use ('openai', 'anthropic'). Defaults to 'openai'.
            **kwargs: The values to replace placeholders in the messages.

        Returns:
            Union[List[Dict[str, Any]], Dict[str, Any]]: Populated and formatted messages.
        """
        populated_messages = self.populate_template(**kwargs)
        return self.format_for_client(populated_messages, client)
        
        
    def _template_to_anthropic(self, messages_populated: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
        """Convert messages to the format expected by the Anthropic client."""

        messages_anthropic = {
            "system": next((msg["content"] for msg in messages_populated if msg["role"] == "system"), None),
            "messages": [msg for msg in messages_populated if msg["role"] != "system"]
        }
        return messages_anthropic


    def to_langchain_template(self) -> "ChatPromptTemplate":
        """Convert the ChatPromptTemplate to a LangChain ChatPromptTemplate.

        Returns:
            ChatPromptTemplate: A LangChain ChatPromptTemplate object.

        Raises:
            ImportError: If LangChain is not installed.
        """

        try:
            from langchain.prompts import ChatPromptTemplate as LC_ChatPromptTemplate
        except ImportError:
            raise ImportError("LangChain is not installed. Please install it with 'pip install langchain' to use this feature.")

        lc_chat_prompt_template = LC_ChatPromptTemplate(
            messages=[(msg['role'], msg['content']) for msg in self.messages],
            input_variables=self.input_variables if hasattr(self, 'input_variables') else None,
            metadata=self.metadata if hasattr(self, 'metadata') else None,
        )

        return lc_chat_prompt_template
