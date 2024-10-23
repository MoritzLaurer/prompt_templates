from typing import Any, Dict, List, Optional, Literal
import re
import json
import yaml
import logging

logger = logging.getLogger(__name__)

SUPPORTED_CLIENT_FORMATS = ["openai", "anthropic"]  # TODO: add more clients


class PromptTemplate:
    """A class representing a prompt template that can be formatted and used with various LLM clients.

    Attributes:
        messages (List[Dict[str, Any]]): A list of message dictionaries.
        input_variables (Optional[List[str]]): List of input variable names expected by the prompt.
        metadata (Optional[Dict[str, Any]]): Additional metadata associated with the prompt.
        full_yaml_content (Optional[Dict[str, Any]]): The full content the YAML prompt file.
        prompt_url (Optional[str]): URL of the prompt on Hugging Face Hub.
    """
    
    def __init__(self, full_yaml_content: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Initialize a PromptTemplate instance.

        Args:
            full_yaml_content (Optional[Dict[str, Any]]): The full content of YAML prompt file.
            **kwargs: Arbitrary keyword arguments corresponding to the keys in the YAML file.

        Raises:
            ValueError: If 'messages' key is not provided in kwargs.
        """
        
        if "messages" not in kwargs:
            raise ValueError("You must always provide 'messages' to PromptTemplate.")
        
        # set all YAML file keys as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        self.full_yaml_content = full_yaml_content
                    

    def format_messages(self, client: str = "openai", **kwargs: Any) -> List[Dict[str, str]]:
        """Format the prompt messages by replacing placeholders with provided values.

        Args:
            client (str): The client format to use ('openai', 'anthropic'). Defaults to 'openai'.
            **kwargs: The values to replace placeholders in the messages.

        Returns:
            List[Dict[str, str]]: A list of formatted message dictionaries.

        Raises:
            ValueError: If input variables provided do not match the expected input variables.
            ValueError: If an unsupported client format is specified.
        """
            
        def _replace_placeholders(obj: Any, kwargs: Dict[str, Any]) -> Any:
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
                return {key: _replace_placeholders(value, kwargs) for key, value in obj.items()}
            
            elif isinstance(obj, list):
                # Recursively handle lists
                return [_replace_placeholders(item, kwargs) for item in obj]
            
            return obj  # For non-string, non-dict, non-list types, return as is
        
        
        # validation if user user provided input variables are the same as the ones in the YAML file
        if hasattr(self, 'input_variables'):
            if set(kwargs.keys()) != set(self.input_variables):
                raise ValueError(
                    "Your input variables do not match the prompt template's input variables from the YAML file.\n"
                    f"You provided: {list(kwargs.keys())}.\n"
                    f"The YAML file's input_variables are: {self.input_variables}.\n"
                    f"See the YAML file at: {self.prompt_url}"
                )
        else:
            logger.warning(
                "It is recommended to provide 'input_variables' in the YAML file." 
                "This enables input validation when populating prompt template."
            )
        
        # Populate messages
        messages_populated = [
            {**msg, "content": _replace_placeholders(msg["content"], kwargs)}
            for msg in self.messages
        ]
        
        # convert messages to the desired client format
        if client == "openai":
            # is default
            pass
        elif client == "anthropic":
            messages_populated = self._template_to_anthropic(messages_populated)
        elif client not in SUPPORTED_CLIENT_FORMATS:
            raise ValueError(
                f"Unsupported client format: {client}. Supported formats are: {SUPPORTED_CLIENT_FORMATS}"
            )
        
        return messages_populated

    
    def to_langchain_template(self) -> "ChatPromptTemplate":
        """Convert the PromptTemplate to a LangChain ChatPromptTemplate.

        Returns:
            ChatPromptTemplate: A LangChain ChatPromptTemplate object.

        Raises:
            ImportError: If LangChain is not installed.
        """
        
        try:
            from langchain_core.prompts import ChatPromptTemplate
        except ImportError:
            raise ImportError("LangChain is not installed. Please install it with 'pip install langchain-core' to use this feature.")
                
        chat_prompt_template = ChatPromptTemplate(
            messages=[(msg['role'], msg['content']) for msg in self.messages],
            input_variables=self.input_variables if hasattr(self, 'input_variables') else None,
            metadata=self.metadata if hasattr(self, 'metadata') else None,
        )

        return chat_prompt_template
    
    
    def _template_to_anthropic(self, messages_populated: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
        """Convert messages to the format expected by the Anthropic client."""
        
        messages_anthropic = {
            "system": next((msg["content"] for msg in messages_populated if msg["role"] == "system"), None),
            "messages": [msg for msg in messages_populated if msg["role"] != "system"]
        }
        return messages_anthropic
    
    
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
        attributes = ', '.join(f"{key}={value!r}" for key, value in self.__dict__.items())
        return f"PromptTemplate({attributes})"
    