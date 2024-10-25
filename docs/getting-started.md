# Getting Started

## Installation

Install the package using pip:

```bash
pip install hf-hub-prompts
```

Or using Poetry:

```bash
poetry add hf-hub-prompts
```

## Quick Start

### Downloading a Prompt

Download existing prompts from the Hugging Face Hub:

```python
from hf_hub_prompts import download_prompt

# Download a prompt template
prompt = download_prompt(
    repo_id="username/repository",
    filename="prompt.yaml"
)
```

### Text Prompts

Text prompts are defined in YAML files:

```yaml
prompt:
  template: "Write a {tone} story about {subject}"
  input_variables: ["tone", "subject"]
```

Use them in your code:

```python
# Download and use the prompt
prompt = download_prompt("username/repo", "story-prompt.yaml")

# Populate the template
populated = prompt.populate_template(
    tone="humorous",
    subject="a talking cat"
)

# Get the final text
text = populated.content
print(text)  # "Write a humorous story about a talking cat"
```

### Chat Prompts

Chat prompts support multiple messages and can be formatted for different LLM clients:

```yaml
prompt:
  messages:
    - role: "system"
      content: "You are a {role} specialized in {domain}."
    - role: "user"
      content: "Help me understand {topic}"
  input_variables: ["role", "domain", "topic"]
```

Use them with different LLM clients:

```python
# Download and use the chat prompt
chat_prompt = download_prompt("username/repo", "chat-prompt.yaml")

# Populate the template
populated = chat_prompt.populate_template(
    role="tutor",
    domain="physics",
    topic="quantum mechanics"
)

# Format for different clients
openai_messages = populated.format_for_client("openai")
anthropic_messages = populated.format_for_client("anthropic")

# Or use the convenience method
messages = chat_prompt.create_messages(
    client="openai",
    role="tutor",
    domain="physics",
    topic="quantum mechanics"
)
```

### Listing Available Prompts

List all YAML files in a repository:

```python
from hf_hub_prompts import list_prompts

prompts = list_prompts("username/repository")
print(prompts)  # ['prompt1.yaml', 'prompt2.yaml', ...]
```
