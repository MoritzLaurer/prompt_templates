# Testing Documentation

## Running Tests
```bash
# Run all tests
poetry run pytest

# Run with verbose output
poetry run pytest -v

# Run tests with coverage report
poetry run pytest --cov=prompt_templates

# Run doctests
run: poetry run pytest --doctest-modules --cov=prompt_templates --cov-report=xml

# Run specific test file
poetry run pytest tests/test_prompt_templates.py

# Run specific test class or function
poetry run pytest tests/test_prompt_templates.py::TestChatPromptTemplate
poetry run pytest tests/test_prompt_templates.py::TestChatPromptTemplate::test_initialization
```





