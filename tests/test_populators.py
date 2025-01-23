import pytest

from prompt_templates.populators import (
    DoubleBracePopulator,
    Jinja2TemplatePopulator,
    SingleBracePopulator,
)


class TestSingleBracePopulator:
    def test_basic_population(self):
        populator = SingleBracePopulator()
        template = "Hello {name}!"
        result = populator.populate(template, {"name": "world"})
        assert result == "Hello world!"

    def test_multiple_variables(self):
        populator = SingleBracePopulator()
        template = "Hello {name}, you are {age} years old"
        result = populator.populate(template, {"name": "Alice", "age": 30})
        assert result == "Hello Alice, you are 30 years old"

    def test_missing_variable(self):
        populator = SingleBracePopulator()
        template = "Hello {name}!"
        with pytest.raises(ValueError, match="Variable 'name' not found"):
            populator.populate(template, {})

    def test_get_variable_names(self):
        populator = SingleBracePopulator()
        template = "Hello {name}, you are {age} years old"
        variables = populator.get_variable_names(template)
        assert variables == {"name", "age"}

    def test_nested_braces(self):
        populator = SingleBracePopulator()
        template = "Nested {outer{inner}}"
        with pytest.raises(ValueError):
            populator.populate(template, {"outer": "value"})


class TestDoubleBracePopulator:
    def test_basic_population(self):
        populator = DoubleBracePopulator()
        template = "Hello {{name}}!"
        result = populator.populate(template, {"name": "world"})
        assert result == "Hello world!"

    def test_multiple_variables(self):
        populator = DoubleBracePopulator()
        template = "Hello {{name}}, you are {{age}} years old"
        result = populator.populate(template, {"name": "Alice", "age": 30})
        assert result == "Hello Alice, you are 30 years old"

    def test_missing_variable(self):
        populator = DoubleBracePopulator()
        template = "Hello {{name}}!"
        with pytest.raises(ValueError, match="Variable 'name' not found"):
            populator.populate(template, {})

    def test_get_variable_names(self):
        populator = DoubleBracePopulator()
        template = "Hello {{name}}, you are {{age}} years old"
        variables = populator.get_variable_names(template)
        assert variables == {"name", "age"}

    def test_nested_braces(self):
        populator = DoubleBracePopulator()
        template = "Nested {{outer{{inner}}}}"
        with pytest.raises(ValueError):
            populator.populate(template, {"outer": "value"})


class TestJinja2TemplatePopulator:
    def test_basic_population(self):
        populator = Jinja2TemplatePopulator()
        template = "Hello {{name}}!"
        result = populator.populate(template, {"name": "world"})
        assert result == "Hello world!"

    def test_strict_mode(self):
        populator = Jinja2TemplatePopulator(security_level="strict")
        # Test allowed filter
        template = "Hello {{name|upper}}!"
        result = populator.populate(template, {"name": "world"})
        assert result == "Hello WORLD!"

        # Test disallowed filter
        template = "Hello {{name|replace('o', 'a')}}!"
        with pytest.raises(ValueError, match="No filter named 'replace'"):
            populator.populate(template, {"name": "world"})

    def test_standard_mode(self):
        populator = Jinja2TemplatePopulator(security_level="standard")
        # Test additional allowed filters
        template = "Hello {{name|replace('o', 'a')|title}}!"
        result = populator.populate(template, {"name": "world"})
        assert result == "Hello Warld!"

        # Test allowed globals
        template = "Count: {{range(3)|join(', ')}}"
        result = populator.populate(template, {})
        assert result == "Count: 0, 1, 2"

    def test_relaxed_mode(self):
        populator = Jinja2TemplatePopulator(security_level="relaxed")
        # Test complex template with multiple features
        template = """
        {% set greeting = 'Hello' %}
        {{ greeting }} {{ name|title }}!
        {% for i in range(count) %}
        Item {{ i + 1 }}
        {% endfor %}
        """
        result = populator.populate(template, {"name": "world", "count": 2})
        assert "Hello World!" in result
        assert "Item 1" in result
        assert "Item 2" in result

    def test_invalid_security_level(self):
        with pytest.raises(ValueError, match="Invalid security level"):
            Jinja2TemplatePopulator(security_level="invalid")

    def test_syntax_error(self):
        populator = Jinja2TemplatePopulator()
        template = "Hello {{name!"  # Missing closing brace
        with pytest.raises(ValueError, match="Invalid template syntax"):
            populator.populate(template, {"name": "world"})

    def test_undefined_variable(self):
        populator = Jinja2TemplatePopulator()
        template = "Hello {{name}}!"
        with pytest.raises(ValueError, match="Undefined variable"):
            populator.populate(template, {})

    def test_get_variable_names(self):
        populator = Jinja2TemplatePopulator()
        template = """
        {% set local = 'value' %}
        Hello {{name}}, you are {{age}} years old.
        {% if show_extra %}
        Extra info: {{extra}}
        {% endif %}
        """
        variables = populator.get_variable_names(template)
        assert variables == {"name", "age", "show_extra", "extra"}
        # Note: 'local' is not included as it's a local variable

    def test_control_structures(self):
        populator = Jinja2TemplatePopulator(security_level="standard")
        template = """
        {% if show_greeting %}
        Hello {{name}}!
        {% else %}
        Goodbye {{name}}!
        {% endif %}
        """
        result = populator.populate(template, {"name": "world", "show_greeting": True})
        assert "Hello world!" in result.strip()

    def test_autoescaping(self):
        populator = Jinja2TemplatePopulator(security_level="standard")
        template = "Hello {{name}}!"
        result = populator.populate(template, {"name": "<script>alert('xss')</script>"})
        assert result == "Hello &lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;!"

        # Test safe filter
        template = "Hello {{name|safe}}!"
        result = populator.populate(template, {"name": "<b>world</b>"})
        assert result == "Hello <b>world</b>!"
