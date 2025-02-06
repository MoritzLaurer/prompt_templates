import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Match, Set

import jinja2
from jinja2 import Environment, meta
from jinja2.sandbox import SandboxedEnvironment

from .constants import Jinja2SecurityLevel


class TemplatePopulator(ABC):
    """Abstract base class for template populating strategies."""

    @abstractmethod
    def populate(self, template_str: str, user_provided_variables: Dict[str, Any]) -> str:
        """Populate the template with given user_provided_variables."""
        pass

    @abstractmethod
    def get_variable_names(self, template_str: str) -> Set[str]:
        """Extract variable names from template."""
        pass


class SingleBracePopulator(TemplatePopulator):
    """Template populator using regex for basic {var} substitution."""

    def populate(self, template_str: str, user_provided_variables: Dict[str, Any]) -> str:
        pattern = re.compile(r"\{([^{}]+)\}")

        def replacer(match: Match[str]) -> str:
            key = match.group(1).strip()
            if key not in user_provided_variables:
                raise ValueError(f"Variable '{key}' not found in provided variables")
            return str(user_provided_variables[key])

        return pattern.sub(replacer, template_str)

    def get_variable_names(self, template_str: str) -> Set[str]:
        pattern = re.compile(r"\{([^{}]+)\}")
        return {match.group(1).strip() for match in pattern.finditer(template_str)}


class DoubleBracePopulator(TemplatePopulator):
    """Template populator using regex for {{var}} substitution."""

    def populate(self, template_str: str, user_provided_variables: Dict[str, Any]) -> str:
        pattern = re.compile(r"\{\{([^{}]+)\}\}")

        def replacer(match: Match[str]) -> str:
            key = match.group(1).strip()
            if key not in user_provided_variables:
                raise ValueError(f"Variable '{key}' not found in provided variables")
            return str(user_provided_variables[key])

        return pattern.sub(replacer, template_str)

    def get_variable_names(self, template_str: str) -> Set[str]:
        pattern = re.compile(r"\{\{([^{}]+)\}\}")
        return {match.group(1).strip() for match in pattern.finditer(template_str)}


class Jinja2TemplatePopulator(TemplatePopulator):
    """Jinja2 template populator with configurable security levels.

    Security Levels:
        - strict: Minimal set of features, highest security
            Filters: lower, upper, title, safe
            Tests: defined, undefined, none
            Env: autoescape=True, no caching, no globals, no auto-reload
        - standard (default): Balanced set of features
            Filters: lower, upper, title, capitalize, trim, strip, replace, safe,
                    int, float, join, split, length
            Tests: defined, undefined, none, number, string, sequence
            Env: autoescape=True, limited caching, basic globals, no auto-reload
        - relaxed: Default Jinja2 behavior (use with trusted templates only)
            All default Jinja2 features enabled
            Env: autoescape=False, full caching, all globals, auto-reload allowed

    Args:
        security_level: Level of security restrictions ("strict", "standard", "relaxed")
    """

    def __init__(self, security_level: Jinja2SecurityLevel = "standard"):
        # Store security level for error messages
        self.security_level = security_level

        if security_level == "strict":
            # Most restrictive settings
            self.env = SandboxedEnvironment(
                undefined=jinja2.StrictUndefined,
                trim_blocks=True,
                lstrip_blocks=True,
                autoescape=True,  # Force autoescaping
                cache_size=0,  # Disable caching
                auto_reload=False,  # Disable auto reload
            )
            # Remove all globals
            self.env.globals.clear()

            # Minimal set of features
            safe_filters = {"lower", "upper", "title"}
            safe_tests = {"defined", "undefined", "none"}

        elif security_level == "standard":
            # Balanced settings
            self.env = SandboxedEnvironment(
                undefined=jinja2.StrictUndefined,
                trim_blocks=True,
                lstrip_blocks=True,
                autoescape=False,
                cache_size=100,  # Limited cache
                auto_reload=False,  # Still no auto reload
            )
            # Allow some safe globals
            self.env.globals.update(
                {
                    "range": range,  # Useful for iterations
                    "dict": dict,  # Basic dict operations
                    "len": len,  # Length calculations
                }
            )

            # Balanced set of features
            safe_filters = {
                "lower",
                "upper",
                "title",
                "capitalize",
                "trim",
                "strip",
                "replace",
                "safe",
                "int",
                "float",
                "join",
                "split",
                "length",
            }
            safe_tests = {"defined", "undefined", "none", "number", "string", "sequence"}

        elif security_level == "relaxed":
            self.env = Environment(
                undefined=jinja2.StrictUndefined,
                trim_blocks=True,
                lstrip_blocks=True,
                autoescape=False,  # Default Jinja2 behavior
                cache_size=400,  # Default cache size
                auto_reload=True,  # Allow auto reload
            )
            # Keep all default globals and features
            return
        else:
            raise ValueError(f"Invalid security level: {security_level}")

        # Apply security settings for strict and standard modes
        self._apply_security_settings(safe_filters, safe_tests)

    def _apply_security_settings(self, safe_filters: Set[str], safe_tests: Set[str]) -> None:
        """Apply security settings by removing unsafe filters and tests."""
        # Remove unsafe filters
        unsafe_filters = set(self.env.filters.keys()) - safe_filters
        for unsafe in unsafe_filters:
            self.env.filters.pop(unsafe, None)

        # Remove unsafe tests
        unsafe_tests = set(self.env.tests.keys()) - safe_tests
        for unsafe in unsafe_tests:
            self.env.tests.pop(unsafe, None)

    def populate(self, template_str: str, user_provided_variables: Dict[str, Any]) -> str:
        """Populate the template with given user_provided_variables."""
        try:
            template = self.env.from_string(template_str)
            populated = template.render(**user_provided_variables)
            # Ensure we return a string for mypy
            return str(populated)
        except jinja2.TemplateSyntaxError as e:
            raise ValueError(
                f"Invalid template syntax at line {e.lineno}: {str(e)}\n" f"Security level: {self.security_level}"
            ) from e
        except jinja2.UndefinedError as e:
            raise ValueError(
                f"Undefined variable in template: {str(e)}\n" "Make sure all required variables are provided"
            ) from e
        except Exception as e:
            raise ValueError(f"Error populating template: {str(e)}") from e

    def get_variable_names(self, template_str: str) -> Set[str]:
        """Extract variable names from template."""
        try:
            ast = self.env.parse(template_str)
            variables = meta.find_undeclared_variables(ast)
            # Ensure we return a set of strings for mypy
            return {str(var) for var in variables}
        except jinja2.TemplateSyntaxError as e:
            raise ValueError(f"Invalid template syntax: {str(e)}") from e
