"""Jinja2 template renderer for browser-facing HTML responses."""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape


_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


env = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
)


def render(template_name: str, **context) -> str:
    """Render a Jinja2 template and return the HTML string."""
    template = env.get_template(template_name)
    return template.render(**context)
