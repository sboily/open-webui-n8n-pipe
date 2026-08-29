"""Bundle the ``n8n_pipe`` package into the single file Open-WebUI expects.

Usage::

    python scripts/build_single_file.py [output]   # default: dist/n8n_pipe.py

Modules are concatenated in dependency order; relative imports are dropped
because every name ends up in the same namespace. The Open-WebUI frontmatter
is generated from ``n8n_pipe.__version__`` so the version has a single source.
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "n8n_pipe"
DEFAULT_OUTPUT = ROOT / "dist" / "n8n_pipe.py"
MODULE_ORDER = (
    "constants",
    "errors",
    "messages",
    "valves",
    "status",
    "request",
    "attachments",
    "client",
    "pipe",
)
FRONTMATTER = """\
title: N8N Pipe Function
author: Sylvain BOILY (fork from https://openwebui.com/f/coleam/n8n_pipe)
author_url: https://github.com/sboily/open-webui-n8n-pipe
funding_url: https://github.com/sboily/open-webui-n8n-pipe
version: {version}
description: Forward Open-WebUI chat messages (text, images, files) to an n8n webhook workflow.
"""


def read_version() -> str:
    """Return ``__version__`` from the package without importing it."""
    tree = ast.parse((PACKAGE / "__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if "__version__" in names:
            return str(ast.literal_eval(node.value))
    raise RuntimeError("__version__ not found in n8n_pipe/__init__.py")


def _is_docstring(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _start_line(node: ast.stmt) -> int:
    decorators: list[ast.expr] = getattr(node, "decorator_list", [])
    return min([decorator.lineno for decorator in decorators] + [node.lineno])


def split_module(source: str) -> tuple[list[str], str]:
    """Return the external import statements and the body of a module.

    The body keeps the comments preceding each statement; the module docstring
    and relative imports are removed.
    """
    lines = source.splitlines()
    tree = ast.parse(source)
    imports: list[str] = []
    chunks: list[str] = []
    previous_end = 0
    for node in tree.body:
        if _is_docstring(node):
            previous_end = node.end_lineno or node.lineno
            continue
        end = node.end_lineno or node.lineno
        if isinstance(node, ast.ImportFrom) and node.level > 0:
            previous_end = end
            continue
        if isinstance(node, ast.Import | ast.ImportFrom):
            imports.append("\n".join(lines[_start_line(node) - 1 : end]))
        else:
            chunks.append("\n".join(lines[previous_end:end]).strip("\n"))
        previous_end = end
    return imports, "\n\n\n".join(chunks)


def build() -> str:
    """Return the bundled single-file source."""
    imports: dict[str, None] = {}
    bodies: list[str] = []
    for name in MODULE_ORDER:
        module_imports, body = split_module((PACKAGE / f"{name}.py").read_text(encoding="utf-8"))
        for statement in module_imports:
            imports.setdefault(statement, None)
        bodies.append(body)
    header = f'"""\n{FRONTMATTER.format(version=read_version())}"""\n\n'
    source = header + "\n".join(imports) + "\n\n\n" + "\n\n\n".join(bodies) + "\n"
    compile(source, "n8n_pipe.py", "exec")
    return source


def main(argv: list[str]) -> int:
    """Write the bundle to ``argv[0]`` or the default output path."""
    output = Path(argv[0]) if argv else DEFAULT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build(), encoding="utf-8")
    print(f"Bundle written to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
