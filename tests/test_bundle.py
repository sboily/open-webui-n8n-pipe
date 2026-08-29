import importlib.util
import sys
from pathlib import Path

import n8n_pipe
from scripts import build_single_file
from tests.conftest import EventLog, echo_recorder, install_transport

ROOT = Path(__file__).resolve().parent.parent


def load_bundle(path: Path):
    spec = importlib.util.spec_from_file_location("n8n_pipe_bundle", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_version_has_a_single_source():
    assert build_single_file.read_version() == n8n_pipe.__version__


def test_bundle_starts_with_open_webui_frontmatter(tmp_path):
    output = tmp_path / "n8n_pipe.py"
    assert build_single_file.main([str(output)]) == 0
    source = output.read_text(encoding="utf-8")
    assert source.startswith('"""\ntitle: N8N Pipe Function\n')
    assert f"version: {n8n_pipe.__version__}\n" in source
    assert "from ." not in source, "relative imports must be dropped"


async def test_bundle_is_a_working_single_file_pipe(tmp_path, body):
    output = tmp_path / "n8n_pipe.py"
    build_single_file.main([str(output)])
    module = load_bundle(output)
    pipe = module.Pipe()
    pipe.valves = module.Pipe.Valves(n8n_host="http://n8n.test", n8n_webhook_id="hook")
    recorder = echo_recorder("bundled answer")
    install_transport(pipe, recorder)
    events = EventLog()

    assert await pipe.pipe(body, {"id": "u1"}, events, "chat-1") == "bundled answer"
    assert recorder.last_json["sessionId"] == "chat-1"
    assert events.last["done"] is True


def test_bundle_keeps_comments_and_decorators():
    source = build_single_file.build()
    assert "@field_validator" in source
    assert '# n8n "When Last Node Finishes / All Entries"' in source


def test_module_order_covers_every_package_module():
    modules = {path.stem for path in (ROOT / "n8n_pipe").glob("*.py")} - {"__init__"}
    assert set(build_single_file.MODULE_ORDER) == modules
