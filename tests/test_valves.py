import pydantic
import pytest

from n8n_pipe.constants import CONNECT_TIMEOUT_SECONDS, Language, SessionIdMode
from n8n_pipe.valves import Valves


def test_defaults_are_safe():
    valves = Valves()
    assert valves.n8n_bearer_token == ""
    assert valves.allow_remote_images is False
    assert valves.session_id_mode is SessionIdMode.CHAT_ID
    assert valves.language is Language.EN
    assert valves.timeout == 120.0


def test_host_requires_scheme_and_drops_trailing_slash():
    assert Valves(n8n_host="https://n8n.example.com/").n8n_host == "https://n8n.example.com"
    with pytest.raises(pydantic.ValidationError, match="http://"):
        Valves(n8n_host="n8n.example.com")


@pytest.mark.parametrize("webhook_id", ["../../admin/api", "with space", "a/b", ""])
def test_webhook_id_rejects_unsafe_characters(webhook_id):
    with pytest.raises(pydantic.ValidationError, match="n8n_webhook_id"):
        Valves(n8n_webhook_id=webhook_id)


def test_webhook_id_accepts_safe_characters():
    assert Valves(n8n_webhook_id="abc-123_XYZ").n8n_webhook_id == "abc-123_XYZ"


def test_assignment_is_validated():
    valves = Valves()
    with pytest.raises(pydantic.ValidationError):
        valves.n8n_host = "ftp://nope"
    with pytest.raises(pydantic.ValidationError):
        valves.timeout = 0


def test_enums_accept_plain_strings_from_open_webui():
    valves = Valves(session_id_mode="legacy", language="fr")
    assert valves.session_id_mode is SessionIdMode.LEGACY
    assert valves.language is Language.FR


def test_http_timeout_uses_valve_for_read_and_short_connect():
    timeout = Valves(timeout=600).http_timeout
    assert timeout.read == 600
    assert timeout.write == 600
    assert timeout.connect == CONNECT_TIMEOUT_SECONDS
