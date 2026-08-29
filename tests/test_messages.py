import httpx
import pytest

from n8n_pipe.constants import Language
from n8n_pipe.messages import MESSAGES, describe, t


def test_every_language_has_the_same_keys():
    reference = set(MESSAGES[Language.EN])
    for language, catalogue in MESSAGES.items():
        assert set(catalogue) == reference, f"{language.value} catalogue differs"


@pytest.mark.parametrize("language", list(Language))
def test_every_message_formats_with_its_placeholders(language):
    params = {
        "attempt": 1,
        "total": 3,
        "elapsed": 4,
        "delay": 1.0,
        "error": "x",
        "seconds": 5,
        "status": 500,
        "body": "b",
        "field": "f",
        "name": "n",
        "limit": 1,
        "mime": "m",
    }
    for key in MESSAGES[language]:
        assert t(language, key, **params)


def test_translation_interpolates_in_french():
    assert t(Language.FR, "status.attempt", attempt=2, total=3) == "Tentative 2/3"


def test_describe_never_returns_an_empty_message():
    assert describe(httpx.ReadTimeout("")) == "ReadTimeout"
    assert describe(ValueError("boom")) == "ValueError: boom"
