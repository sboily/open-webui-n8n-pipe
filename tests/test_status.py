from n8n_pipe.constants import Language
from n8n_pipe.status import StatusEmitter
from n8n_pipe.valves import Valves


async def test_emits_translated_status_events(events):
    status = StatusEmitter(events, Valves(language=Language.FR))
    await status.info("status.attempt", attempt=1, total=2)
    await status.warning("status.retry", delay=1.0, attempt=1, total=2)
    await status.done("status.complete")
    await status.error("boom")
    assert [event["type"] for event in events.events] == ["status"] * 4
    assert events.descriptions == [
        "Tentative 1/2",
        "Nouvel essai dans 1.0s (1/2)",
        "Terminé",
        "boom",
    ]
    assert [event["data"]["level"] for event in events.events] == [
        "info",
        "warning",
        "info",
        "error",
    ]
    assert [event["data"]["done"] for event in events.events] == [False, False, True, True]


async def test_nothing_is_emitted_without_emitter_or_when_disabled(events):
    await StatusEmitter(None, Valves()).info("status.calling")
    await StatusEmitter(events, Valves(enable_status_indicator=False)).info("status.calling")
    assert events.events == []
