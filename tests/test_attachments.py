import base64
import ipaddress
import sys
import types

import httpx
import pytest

from n8n_pipe import attachments
from n8n_pipe.attachments import AttachmentCollector, extension_for
from n8n_pipe.errors import AttachmentError
from n8n_pipe.valves import Valves
from tests.conftest import PNG_BYTES, PNG_DATA_URL, Recorder

PUBLIC_IP = ipaddress.ip_address("93.184.216.34")
PRIVATE_IP = ipaddress.ip_address("10.0.0.5")


def collector(handler=None, **valves) -> AttachmentCollector:
    transport = httpx.MockTransport(handler or (lambda request: httpx.Response(404)))
    return AttachmentCollector(Valves(**valves), httpx.AsyncClient(transport=transport))


@pytest.fixture
def public_dns(monkeypatch):
    async def resolve(host):
        return [PUBLIC_IP]

    monkeypatch.setattr(attachments, "resolve_addresses", resolve)


def install_fake_open_webui(monkeypatch, stored, *, async_lookup=False):
    class Files:
        @staticmethod
        def get_file_by_id(file_id):
            if async_lookup:

                async def lookup():
                    return stored

                return lookup()
            return stored

    class Storage:
        @staticmethod
        def get_file(path):
            return path

    modules = {
        "open_webui": types.ModuleType("open_webui"),
        "open_webui.models": types.ModuleType("open_webui.models"),
        "open_webui.models.files": types.ModuleType("open_webui.models.files"),
        "open_webui.storage": types.ModuleType("open_webui.storage"),
        "open_webui.storage.provider": types.ModuleType("open_webui.storage.provider"),
    }
    modules["open_webui.models.files"].Files = Files
    modules["open_webui.storage.provider"].Storage = Storage
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_extension_for_known_and_unknown_mime_types():
    assert extension_for("image/jpeg") == ".jpg"
    assert extension_for("image/png") == ".png"
    assert extension_for("bogus/type") == ".bin"


async def test_data_url_image_becomes_a_multipart_part():
    parts = await collector().collect([PNG_DATA_URL], [])
    assert parts == [("image_0", ("image_0.png", PNG_BYTES, "image/png"))]


async def test_data_url_with_extra_parameters_and_jpeg_extension():
    url = "data:image/jpeg;name=photo.jpeg;base64," + base64.b64encode(b"jpg").decode()
    [(name, (filename, content, mime))] = await collector().collect([url], [])
    assert (name, filename, content, mime) == ("image_0", "image_0.jpg", b"jpg", "image/jpeg")


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("data:image/png,not-base64", "invalid data URL"),
        ("data:image/png;base64,@@@not-base64@@@", "invalid data URL"),
        ("ftp://example.com/a.png", "unsupported URL scheme"),
        ("data:application/pdf;base64," + base64.b64encode(b"%PDF").decode(), "not an image"),
    ],
)
async def test_invalid_image_urls_are_rejected(url, expected):
    with pytest.raises(AttachmentError, match=expected):
        await collector().collect([url], [])


async def test_oversized_data_url_is_rejected_before_decoding():
    url = "data:image/png;base64," + base64.b64encode(b"x" * (1024 * 1024 + 10)).decode()
    with pytest.raises(AttachmentError, match="exceeds 1 MB"):
        await collector(max_attachment_size_mb=1).collect([url], [])


async def test_remote_images_are_disabled_by_default():
    with pytest.raises(AttachmentError, match="allow_remote_images"):
        await collector().collect(["https://example.com/a.png"], [])


async def test_remote_image_is_downloaded_when_allowed(public_dns):
    recorder = Recorder(
        lambda request: httpx.Response(
            200, content=PNG_BYTES, headers={"content-type": "image/png; charset=binary"}
        )
    )
    parts = await collector(recorder, allow_remote_images=True).collect(
        ["https://example.com/a.png"], []
    )
    assert parts == [("image_0", ("image_0.png", PNG_BYTES, "image/png"))]
    assert recorder.requests[0].url == "https://example.com/a.png"


async def test_remote_image_from_private_network_is_refused(monkeypatch):
    async def resolve(host):
        return [PUBLIC_IP, PRIVATE_IP]

    monkeypatch.setattr(attachments, "resolve_addresses", resolve)
    with pytest.raises(AttachmentError, match="private network"):
        await collector(allow_remote_images=True).collect(["http://internal.test/a.png"], [])


async def test_remote_image_dns_failure_is_reported(monkeypatch):
    async def resolve(host):
        raise OSError("Name or service not known")

    monkeypatch.setattr(attachments, "resolve_addresses", resolve)
    with pytest.raises(AttachmentError, match="Could not download image_0: OSError"):
        await collector(allow_remote_images=True).collect(["http://nowhere.test/a.png"], [])


async def test_remote_image_http_error_is_reported(public_dns):
    with pytest.raises(AttachmentError, match="Could not download image_0: HTTPStatusError"):
        await collector(allow_remote_images=True).collect(["https://example.com/a.png"], [])


async def test_remote_image_declared_too_large_is_refused(public_dns):
    def handler(request):
        return httpx.Response(200, content=b"x", headers={"content-length": str(2 * 1024 * 1024)})

    with pytest.raises(AttachmentError, match="exceeds 1 MB"):
        await collector(handler, allow_remote_images=True, max_attachment_size_mb=1).collect(
            ["https://example.com/a.png"], []
        )


async def test_remote_image_streamed_too_large_is_refused(public_dns):
    def handler(request):
        return httpx.Response(
            200, content=b"x" * (1024 * 1024 + 1), headers={"content-type": "image/png"}
        )

    with pytest.raises(AttachmentError, match="exceeds 1 MB"):
        await collector(handler, allow_remote_images=True, max_attachment_size_mb=1).collect(
            ["https://example.com/a.png"], []
        )


def file_item(**record):
    return {"type": "file", "id": record.get("id", "f1"), "name": "report.pdf", "file": record}


async def test_uploaded_file_bytes_come_from_open_webui_storage(monkeypatch, tmp_path):
    stored_path = tmp_path / "f1.pdf"
    stored_path.write_bytes(b"%PDF-1.7")
    install_fake_open_webui(monkeypatch, types.SimpleNamespace(path=str(stored_path)))
    item = file_item(id="f1", filename="../report.pdf", meta={"content_type": "application/pdf"})
    parts = await collector().collect([], [item])
    assert parts == [("file_0", ("report.pdf", b"%PDF-1.7", "application/pdf"))]


async def test_uploaded_file_lookup_can_be_async(monkeypatch, tmp_path):
    stored_path = tmp_path / "f1.bin"
    stored_path.write_bytes(b"raw")
    install_fake_open_webui(
        monkeypatch, types.SimpleNamespace(path=str(stored_path)), async_lookup=True
    )
    parts = await collector().collect([], [file_item(id="f1")])
    assert parts == [("file_0", ("f1", b"raw", "application/octet-stream"))]


async def test_uploaded_file_without_path_uses_extracted_text(monkeypatch):
    install_fake_open_webui(monkeypatch, types.SimpleNamespace(path=None))
    item = file_item(id="f1", filename="notes.docx", data={"content": "Extracted text"})
    parts = await collector().collect([], [item])
    assert parts == [("file_0", ("notes.txt", b"Extracted text", "text/plain"))]


async def test_uploaded_file_outside_open_webui_uses_extracted_text():
    item = file_item(id="f1", filename="notes.md", data={"content": "# Title"})
    parts = await collector().collect([], [item])
    assert parts == [("file_0", ("notes.txt", b"# Title", "text/plain"))]


async def test_uploaded_file_without_any_content_is_an_error():
    with pytest.raises(AttachmentError, match=r"report\.pdf could not be read"):
        await collector().collect([], [file_item(id="f1")])


async def test_uploaded_file_too_large_is_refused():
    item = file_item(id="f1", filename="big.txt", data={"content": "x" * (1024 * 1024 + 1)})
    with pytest.raises(AttachmentError, match="exceeds 1 MB"):
        await collector(max_attachment_size_mb=1).collect([], [item])


async def test_non_file_items_are_ignored():
    items = [{"type": "collection", "id": "kb"}, {"type": "web_search"}]
    assert await collector().collect([], items) == []


async def test_file_item_without_id_is_unreadable():
    with pytest.raises(AttachmentError, match="could not be read"):
        await collector().collect([], [{"type": "file", "file": {}}])
