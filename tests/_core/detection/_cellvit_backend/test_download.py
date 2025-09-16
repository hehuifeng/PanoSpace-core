import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DOWNLOAD_MODULE_PATH = ROOT / "panospace" / "_core" / "detection" / "_cellvit_backend" / "download.py"


class _DummyResponse:
    headers = {"content-length": "0"}

    def raise_for_status(self):  # pragma: no cover - not exercised in tests
        pass

    def iter_content(self, block_size):  # pragma: no cover - not exercised in tests
        return []


sys.modules.setdefault("requests", SimpleNamespace(get=lambda *args, **kwargs: _DummyResponse()))


class _DummyTqdm:
    def __init__(self, *args, **kwargs):  # pragma: no cover - not exercised in tests
        self.n = 0

    def update(self, *args, **kwargs):  # pragma: no cover - not exercised in tests
        pass

    def close(self):  # pragma: no cover - not exercised in tests
        pass


sys.modules.setdefault("tqdm", SimpleNamespace(tqdm=_DummyTqdm))

spec = importlib.util.spec_from_file_location("download", DOWNLOAD_MODULE_PATH)
assert spec and spec.loader  # for mypy type checking assurance
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


class DummyLogger(logging.Logger):
    """A simple logger that records info messages for inspection."""

    def __init__(self):
        super().__init__(name="dummy")
        self.messages = []

    def info(self, msg, *args, **kwargs):  # type: ignore[override]
        self.messages.append(msg)


def test_print_logger_prints_to_stdout(capsys):
    logger = download.PrintLogger()

    logger.info("info message")
    logger.warning("warning message")

    captured = capsys.readouterr()
    assert "info message" in captured.out
    assert "warning message" in captured.out


def test_check_and_download_logs_when_file_exists(tmp_path: Path):
    logger = DummyLogger()
    file_name = "existing.txt"
    (tmp_path / file_name).write_text("content")

    download.check_and_download(tmp_path, file_name, "http://example.com", logger=logger)

    assert logger.messages == [
        f"The file {file_name} already exists in {tmp_path}."
    ]


def test_check_and_download_passes_logger(monkeypatch, tmp_path: Path):
    logger = DummyLogger()
    file_name = "new.txt"
    download_link = "http://example.com/new.txt"
    captured = {}

    def fake_download_file(link, file_path, logger=None):
        captured["link"] = link
        captured["file_path"] = file_path
        captured["logger"] = logger

    monkeypatch.setattr(download, "download_file", fake_download_file)

    download.check_and_download(tmp_path, file_name, download_link, logger=logger)

    expected_path = tmp_path / file_name
    assert captured["link"] == download_link
    assert captured["file_path"] == expected_path
    assert captured["logger"] is logger
    assert logger.messages == [f"Downloading file to {expected_path}"]
