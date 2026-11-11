import pytest

from src.hub.dataset import download_folder, upload_file, upload_folder


class FakeApi:
    def __init__(self):
        self.calls = []

    def create_repo(self, **kwargs):
        self.calls.append(("create_repo", kwargs))

    def upload_file(self, **kwargs):
        self.calls.append(("upload_file", kwargs))

    def upload_folder(self, **kwargs):
        self.calls.append(("upload_folder", kwargs))


def test_upload_file_creates_a_private_dataset_repo(tmp_path):
    f = tmp_path / "chunks.jsonl"
    f.write_text("{}\n")
    api = FakeApi()
    url = upload_file("me/data", f, api=api)
    assert url.endswith("/datasets/me/data")
    assert api.calls[0][1]["private"] is True and api.calls[0][1]["repo_type"] == "dataset"
    assert api.calls[1][1]["path_in_repo"] == "chunks.jsonl"


def test_upload_folder_goes_to_the_requested_path(tmp_path):
    (tmp_path / "parsed").mkdir()
    api = FakeApi()
    upload_folder("me/data", tmp_path / "parsed", "parsed", api=api)
    assert api.calls[1][0] == "upload_folder" and api.calls[1][1]["path_in_repo"] == "parsed"


def test_missing_local_paths_are_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        upload_file("me/data", tmp_path / "nope.jsonl", api=FakeApi())
    with pytest.raises(FileNotFoundError):
        upload_folder("me/data", tmp_path / "nope", "x", api=FakeApi())


def test_download_requires_a_token(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        download_folder("me/data", "parsed", tmp_path, downloader=lambda **kw: None)


def test_download_asks_only_for_the_requested_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "dummy")
    seen = {}
    path = download_folder("me/data", "parsed", tmp_path, downloader=lambda **kw: seen.update(kw))
    assert path == tmp_path / "parsed" and seen["allow_patterns"] == "parsed/*"
