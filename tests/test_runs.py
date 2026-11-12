from src.hub.runs import backup_runs, merge_runs, n_lines, restore_runs


def write(path, n):
    path.write_text("".join(f'{{"id": "Q{i}"}}\n' for i in range(n)), encoding="utf-8")


def test_longer_file_wins_and_shorter_is_kept(tmp_path):
    remote, local = tmp_path / "remote", tmp_path / "local"
    remote.mkdir()
    local.mkdir()
    write(remote / "a.jsonl", 5)
    write(local / "a.jsonl", 2)  # the dataset has more: copied
    write(remote / "b.jsonl", 3)
    write(local / "b.jsonl", 7)  # this session has more: kept
    write(remote / "c.jsonl", 1)  # not here yet: copied
    assert merge_runs(remote, local) == ["a.jsonl", "c.jsonl"]
    assert n_lines(local / "a.jsonl") == 5 and n_lines(local / "b.jsonl") == 7
    assert n_lines(local / "c.jsonl") == 1


def test_first_run_has_nothing_to_restore(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "x")

    def missing(**kwargs):
        raise RuntimeError("repository folder not found")

    assert restore_runs("u/r", "text", tmp_path / "runs", downloader=missing) == []


def test_restore_copies_downloaded_files(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "x")

    def fake_download(**kwargs):
        folder = __import__("pathlib").Path(kwargs["local_dir"]) / "runs" / "text"
        folder.mkdir(parents=True)
        write(folder / "rag5.jsonl", 4)

    assert restore_runs("u/r", "text", tmp_path / "runs", downloader=fake_download) == [
        "rag5.jsonl"
    ]
    assert n_lines(tmp_path / "runs" / "rag5.jsonl") == 4


def test_backup_uploads_only_when_there_are_results(tmp_path):
    calls = []

    class Api:
        def create_repo(self, **kwargs):
            pass

        def upload_folder(self, **kwargs):
            calls.append(kwargs["path_in_repo"])

    (tmp_path / "runs").mkdir()
    backup_runs("u/r", "text", tmp_path / "runs", api=Api())
    assert calls == []
    write(tmp_path / "runs" / "a.jsonl", 1)
    backup_runs("u/r", "text", tmp_path / "runs", api=Api())
    assert calls == ["runs/text"]
