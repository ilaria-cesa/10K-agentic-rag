"""Keep the raw outputs of a run (runs/<system>.jsonl) in the private dataset, so a run can be
continued from another Colab session or another Google account.

Every answered question is one line of a .jsonl file. `restore_runs` brings back the files from
the dataset (keeping a local file when it already has at least as many lines), and `backup_runs`
uploads the local folder. Both are called by the notebook; the network parts can be replaced by
fakes, so the merge rule is tested without a network.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from src.hub.dataset import download_folder, upload_folder


def n_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def merge_runs(source: Path, target: Path) -> list[str]:
    """Copy each .jsonl of `source` into `target` unless `target` already has as many rows.
    Returns the names of the files that were copied."""
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(source.glob("*.jsonl")):
        if n_lines(path) > n_lines(target / path.name):
            shutil.copy(path, target / path.name)
            copied.append(path.name)
    return copied


def restore_runs(repo_id: str, variant: str, local_dir: Path, downloader=None) -> list[str]:
    """Bring runs/<variant>/ of the dataset into `local_dir`. Nothing to restore on a first run."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            download_folder(repo_id, f"runs/{variant}", Path(tmp), downloader=downloader)
        except Exception as error:  # noqa: BLE001  no such folder yet: this is the first run
            print("nothing to restore:", type(error).__name__)
            return []
        return merge_runs(Path(tmp) / "runs" / variant, local_dir)


def backup_runs(repo_id: str, variant: str, local_dir: Path, api=None) -> None:
    if any(local_dir.glob("*.jsonl")):
        upload_folder(repo_id, local_dir, f"runs/{variant}", api=api)
