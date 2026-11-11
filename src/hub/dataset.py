"""Move data between notebooks through a PRIVATE Hugging Face dataset repo.

Colab sessions are temporary, so each notebook uploads what the next one needs:
    raw/ and parsed/     (notebook 01: downloaded filings and extracted sections)
    chunks and vectors   (notebooks 02a / 02b)

The token is read from the HF_TOKEN environment variable (a Colab Secret), never from code.
`api` and `downloader` can be replaced by fakes, so these helpers are tested without a network.
"""

from __future__ import annotations

import os
from pathlib import Path


def _token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set the HF_TOKEN environment variable (a token with write access).")
    return token


def _api(api=None):
    if api is not None:
        return api
    from huggingface_hub import HfApi

    return HfApi(token=_token())


def _url(repo_id: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}"


def upload_file(repo_id: str, path: Path, path_in_repo: str | None = None, api=None) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    api = _api(api)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path_in_repo or path.name,
        repo_id=repo_id,
        repo_type="dataset",
    )
    return _url(repo_id)


def upload_folder(repo_id: str, folder: Path, path_in_repo: str, api=None) -> str:
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    api = _api(api)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(folder),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
    )
    return _url(repo_id)


def download_folder(repo_id: str, path_in_repo: str, local_dir: Path, downloader=None) -> Path:
    """Download one folder of the dataset to `local_dir/path_in_repo` and return that path."""
    if downloader is None:
        from huggingface_hub import snapshot_download

        downloader = snapshot_download
    downloader(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=f"{path_in_repo}/*",
        local_dir=str(local_dir),
        token=_token(),
    )
    return local_dir / path_in_repo
