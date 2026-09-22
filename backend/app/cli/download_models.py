"""Download the finder and outlining models once, so the first catalogue does not have to wait:

    docker compose run --rm backend python -m app.cli.download_models

About 1.1 GB in total, kept in a Docker volume (so a rebuild does not download it again).
Needs internet access; nothing here runs the models.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from app.core.config import get_settings

# Configuration and tokenizer files, and one copy of the weights. Repositories often hold the
# weights in several formats; fetching them all would multiply the download.
COMMON = ["*.json", "*.txt", "*.model"]
FORMATS = (["*.safetensors"], ["*.bin"])


def _download(repo_id: str) -> Path:
    from huggingface_hub import snapshot_download

    for weights in FORMATS:
        folder = Path(snapshot_download(repo_id=repo_id, allow_patterns=COMMON + weights))
        if any(folder.rglob(weights[0])):
            return folder
    raise RuntimeError(f"{repo_id} has no weights in a format this tool downloads")


def main(argv: Sequence[str] | None = None) -> int:
    settings = get_settings()
    failed = False
    for repo_id in (settings.detection_model, settings.segmentation_model):
        print(f"Downloading {repo_id} ...")
        try:
            folder = _download(repo_id)
        except Exception as exc:
            print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            failed = True
            continue
        size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 1e6
        print(f"  ready ({size:.0f} MB) in {folder}")
    if failed:
        print("Some downloads failed. Check the internet connection and run this again.")
        return 1
    print("All models are downloaded.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
