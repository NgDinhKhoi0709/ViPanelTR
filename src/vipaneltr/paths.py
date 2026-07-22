from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def dataset_dir() -> Path:
    return repo_root() / "data" / "open_vitabqa"


def outputs_dir() -> Path:
    return repo_root() / "outputs"
