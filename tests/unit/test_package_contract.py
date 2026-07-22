from pathlib import Path

from vipaneltr.paths import dataset_dir, outputs_dir, repo_root


def test_repository_default_paths_are_stable():
    root = repo_root()
    assert root.name == "ViPanel_github"
    assert dataset_dir() == root / "data" / "open_vitabqa"
    assert outputs_dir() == root / "outputs"
    assert isinstance(root, Path)
