import runpy
from pathlib import Path


def test_supported_scripts_load_from_the_project_package():
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    for name in (
        "split_qas_json.py",
        "answerability_classification_f1.py",
        "metrics_by_table_type.py",
        "create_subset.py",
    ):
        runpy.run_path(str(scripts_dir / name))
