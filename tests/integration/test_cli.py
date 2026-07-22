import pytest
import json

from vipaneltr.cli import main


@pytest.mark.parametrize("command", ["prepare-data", "infer", "evaluate", "baseline"])
def test_each_command_has_help(command):
    with pytest.raises(SystemExit) as exc:
        main([command, "--help"])
    assert exc.value.code == 0


def test_prepare_data_validates_repository_dataset(capsys):
    assert main(["prepare-data"]) == 0
    assert "Dataset valid" in capsys.readouterr().out


def test_evaluate_writes_report_inside_output_directory(tmp_path):
    qas_path = tmp_path / "qas_test.json"
    preds_path = tmp_path / "predictions.json"
    qas_path.write_text(
        json.dumps({"qas": [{"qa_id": "q1", "answer": "A", "hints": []}]}),
        encoding="utf-8",
    )
    preds_path.write_text(
        json.dumps([{"qa_id": "q1", "answer": "A"}]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "evaluation"

    status = main(
        [
            "evaluate",
            "--qas",
            str(qas_path),
            "--preds",
            str(preds_path),
            "--dataset-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert status == 0
    assert (output_dir / "evaluation.json").is_file()
