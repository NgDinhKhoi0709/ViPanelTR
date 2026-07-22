import json

import pytest

from vipaneltr.data.validation import DatasetValidationError, validate_dataset


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def make_dataset(tmp_path):
    write_json(
        tmp_path / "table.json",
        {
            "table": [
                {
                    "table_id": "t1",
                    "table_html": "<table><tr><td>A</td></tr></table>",
                }
            ]
        },
    )
    for split in ("train", "dev", "test"):
        write_json(
            tmp_path / f"qas_{split}.json",
            {
                "qas": [
                    {
                        "qa_id": f"{split}-1",
                        "table_id": "t1",
                        "question": "Q?",
                        "answer": "A",
                        "hints": [],
                    }
                ]
            },
        )
    return tmp_path


def test_validate_dataset_reports_counts(tmp_path):
    report = validate_dataset(make_dataset(tmp_path))
    assert report.table_count == 1
    assert report.split_counts == {"train": 1, "dev": 1, "test": 1}


def test_validate_dataset_rejects_unknown_table(tmp_path):
    root = make_dataset(tmp_path)
    write_json(
        root / "qas_test.json",
        {"qas": [{"qa_id": "bad", "table_id": "missing"}]},
    )
    with pytest.raises(DatasetValidationError, match="unknown table_id 'missing'"):
        validate_dataset(root)


def test_validate_dataset_requires_every_split(tmp_path):
    root = make_dataset(tmp_path)
    (root / "qas_dev.json").unlink()
    with pytest.raises(DatasetValidationError, match="Missing required dataset file: qas_dev.json"):
        validate_dataset(root)


def test_validate_dataset_rejects_duplicate_qa_ids_across_splits(tmp_path):
    root = make_dataset(tmp_path)
    write_json(
        root / "qas_dev.json",
        {"qas": [{"qa_id": "train-1", "table_id": "t1"}]},
    )
    with pytest.raises(DatasetValidationError, match="duplicate qa_id 'train-1'"):
        validate_dataset(root)
