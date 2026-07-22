from vipaneltr.evaluation.io import align_records
from vipaneltr.evaluation.run import _core_metrics


def test_perfect_prediction_scores_one():
    predictions = [{"qa_id": "q1", "answer": "Hà Nội"}]
    references = [{"qa_id": "q1", "answer": "Hà Nội", "hints": ["what"]}]
    aligned, coverage = align_records(predictions, references)
    scores = _core_metrics(aligned, {"em", "f1", "rouge1"})
    assert coverage.evaluated_ids == ["q1"]
    assert scores["em"]["value"] == 1.0
    assert scores["f1"]["f1"] == 1.0
    assert scores["rouge1"]["f1"] == 1.0
