from concurrent.futures import ThreadPoolExecutor

from vipaneltr.system.agents.llm_client import (
    UsageTrackingMixin,
    _normalize_usage,
    calculate_call_cost,
)
from vipaneltr.utils.trace import StructuredOutputSaver


class DummyClient(UsageTrackingMixin):
    model = "gpt-4o-mini"

    def __init__(self):
        self._init_usage_tracking()


def test_cost_policy_uses_documented_fallback():
    assert calculate_call_cost("gpt-4o-mini", 1_000_000, 1_000_000) == 0.75
    assert calculate_call_cost("gpt-4o", 1_000_000, 1_000_000) == 12.5
    assert calculate_call_cost("unknown/model", 1_000_000, 1_000_000) == 0.75


def test_usage_normalization_prefers_provider_values_then_estimates():
    exact = _normalize_usage(
        {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
        "prompt",
        "answer",
    )
    assert exact["prompt_tokens"] == 12
    assert exact["completion_tokens"] == 8
    assert exact["total_tokens"] == 20

    estimated = _normalize_usage(None, "abcdefgh", "ijkl")
    assert estimated["prompt_tokens"] == 2
    assert estimated["completion_tokens"] == 1
    assert estimated["total_tokens"] == 3


def test_usage_tracker_keeps_parallel_qas_isolated():
    client = DummyClient()

    def record(qa_id):
        for _ in range(10):
            client._record_usage(
                qa_id,
                "prompt",
                "reply",
                {"prompt_tokens": 2, "completion_tokens": 3},
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(record, ["qa-a", "qa-b"]))

    for qa_id in ("qa-a", "qa-b"):
        usage = client.get_usage(qa_id)
        assert usage["prompt_tokens"] == 20
        assert usage["completion_tokens"] == 30
        assert usage["total_tokens"] == 50
        assert usage["cost_usd"] > 0


def test_structured_output_saves_per_qa_and_batch_costs(tmp_path):
    saver = StructuredOutputSaver(str(tmp_path), run_id="cost-test")
    saver.add_result(
        {
            "qa_id": "a",
            "answerable": True,
            "confidence": 1,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cost_usd": 0.1,
        }
    )
    saver.add_result(
        {
            "qa_id": "b",
            "answerable": False,
            "confidence": 0,
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "total_tokens": 30,
            "cost_usd": 0.2,
        }
    )

    stats = saver._compute_stats()
    assert stats["total_prompt_tokens"] == 30
    assert stats["total_completion_tokens"] == 15
    assert stats["total_tokens"] == 45
    assert stats["total_cost_usd"] == 0.3
    assert stats["average_cost_usd"] == 0.15
