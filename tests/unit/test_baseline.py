from vipaneltr.baseline.llm_client import calculate_call_cost
from vipaneltr.baseline.run import _calculate_batch_stats


def test_baseline_cost_and_batch_stats_are_offline():
    assert calculate_call_cost("unknown/model", 1_000_000, 1_000_000) == 0.75
    stats = _calculate_batch_stats(
        [
            {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
                "cost_usd": 0.2,
            }
        ]
    )
    assert stats["total_tokens"] == 3
    assert stats["total_cost_usd"] == 0.2
