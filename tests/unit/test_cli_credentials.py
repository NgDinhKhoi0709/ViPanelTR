from vipaneltr.cli import require_provider_credentials


def test_missing_provider_credentials_are_reported_without_values(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    status = require_provider_credentials(
        ["openai/gpt-4o-mini", "openrouter/qwen/qwen3-8b"]
    )

    assert status == 2
    captured = capsys.readouterr()
    assert "Missing required environment variable: OPENAI_API_KEY" in captured.err
    assert "Missing required environment variable: OPENROUTER_API_KEY" in captured.err
