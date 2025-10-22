from app.config import Settings


def test_zai_api_key_alias(monkeypatch):
    """Z AI API key should load from legacy aliases when present."""

    monkeypatch.setenv("Z_API_KEY", "")
    monkeypatch.setenv("ZAI_API_KEY", "alias-from-legacy")
    monkeypatch.delenv("ZAI_KEY", raising=False)

    settings = Settings()

    assert settings.z_api_key == "alias-from-legacy"


def test_zai_api_key_direct_env(monkeypatch):
    """Environment-provided Z API key remains the primary source."""

    monkeypatch.setenv("Z_API_KEY", "primary-key")
    monkeypatch.setenv("ZAI_API_KEY", "alias-from-legacy")

    settings = Settings()

    assert settings.z_api_key == "primary-key"
