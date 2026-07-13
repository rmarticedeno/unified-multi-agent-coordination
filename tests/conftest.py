import pytest


@pytest.fixture(autouse=True)
def _explicit_test_a2a_transport_exception(monkeypatch):
    """All HTTP agents in the test corpus are development-only fixtures."""
    monkeypatch.setenv("COORDINATION_ALLOW_INSECURE_A2A", "1")
