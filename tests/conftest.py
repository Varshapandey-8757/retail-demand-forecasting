import os

import pytest

# Lets the tests run without a .env file (the API only needs the URL at import time).
os.environ.setdefault("DATABASE_URL", "sqlite://")


@pytest.fixture(autouse=True)
def _no_api_keys(monkeypatch):
    """Tests run with access control off unless a test turns it on."""
    for name in ("VIEWER_API_KEY", "ADMIN_API_KEY"):
        monkeypatch.delenv(name, raising=False)