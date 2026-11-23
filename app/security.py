"""API-key access control with two roles.

* viewer - read forecasts, sales, run what-if scenarios and restock plans
* admin  - everything a viewer can do, plus endpoints that write to the database

Keys come from the environment (.env):  VIEWER_API_KEY, ADMIN_API_KEY.
Clients send the key in the `X-API-Key` header.
If NO key is configured, access control is switched off (local development).
"""
from __future__ import annotations

import os
import secrets

from fastapi import Header, HTTPException, status


def _keys() -> tuple[str, str]:
    return os.getenv("VIEWER_API_KEY", ""), os.getenv("ADMIN_API_KEY", "")


def _matches(candidate: str | None, expected: str) -> bool:
    return bool(expected) and bool(candidate) and secrets.compare_digest(candidate, expected)


def require_viewer(x_api_key: str | None = Header(default=None)) -> str:
    viewer, admin = _keys()
    if not viewer and not admin:
        return "anonymous"  # access control disabled
    if _matches(x_api_key, viewer):
        return "viewer"
    if _matches(x_api_key, admin):
        return "admin"
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid API key.",
        headers={"WWW-Authenticate": "ApiKey"},
    )


def require_admin(x_api_key: str | None = Header(default=None)) -> str:
    viewer, admin = _keys()
    if not viewer and not admin:
        return "anonymous"
    if _matches(x_api_key, admin):
        return "admin"
    if _matches(x_api_key, viewer):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="This endpoint requires an admin API key.")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid API key.",
        headers={"WWW-Authenticate": "ApiKey"},
    )