"""Streamlit login and API-key helpers for the dashboard."""

import os

import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def _configured_key(name: str) -> str:
    key = os.getenv(name, "")
    if key:
        return key
    try:
        return str(st.secrets[name])
    except Exception:
        return ""


def require_login() -> None:
    """Require a dashboard/API key when one is configured.

    If no keys are configured, local development remains open.
    """
    viewer_key = _configured_key("VIEWER_API_KEY")
    admin_key = _configured_key("ADMIN_API_KEY")
    if not viewer_key and not admin_key:
        return

    if "dashboard_api_key" not in st.session_state:
        st.session_state.dashboard_api_key = ""

    if st.session_state.dashboard_api_key not in (viewer_key, admin_key):
        st.sidebar.title("Dashboard login")
        entered = st.sidebar.text_input(
            "API key",
            type="password",
            key="dashboard_api_key_input",
        )
        if st.sidebar.button("Sign in"):
            if entered in (viewer_key, admin_key):
                st.session_state.dashboard_api_key = entered
                st.rerun()
            else:
                st.sidebar.error("Invalid API key")
        st.stop()


def api_headers() -> dict[str, str]:
    key = st.session_state.get("dashboard_api_key", "")
    return {"X-API-Key": key} if key else {}