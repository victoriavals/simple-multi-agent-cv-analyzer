from __future__ import annotations

import os


def get_setting(name: str, default: str = "") -> str:
    """Read the same setting names locally and on Streamlit Community Cloud."""
    try:
        import streamlit as st

        value = st.secrets.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    except (FileNotFoundError, RuntimeError):
        pass
    return os.getenv(name, "").strip() or default
