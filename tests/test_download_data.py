"""Tests for scripts/download_data.py's HF_TOKEN header helper.

Offline only -- no network call, no real corpus download. What's under test
is the header construction itself: unset stays byte-identical to the
previous (headerless) behavior, set produces the Bearer header both
list_shards' requests.get and _http_file's fsspec.open pass through.
"""

from __future__ import annotations

from scripts import download_data as DD


def test_no_token_returns_empty_dict(monkeypatch):
    """Unset HF_TOKEN must be a true no-op -- an empty headers dict is
    exactly what both call sites passed before this helper existed."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert DD._hf_headers() == {}


def test_token_produces_bearer_header(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_abc123")
    assert DD._hf_headers() == {"Authorization": "Bearer hf_abc123"}


def test_empty_string_token_treated_as_unset(monkeypatch):
    """An empty-string env var (e.g. a Colab secret left blank) should not
    produce a malformed 'Bearer ' header."""
    monkeypatch.setenv("HF_TOKEN", "")
    assert DD._hf_headers() == {}
