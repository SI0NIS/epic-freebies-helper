# -*- coding: utf-8 -*-
"""Tests for the captcha OCR assist (ported from ocr_any_provider.py)."""
from __future__ import annotations

import asyncio
import base64
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from extensions.ocr_provider import (
    DEFAULT_MAX_SIZE,
    CaptchaOCRClient,
    resolve_ocr_channel,
)


class _Secret:
    def __init__(self, value: str):
        self._value = value

    def get_secret_value(self) -> str:
        return self._value


def _settings(provider: str = "openai") -> SimpleNamespace:
    return SimpleNamespace(
        LLM_PROVIDER=provider,
        OPENAI_BASE_URL="https://relay.example.com/v1",
        OPENAI_API_KEY=_Secret("sk-relay-key"),
        OPENAI_MODEL="gpt-4o-mini",
        GLM_BASE_URL="https://open.bigmodel.cn/api/paas/v4",
        GLM_API_KEY=_Secret("glm-key"),
        GLM_MODEL="glm-4.6v",
        OCR_ENABLED=True,
        OCR_TIMEOUT_SECONDS=60.0,
        OCR_MAX_RETRIES=2,
    )


def _empty_settings() -> SimpleNamespace:
    return SimpleNamespace(
        LLM_PROVIDER="gemini",
        OPENAI_BASE_URL="",
        OPENAI_API_KEY=None,
        OPENAI_MODEL="",
        GLM_BASE_URL="",
        GLM_API_KEY=None,
        GLM_MODEL="",
        OCR_ENABLED=True,
        OCR_TIMEOUT_SECONDS=60.0,
        OCR_MAX_RETRIES=1,
    )


def _png_bytes(width: int = 1600, height: int = 900) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_resolve_ocr_channel_prefers_relay_when_provider_is_openai():
    base_url, api_key, model, label = resolve_ocr_channel(_settings("openai"))

    assert base_url == "https://relay.example.com/v1"
    assert api_key == "sk-relay-key"
    assert model == "gpt-4o-mini"
    assert label == "OpenAI-relay"


def test_resolve_ocr_channel_uses_glm_channel():
    _, _, model, label = resolve_ocr_channel(_settings("glm"))

    assert model == "glm-4.6v"
    assert label == "GLM"


def test_resolve_ocr_channel_falls_back_to_relay_for_gemini_provider():
    # Gemini speaks its own protocol, but an explicitly configured relay keeps OCR usable.
    _, _, model, label = resolve_ocr_channel(_settings("gemini"))

    assert model == "gpt-4o-mini"
    assert label == "OpenAI-relay"


def test_resolve_ocr_channel_returns_none_without_openai_compatible_channel():
    assert resolve_ocr_channel(_empty_settings()) is None


def test_encode_image_downscales_to_default_max_size():
    encoded = CaptchaOCRClient.encode_image(_png_bytes(), DEFAULT_MAX_SIZE)

    decoded = Image.open(io.BytesIO(base64.b64decode(encoded)))
    assert decoded.size[0] <= DEFAULT_MAX_SIZE[0]
    assert decoded.size[1] <= DEFAULT_MAX_SIZE[1]


def test_clean_text_strips_punctuation_and_whitespace():
    assert CaptchaOCRClient.clean_text(" A-B*C`\n 12 ") == "ABC12"


def test_client_reports_unavailable_without_channel():
    client = CaptchaOCRClient(_empty_settings())

    assert client.available is False
    assert asyncio.run(client.recognize(_png_bytes())) is None


def test_client_uses_resolved_model():
    client = CaptchaOCRClient(_settings())

    assert client.available is True
    assert client.model == "gpt-4o-mini"


@pytest.mark.parametrize("provider", ["openai", "glm", "gemini"])
def test_client_available_for_every_configured_provider(provider):
    assert CaptchaOCRClient(_settings(provider)).available is True
