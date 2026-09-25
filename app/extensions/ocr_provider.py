# -*- coding: utf-8 -*-
"""Captcha text recognition (OCR) for any OpenAI-compatible vision channel.

Ported and upgraded from
``openai-captcha-detection/src/ocr_any_provider.py``.

The original was a standalone, blocking CLI helper driven by ``os.getenv``
plus ``python-dotenv`` and the ``openai`` SDK. This version is integrated with
the project instead:

* **Async** — uses ``httpx.AsyncClient`` so it never blocks the event loop of
  the Playwright/hCaptcha flow it runs inside.
* **Project configuration** — resolves endpoint/key/model from ``settings``, so
  it automatically inherits whichever channel is active (the OpenAI-compatible
  relay / 中转站 provider, or GLM). No separate env plumbing.
* **Structured logging** — loguru, consistent with the rest of the codebase.
* **Accepts bytes** — can OCR an in-memory screenshot directly, not just a path.

The only hard requirement is unchanged: the channel must expose an
OpenAI-compatible ``/chat/completions`` endpoint with a vision-capable model.
"""
from __future__ import annotations

import asyncio
import base64
import io
from pathlib import Path
from typing import Any

import httpx
from loguru import logger
from PIL import Image

# Prompt asks for the captcha text only. Chinese captchas respond better to
# domestic models; the English clause is kept so mixed-language images work.
OCR_PROMPT = (
    "请对这张图片进行 OCR 识别，输出图片中的验证码文本。"
    "只输出验证码字符本身，不要任何解释、标点或多余内容。"
    "\n(For English captchas: output only the characters you see, nothing else.)"
)

# Original project used (300, 100), which visibly degraded accuracy.
DEFAULT_MAX_SIZE = (512, 256)
NOISE_CHARACTERS = ("-", "*", "\u2022", "`", "\n", " ", "'", '"')


def _coerce_secret(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "get_secret_value"):
        return str(value.get_secret_value())
    return str(value)


def resolve_ocr_channel(settings: Any) -> tuple[str, str, str, str] | None:
    """Return ``(base_url, api_key, model, label)`` for the active channel.

    Returns ``None`` when no OpenAI-compatible channel is configured, in which
    case OCR is simply skipped by callers.
    """
    provider = (getattr(settings, "LLM_PROVIDER", "") or "").strip().lower()

    if provider == "openai" and getattr(settings, "OPENAI_API_KEY", None):
        return (
            settings.OPENAI_BASE_URL,
            _coerce_secret(settings.OPENAI_API_KEY),
            settings.OPENAI_MODEL,
            "OpenAI-relay",
        )

    if provider == "glm" and getattr(settings, "GLM_API_KEY", None):
        return (
            settings.GLM_BASE_URL,
            _coerce_secret(settings.GLM_API_KEY),
            settings.GLM_MODEL,
            "GLM",
        )

    # Provider is Gemini (native protocol, not /chat/completions), but a relay
    # may still be configured explicitly — prefer it so OCR keeps working.
    if getattr(settings, "OPENAI_API_KEY", None):
        return (
            settings.OPENAI_BASE_URL,
            _coerce_secret(settings.OPENAI_API_KEY),
            settings.OPENAI_MODEL,
            "OpenAI-relay",
        )

    return None


class CaptchaOCRClient:
    """Recognize text in a captcha image via an OpenAI-compatible vision model."""

    def __init__(
        self,
        settings: Any = None,
        *,
        prompt: str = OCR_PROMPT,
        max_size: tuple[int, int] = DEFAULT_MAX_SIZE,
        max_tokens: int = 300,
    ):
        if settings is None:
            from settings import settings as project_settings

            settings = project_settings

        self._settings = settings
        self.prompt = prompt
        self.max_size = max_size
        self.max_tokens = max_tokens

        channel = resolve_ocr_channel(settings)
        if channel is None:
            self.base_url = ""
            self.api_key = ""
            self.model = ""
            self.label = ""
            self.available = False
            return

        self.base_url, self.api_key, self.model, self.label = channel
        self.available = True

    # ---------- image handling ----------
    @staticmethod
    def encode_image(image_bytes: bytes, max_size: tuple[int, int]) -> str:
        """Downscale to ``max_size`` and return a base64 PNG payload."""
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGB")
            img.thumbnail(max_size)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @staticmethod
    def clean_text(text: str) -> str:
        for character in NOISE_CHARACTERS:
            text = text.replace(character, "")
        return text.strip()

    # ---------- transport ----------
    async def _request(self, encoded_image: str) -> str | None:
        endpoint = self.base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/chat/completions"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                        },
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        timeout = float(getattr(self._settings, "OCR_TIMEOUT_SECONDS", 60.0))
        max_retries = int(getattr(self._settings, "OCR_MAX_RETRIES", 3))

        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=min(30.0, timeout))) as client:
            for attempt in range(1, max_retries + 1):
                try:
                    response = await client.post(endpoint, headers=headers, json=payload)
                    if response.is_error:
                        logger.warning(
                            "OCR | {} | attempt {}/{} failed | status={} | body={}",
                            self.label,
                            attempt,
                            max_retries,
                            response.status_code,
                            response.text[:500],
                        )
                    else:
                        content = (
                            (response.json().get("choices") or [{}])[0]
                            .get("message", {})
                            .get("content")
                        )
                        if not content:
                            raise RuntimeError("model returned empty content")
                        return self.clean_text(str(content))
                except Exception as err:  # noqa: BLE001 - retried below
                    logger.warning(
                        "OCR | {} | attempt {}/{} error: {}", self.label, attempt, max_retries, err
                    )

                if attempt < max_retries:
                    await asyncio.sleep(2 * attempt)

        logger.error("OCR | {} | giving up after {} attempt(s)", self.label, max_retries)
        return None

    # ---------- public API ----------
    async def recognize(self, image: bytes | str | Path) -> str | None:
        """Recognize captcha text. Accepts raw bytes, a path, or a str path."""
        if not self.available:
            logger.debug("OCR skipped: no OpenAI-compatible channel configured")
            return None

        try:
            if isinstance(image, (str, Path)):
                image = Path(image).read_bytes()
            encoded = self.encode_image(image, self.max_size)
        except Exception as err:  # noqa: BLE001
            logger.warning("OCR could not read image: {}", err)
            return None

        logger.debug("OCR | {} | model={} | requesting", self.label, self.model)
        return await self._request(encoded)


async def recognize_captcha_text(image: bytes | str | Path) -> str | None:
    """Convenience wrapper used by the challenge flow."""
    return await CaptchaOCRClient().recognize(image)


def main() -> None:
    """Dev entry point: ``PYTHONPATH=app python -m extensions.ocr_provider img.png``."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Captcha OCR via the project LLM channel")
    parser.add_argument("image", help="captcha image path")
    parser.add_argument("--model", default=None, help="override the resolved model")
    args = parser.parse_args()

    client = CaptchaOCRClient()
    if not client.available:
        print(
            "未配置 OpenAI 兼容通道：请设置 LLM_PROVIDER=openai 及 OPENAI_BASE_URL / "
            "OPENAI_API_KEY / OPENAI_MODEL（或 LLM_PROVIDER=glm 及 GLM_*）。"
        )
        sys.exit(1)
    if args.model:
        client.model = args.model

    print("channel:", client.label, "| model:", client.model)
    print("识别出的验证码是：", asyncio.run(client.recognize(args.image)) or "识别失败")


if __name__ == "__main__":
    main()
