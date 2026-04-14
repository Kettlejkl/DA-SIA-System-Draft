"""
TranslationService
──────────────────
Strategy-pattern wrapper around multiple translation providers.
Currently wired: LibreTranslate (free/self-hostable), Google Translate, DeepL.

Future target languages (e.g. Tagalog "tl") are added purely via config —
no code changes needed as long as the provider supports the language.
"""

from __future__ import annotations
import requests
from typing import Literal

Provider = Literal["libretranslate", "google", "deepl", "mock"]


class TranslationService:

    def __init__(
        self,
        provider: Provider = "libretranslate",
        libretranslate_url: str = "https://libretranslate.com",
        libretranslate_key: str = "",
        google_key: str = "",
        deepl_key: str = "",
    ):
        self.provider           = provider
        self._lt_url            = libretranslate_url.rstrip("/")
        self._lt_key            = libretranslate_key
        self._google_key        = google_key
        self._deepl_key         = deepl_key

    # ── Public API ────────────────────────────────────────────────────────────
    def translate(self, text: str, source: str = "fa", target: str = "en") -> str:
        dispatch = {
            "libretranslate": self._libretranslate,
            "google":         self._google,
            "deepl":          self._deepl,
            "mock":           self._mock,
        }
        fn = dispatch.get(self.provider)
        if fn is None:
            raise ValueError(f"Unknown provider: {self.provider}")
        return fn(text, source, target)

    # ── Provider implementations ──────────────────────────────────────────────
    def _libretranslate(self, text: str, source: str, target: str) -> str:
        """
        LibreTranslate is self-hostable and free-tier available.
        Docs: https://libretranslate.com/docs
        """
        payload = {
            "q":      text,
            "source": source,
            "target": target,
            "format": "text",
        }
        if self._lt_key:
            payload["api_key"] = self._lt_key

        resp = requests.post(f"{self._lt_url}/translate", json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()["translatedText"]

    def _google(self, text: str, source: str, target: str) -> str:
        """
        Google Cloud Translation v2 (Basic).
        Requires GOOGLE_TRANSLATE_KEY in env.
        Docs: https://cloud.google.com/translate/docs/reference/rest
        """
        if not self._google_key:
            raise RuntimeError("GOOGLE_TRANSLATE_KEY not configured")

        resp = requests.post(
            "https://translation.googleapis.com/language/translate/v2",
            params={"key": self._google_key},
            json={"q": text, "source": source, "target": target, "format": "text"},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()["data"]["translations"][0]["translatedText"]

    def _deepl(self, text: str, source: str, target: str) -> str:
        """
        DeepL API Free/Pro.
        Requires DEEPL_API_KEY in env.
        Docs: https://www.deepl.com/docs-api
        NOTE: DeepL Free API uses api-free.deepl.com
        """
        if not self._deepl_key:
            raise RuntimeError("DEEPL_API_KEY not configured")

        base = (
            "https://api-free.deepl.com"
            if self._deepl_key.endswith(":fx")
            else "https://api.deepl.com"
        )
        resp = requests.post(
            f"{base}/v2/translate",
            headers={"Authorization": f"DeepL-Auth-Key {self._deepl_key}"},
            json={"text": [text], "source_lang": source.upper(), "target_lang": target.upper()},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()["translations"][0]["text"]

    def _mock(self, text: str, source: str, target: str) -> str:
        """Development stub — returns the input with a placeholder prefix."""
        return f"[{target.upper()} translation of: {text}]"
