"""
Provider seam for the one thing this project needs from an LLM: turn a system
prompt plus a user message into text.

Two backends implement that single method:
  anthropic (default)  claude-sonnet-4-6      via ANTHROPIC_API_KEY
  gemini               gemini-2.5-flash       via GEMINI_API_KEY

Select one with the LLM_PROVIDER environment variable. Everything above this
module is provider-agnostic, so the recommender, the guardrails, the agentic
self-check, and the test harness are written once and run against either
backend. Tests substitute a FakeProvider and never touch the network.
"""

from __future__ import annotations

import os
from typing import Optional


class NLRecommenderError(Exception):
    """Raised for conditions the caller is expected to show to the user."""


class APIUnavailableError(NLRecommenderError):
    """The model API could not be reached, is misconfigured, or refused."""


class Provider:
    """Interface every backend implements."""

    name = "provider"
    model = "unknown"

    def complete(self, system: str, user: str, max_tokens: int, effort: str) -> str:
        """Return the model's text response. Raise APIUnavailableError on failure."""
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.name} ({self.model})"


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model: Optional[str] = None, client=None):
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

        try:
            import anthropic
        except ImportError as exc:
            raise APIUnavailableError(
                "The 'anthropic' package is not installed. "
                "Run: pip install -r requirements.txt"
            ) from exc
        self._sdk = anthropic

        if client is not None:
            self._client = client
            return

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise APIUnavailableError(
                "ANTHROPIC_API_KEY is not set. Either export it, switch provider "
                "with LLM_PROVIDER=gemini (plus GEMINI_API_KEY), or run the "
                "non-AI version:  python -m src.main --classic"
            )
        self._client = anthropic.Anthropic()

    def complete(self, system: str, user: str, max_tokens: int, effort: str) -> str:
        sdk = self._sdk
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                output_config={"effort": effort},
                messages=[{"role": "user", "content": user}],
            )
        except sdk.AuthenticationError as exc:
            raise APIUnavailableError(
                "The Anthropic API rejected your key. Check ANTHROPIC_API_KEY."
            ) from exc
        except sdk.RateLimitError as exc:
            raise APIUnavailableError(
                "The Anthropic API is rate limiting this key. Wait and retry."
            ) from exc
        except sdk.APIConnectionError as exc:
            raise APIUnavailableError(
                "Could not reach the Anthropic API. Check your connection."
            ) from exc
        except sdk.APIStatusError as exc:
            raise APIUnavailableError(
                f"The Anthropic API returned an error (HTTP {exc.status_code})."
            ) from exc

        if getattr(response, "stop_reason", None) == "refusal":
            raise APIUnavailableError(
                "The model declined to answer that request. Try rephrasing it."
            )

        text = "".join(
            b.text for b in response.content if getattr(b, "type", "") == "text"
        )
        return _require_text(text)


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #

# The Anthropic 'effort' levels have no direct Gemini equivalent; they are
# mapped onto a thinking-token budget instead.
_GEMINI_THINKING_BUDGET = {"low": 0, "medium": 512, "high": 2048}


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, model: Optional[str] = None, client=None):
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

        try:
            from google import genai
            from google.genai import errors as genai_errors
            from google.genai import types as genai_types
        except ImportError as exc:
            raise APIUnavailableError(
                "The 'google-genai' package is not installed. "
                "Run: pip install -r requirements.txt"
            ) from exc
        self._types = genai_types
        self._errors = genai_errors

        if client is not None:
            self._client = client
            return

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
            "GOOGLE_API_KEY"
        )
        if not api_key:
            raise APIUnavailableError(
                "GEMINI_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey, or run the non-AI version: "
                " python -m src.main --classic"
            )
        self._client = genai.Client(api_key=api_key)

    def complete(self, system: str, user: str, max_tokens: int, effort: str) -> str:
        types = self._types
        budget = _GEMINI_THINKING_BUDGET.get(effort, 512)
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens + budget,
            thinking_config=types.ThinkingConfig(thinking_budget=budget),
        )
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=user, config=config
            )
        except self._errors.ClientError as exc:
            raise APIUnavailableError(
                f"Gemini rejected the request ({exc}). Check GEMINI_API_KEY and "
                f"that GEMINI_MODEL='{self.model}' is a model your key can use."
            ) from exc
        except self._errors.ServerError as exc:
            raise APIUnavailableError(
                f"Gemini had a server-side error ({exc}). Wait and retry."
            ) from exc
        except self._errors.APIError as exc:
            raise APIUnavailableError(f"The Gemini API returned an error: {exc}") from exc
        except Exception as exc:  # network/DNS failures surface as generic errors
            raise APIUnavailableError(f"Could not reach the Gemini API: {exc}") from exc

        return _require_text(response.text or "")


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #

PROVIDERS = {"anthropic": AnthropicProvider, "gemini": GeminiProvider}


def _require_text(text: str) -> str:
    if not text.strip():
        raise APIUnavailableError("The model returned an empty response.")
    return text.strip()


def build_provider(name: Optional[str] = None, **kwargs) -> Provider:
    """Construct the configured provider. Defaults to Anthropic."""
    name = (name or os.environ.get("LLM_PROVIDER") or "anthropic").strip().lower()
    if name not in PROVIDERS:
        raise NLRecommenderError(
            f"Unknown LLM_PROVIDER '{name}'. Choose one of: "
            f"{', '.join(sorted(PROVIDERS))}."
        )
    return PROVIDERS[name](**kwargs)


class FakeProvider(Provider):
    """Scripted provider for tests — never touches the network.

    Pass a list of replies (returned in order) or a callable that receives
    (system, user) and returns a reply.
    """

    name = "fake"
    model = "fake-model"

    def __init__(self, replies):
        self._replies = replies
        self.calls = []

    def complete(self, system: str, user: str, max_tokens: int, effort: str) -> str:
        self.calls.append({"system": system, "user": user, "effort": effort})
        if callable(self._replies):
            return self._replies(system, user)
        if not self._replies:
            raise AssertionError("FakeProvider ran out of scripted replies")
        return self._replies.pop(0)
