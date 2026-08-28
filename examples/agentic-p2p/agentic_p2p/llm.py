"""LiteLLM access layer.

The rest of the app should only ever call `generate_response(...)` from this
module. Swapping providers/models is then a config change (`LLM_MODEL` /
`LLM_API_KEY` / `LLM_API_BASE`), never a code change.

    Agent -> llm.generate_response() -> LiteLLM -> OpenAI/Anthropic/Local/...
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised whenever the underlying LLM call fails for any reason."""


async def generate_response(
    prompt: str,
    *,
    system: Optional[str] = None,
    json_mode: bool = False,
    settings: Optional[Settings] = None,
) -> str:
    """Call the configured LLM via LiteLLM and return the text response.

    Raises LLMError on any failure (network, auth, provider error) so
    callers have a single exception type to handle, matching the
    "LLM failure -> ERROR -> Agent -> send error response" flow described
    in the design notes.
    """
    settings = settings or get_settings()

    try:
        import litellm
    except ImportError as exc:  # pragma: no cover - exercised only if litellm missing
        raise LLMError(
            "litellm is not installed. Run `pip install litellm`."
        ) from exc

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict = {
        "model": settings.llm_model,
        "messages": messages,
    }
    if settings.llm_api_key:
        kwargs["api_key"] = settings.llm_api_key
    if settings.llm_api_base:
        kwargs["api_base"] = settings.llm_api_base
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = await litellm.acompletion(**kwargs)
    except Exception as exc:  # noqa: BLE001 - provider errors vary widely
        logger.error("LLM request failed: %s", exc)
        raise LLMError(f"LLM request failed: {exc}") from exc

    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected LLM response shape: {exc}") from exc

    if not content:
        raise LLMError("LLM returned an empty response")

    return content


async def generate_json(
    prompt: str,
    *,
    system: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> dict:
    """Convenience wrapper: ask for JSON, parse it, raise LLMError if it isn't."""
    text = await generate_response(prompt, system=system, json_mode=True, settings=settings)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"LLM did not return valid JSON: {exc}\nRaw: {text[:500]}") from exc
