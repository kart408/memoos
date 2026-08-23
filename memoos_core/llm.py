"""
LLM access: a thin but defensive wrapper over a local Ollama server.

Two things make this more than a bare requests.post():

  1. `generate_json` uses Ollama's JSON mode *and* still assumes the
     model will sometimes ignore it — wrapping output in prose, fencing
     it in markdown, or trailing a second object. Small local models do
     all three. The repair pass here is what makes structured extraction
     viable without a frontier model.

  2. Model *roles* are separated. Extraction and chat have different
     quality requirements, so they get different models (see config).
     Callers ask for a role, not a model name.
"""

import json
from typing import Any, Optional

import requests

from . import config


class LLMError(RuntimeError):
    """Raised when the model is unreachable or returns unusable output."""


def _endpoint() -> str:
    return f"{config.OLLAMA_URL.rstrip('/')}/api/generate"


def _call(prompt: str, *, model: str, system: Optional[str], temperature: float,
          json_mode: bool, timeout: int) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if system:
        payload["system"] = system
    if json_mode:
        payload["format"] = "json"

    try:
        response = requests.post(_endpoint(), json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise LLMError(f"Ollama request failed ({model}): {exc}") from exc

    return response.json().get("response", "").strip()


def generate(prompt: str, *, system: Optional[str] = None, temperature: float = 0.7,
             model: Optional[str] = None, timeout: Optional[int] = None) -> str:
    """Free-text generation. Defaults to the chat model."""
    return _call(
        prompt,
        model=model or config.CHAT_MODEL,
        system=system,
        temperature=temperature,
        json_mode=False,
        timeout=timeout or config.LLM_TIMEOUT,
    )


def generate_reply(prompt: str, temperature: float = 0.7) -> str:
    """Backwards-compatible alias used by the chat/assistant layers."""
    return generate(prompt, temperature=temperature)


# ------------------------------------------------------------ JSON mode


def _strip_fences(raw: str) -> str:
    """Remove a leading ```json / trailing ``` if the model added them."""
    text = raw.strip()
    if not text.startswith("```"):
        return text
    text = text[3:]
    if text[:4].lower() == "json":
        text = text[4:]
    closing = text.rfind("```")
    if closing != -1:
        text = text[:closing]
    return text.strip()


def _extract_balanced(text: str) -> Optional[str]:
    """
    Pull the first complete JSON object or array out of a noisy string.

    Naively slicing between the first '{' and last '}' breaks the moment
    the model emits commentary containing braces, or two objects back to
    back. Scanning for balance — while respecting string literals and
    escapes, so a '}' inside a quoted value doesn't close the object —
    is what makes this reliable on small-model output.
    """
    start = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start is None:
        return None

    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def coerce_json(raw: str) -> Optional[Any]:
    """Best-effort parse of model output into JSON. None if unsalvageable."""
    if not raw:
        return None

    text = _strip_fences(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    candidate = _extract_balanced(text)
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def generate_json(prompt: str, *, system: Optional[str] = None, temperature: float = 0.0,
                  model: Optional[str] = None, retries: Optional[int] = None,
                  timeout: Optional[int] = None) -> Optional[Any]:
    """
    Ask for JSON and return parsed data, or None if the model never
    produced anything parseable.

    Returning None rather than raising is deliberate: extraction runs on
    every write, and one unparseable response should degrade that single
    memory's structure, not fail the ingest.
    """
    attempts = (retries if retries is not None else config.LLM_JSON_RETRIES) + 1
    active_model = model or config.EXTRACT_MODEL
    current_prompt = prompt

    for attempt in range(attempts):
        raw = _call(
            current_prompt,
            model=active_model,
            system=system,
            temperature=temperature,
            json_mode=True,
            timeout=timeout or config.LLM_TIMEOUT,
        )
        parsed = coerce_json(raw)
        if parsed is not None:
            return parsed

        # Retry with the failure shown back to the model. Seeing its own
        # bad output is a far stronger correction than repeating the ask.
        current_prompt = (
            f"{prompt}\n\n"
            f"Your previous response was not valid JSON:\n{raw[:400]}\n\n"
            f"Respond with valid JSON only. No explanation, no markdown."
        )

    return None


def is_available(model: Optional[str] = None) -> bool:
    """Whether the Ollama server is up and (optionally) has a given model."""
    try:
        response = requests.get(f"{config.OLLAMA_URL.rstrip('/')}/api/tags", timeout=5)
        response.raise_for_status()
    except requests.RequestException:
        return False
    if model is None:
        return True
    names = {m.get("name", "") for m in response.json().get("models", [])}
    # Ollama reports "mistral:latest"; callers often say "mistral".
    return model in names or f"{model}:latest" in names
