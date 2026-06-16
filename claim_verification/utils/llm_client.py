"""Unified LLM client for verification experiments.

Routes a model id to the right backend:

- OpenAI proper (`api.openai.com`):     `o4-mini`, `gpt-5-mini`, `gpt-5.2`, `gpt-5-nano`, ...
- Anthropic (`api.anthropic.com`):       `claude-haiku-4-5`, `claude-sonnet-4-6`, `claude-opus-4-7`, ...
- Google AI Studio:                       `gemini-2.5-flash`, `gemini-2.5-pro`, ...
- OpenRouter (`openrouter.ai/api/v1`):   `qwen/qwen-2.5-7b-instruct`, `qwen/qwen3-8b`,
                                          `google/gemma-2-9b-it`, ...

Single entrypoint:

    from claim_verification.utils.llm_client import call_llm
    text = call_llm(model_id, system="...", user="...", max_tokens=1024)

Raises `RuntimeError` on hard failure after retries. The caller is expected to
parse JSON / verify the label themselves.
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
import time
from typing import Optional

# Lazy imports so the module can be imported without every backend installed.

# Heuristic: which backend handles which model id?
ANTHROPIC_PREFIXES = ("claude-",)
GEMINI_PREFIXES = ("gemini-",)
# OpenRouter: anything that starts with "<vendor>/<model>" — they always have a slash.
# Important: OpenAI's own models like "o4-mini" or "gpt-5-mini" never contain a slash.
OPENROUTER_INDICATOR = "/"


def _is_openrouter_model(model: str) -> bool:
    return OPENROUTER_INDICATOR in model


def _is_anthropic_model(model: str) -> bool:
    return any(model.startswith(p) for p in ANTHROPIC_PREFIXES)


def _is_gemini_model(model: str) -> bool:
    return any(model.startswith(p) for p in GEMINI_PREFIXES)


# ---- backend clients (cached) ----

_openai_client = None
_anthropic_client = None
_gemini_client = None
_openrouter_client = None


def _openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        _openai_client = OpenAI(api_key=key)
    return _openai_client


def _anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        _anthropic_client = anthropic.Anthropic(api_key=key)
    return _anthropic_client


def _gemini():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) not set")
        _gemini_client = genai.Client(api_key=key)
    return _gemini_client


def _openrouter():
    global _openrouter_client
    if _openrouter_client is None:
        from openai import OpenAI
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        _openrouter_client = OpenAI(api_key=key,
                                    base_url="https://openrouter.ai/api/v1")
    return _openrouter_client


# ---- public API ----

def call_llm(model: str, system: str = "", user: str = "",
             max_tokens: int = 1024, max_retries: int = 4,
             temperature: Optional[float] = None) -> str:
    """Call the right backend for a model id and return raw assistant text."""
    if _is_anthropic_model(model):
        return _call_anthropic(model, system, user, max_tokens, max_retries, temperature)
    if _is_gemini_model(model):
        return _call_gemini(model, system, user, max_tokens, max_retries, temperature)
    if _is_openrouter_model(model):
        return _call_openai_compat(_openrouter(), model, system, user,
                                   max_tokens, max_retries, temperature)
    # Default: OpenAI proper.
    return _call_openai_proper(model, system, user, max_tokens, max_retries, temperature)


# ---- backend implementations ----

def _is_quota_error(err: Exception) -> bool:
    """Detect billing/quota errors that need indefinite waiting (until the
    user tops up the account). These are NOT normal rate-limit errors."""
    txt = str(err).lower()
    return any(k in txt for k in (
        "credit balance is too low",
        "insufficient_quota",
        "insufficient quota",
        "billing",
        "you have exceeded your current quota",
        "exceeded your monthly",
        "402",  # payment required
    ))


def _retry(label: str, fn, max_retries: int):
    """Call fn() with exponential backoff on transient errors.

    Quota / billing errors get indefinite retries with 60s waits — we want the
    pipeline to pause until the user tops up, not silently fail.
    """
    last_err = None
    quota_attempt = 0
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as e:  # noqa
            last_err = e
            if _is_quota_error(e):
                quota_attempt += 1
                # Exponential up to 5 minutes, then steady at 5 min
                wait = min(60 * (2 ** min(quota_attempt - 1, 3)), 300)
                if quota_attempt == 1 or quota_attempt % 5 == 0:
                    print(f"    [{label}] BILLING/QUOTA error — waiting {wait}s (attempt {quota_attempt}, will retry indefinitely): {str(e)[:140]}",
                          file=sys.stderr, flush=True)
                time.sleep(wait)
                continue  # don't increment normal attempt counter
            txt = str(e)
            transient = any(k in txt.lower() for k in (
                "rate limit", "ratelimit", "overloaded", "timeout", "timed out",
                "connection", "503", "502", "504", "529",
            ))
            if not transient and attempt > 0:
                raise
            if attempt == max_retries - 1:
                raise
            wait = min(2 ** attempt, 30)
            print(f"    [{label}] {type(e).__name__}: retry in {wait}s",
                  file=sys.stderr, flush=True)
            time.sleep(wait)
            attempt += 1


def _call_openai_proper(model: str, system: str, user: str,
                        max_tokens: int, max_retries: int,
                        temperature: Optional[float]) -> str:
    client = _openai()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    def go():
        # gpt-5* / o4-mini are reasoning-style: use max_completion_tokens, no temperature.
        if model.startswith(("gpt-5", "o4-", "o3-", "o1-")):
            kwargs = dict(model=model, messages=messages,
                          max_completion_tokens=max_tokens)
        else:
            kwargs = dict(model=model, messages=messages,
                          max_tokens=max_tokens)
            if temperature is not None:
                kwargs["temperature"] = temperature
        r = client.chat.completions.create(**kwargs)
        return (r.choices[0].message.content or "").strip()
    return _retry(f"openai/{model}", go, max_retries)


def _call_openai_compat(client, model: str, system: str, user: str,
                        max_tokens: int, max_retries: int,
                        temperature: Optional[float]) -> str:
    """For any OpenAI-compatible endpoint — OpenRouter, Together, etc."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    def go():
        kwargs = dict(model=model, messages=messages, max_tokens=max_tokens)
        if temperature is not None:
            kwargs["temperature"] = temperature
        r = client.chat.completions.create(**kwargs)
        return (r.choices[0].message.content or "").strip()
    return _retry(f"openrouter/{model}", go, max_retries)


def _call_anthropic(model: str, system: str, user: str,
                    max_tokens: int, max_retries: int,
                    temperature: Optional[float]) -> str:
    import anthropic  # noqa
    client = _anthropic()

    def go():
        kwargs = dict(model=model, max_tokens=max_tokens,
                      messages=[{"role": "user", "content": user}])
        if system:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature
        r = client.messages.create(**kwargs)
        return (r.content[0].text if r.content else "").strip()
    return _retry(f"anthropic/{model}", go, max_retries)


def _call_gemini(model: str, system: str, user: str,
                 max_tokens: int, max_retries: int,
                 temperature: Optional[float]) -> str:
    client = _gemini()
    prompt = (system + "\n\n" + user) if system else user

    def go():
        # google.genai automatically truncates output; max_tokens not configurable in
        # the simple call form. The caller's max_tokens is advisory here.
        r = client.models.generate_content(model=model, contents=prompt)
        return (getattr(r, "text", "") or "").strip()
    return _retry(f"gemini/{model}", go, max_retries)


# ---- json helpers (re-used by callers) ----

def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    return text


def extract_json_object(text: str):
    import json
    text = strip_code_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


# ---- async layer ----
# Uses native async SDKs (openai.AsyncOpenAI, anthropic.AsyncAnthropic).
# Gemini falls back to a thread executor (no clean async SDK).

_openai_async_client = None
_anthropic_async_client = None
_openrouter_async_client = None


def _openai_async():
    global _openai_async_client
    if _openai_async_client is None:
        from openai import AsyncOpenAI
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        _openai_async_client = AsyncOpenAI(api_key=key)
    return _openai_async_client


def _anthropic_async():
    global _anthropic_async_client
    if _anthropic_async_client is None:
        import anthropic
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        _anthropic_async_client = anthropic.AsyncAnthropic(api_key=key)
    return _anthropic_async_client


def _openrouter_async():
    global _openrouter_async_client
    if _openrouter_async_client is None:
        from openai import AsyncOpenAI
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        _openrouter_async_client = AsyncOpenAI(api_key=key,
                                               base_url="https://openrouter.ai/api/v1")
    return _openrouter_async_client


async def _retry_async(label: str, fn, max_retries: int):
    """Async retry with asyncio.sleep — same semantics as _retry but non-blocking."""
    quota_attempt = 0
    attempt = 0
    while True:
        try:
            return await fn()
        except Exception as e:
            if _is_quota_error(e):
                quota_attempt += 1
                wait = min(60 * (2 ** min(quota_attempt - 1, 3)), 300)
                if quota_attempt == 1 or quota_attempt % 5 == 0:
                    print(f"    [{label}] BILLING/QUOTA — waiting {wait}s (attempt {quota_attempt}): {str(e)[:140]}",
                          file=sys.stderr, flush=True)
                await asyncio.sleep(wait)
                continue
            txt = str(e)
            transient = any(k in txt.lower() for k in (
                "rate limit", "ratelimit", "overloaded", "timeout", "timed out",
                "connection", "503", "502", "504", "529",
            ))
            if not transient and attempt > 0:
                raise
            if attempt == max_retries - 1:
                raise
            wait = min(2 ** attempt, 30)
            print(f"    [{label}] {type(e).__name__}: retry in {wait}s", file=sys.stderr, flush=True)
            await asyncio.sleep(wait)
            attempt += 1


async def _call_openai_proper_async(model: str, system: str, user: str,
                                    max_tokens: int, max_retries: int,
                                    temperature: Optional[float]) -> str:
    client = _openai_async()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    async def go():
        if model.startswith(("gpt-5", "o4-", "o3-", "o1-")):
            kwargs = dict(model=model, messages=messages, max_completion_tokens=max_tokens)
        else:
            kwargs = dict(model=model, messages=messages, max_tokens=max_tokens)
            if temperature is not None:
                kwargs["temperature"] = temperature
        r = await client.chat.completions.create(**kwargs)
        return (r.choices[0].message.content or "").strip()
    return await _retry_async(f"openai/{model}", go, max_retries)


async def _call_openai_compat_async(client, model: str, system: str, user: str,
                                    max_tokens: int, max_retries: int,
                                    temperature: Optional[float]) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    async def go():
        kwargs = dict(model=model, messages=messages, max_tokens=max_tokens)
        if temperature is not None:
            kwargs["temperature"] = temperature
        r = await client.chat.completions.create(**kwargs)
        return (r.choices[0].message.content or "").strip()
    return await _retry_async(f"openrouter/{model}", go, max_retries)


async def _call_anthropic_async(model: str, system: str, user: str,
                                max_tokens: int, max_retries: int,
                                temperature: Optional[float]) -> str:
    client = _anthropic_async()

    async def go():
        kwargs = dict(model=model, max_tokens=max_tokens,
                      messages=[{"role": "user", "content": user}])
        if system:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature
        r = await client.messages.create(**kwargs)
        return (r.content[0].text if r.content else "").strip()
    return await _retry_async(f"anthropic/{model}", go, max_retries)


async def call_llm_async(model: str, system: str = "", user: str = "",
                         max_tokens: int = 1024, max_retries: int = 4,
                         temperature: Optional[float] = None) -> str:
    """Async version of call_llm — use inside an asyncio event loop for max throughput."""
    if _is_anthropic_model(model):
        return await _call_anthropic_async(model, system, user, max_tokens, max_retries, temperature)
    if _is_gemini_model(model):
        # google.genai async support is complex; run sync version in thread executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: _call_gemini(model, system, user, max_tokens, max_retries, temperature)
        )
    if _is_openrouter_model(model):
        return await _call_openai_compat_async(_openrouter_async(), model, system, user,
                                               max_tokens, max_retries, temperature)
    return await _call_openai_proper_async(model, system, user, max_tokens, max_retries, temperature)


# ---- self-test ----

if __name__ == "__main__":
    import json
    targets = [
        "gpt-5-mini",
        "claude-sonnet-4-6",
        "gemini-2.5-flash",
        "qwen/qwen-2.5-7b-instruct",
        "qwen/qwen3-8b",
        "google/gemma-2-9b-it",
    ]
    for m in targets:
        try:
            t = call_llm(m, system="Reply with the literal token PING_OK and nothing else.",
                         user="Now.", max_tokens=64)
            print(f"OK   {m:35s} -> {t[:60]!r}")
        except Exception as e:
            print(f"FAIL {m:35s} -> {type(e).__name__}: {e}")
