"""
utils/llm_client.py — Unified OpenAI-style LLM client with full call logging.

Every single LLM call is recorded in a thread-local call log that the pipeline
can retrieve and persist. This captures:
  - the exact messages sent (system + few-shot + user)
  - the raw response text
  - model, temperature, max_tokens, agent_name
  - latency, attempt count
  - any error

Backends: "anthropic" | "openai" | "openrouter"
"""

import json
import logging
import os
import re
import time
import threading
from typing import Optional

from openai import OpenAI

import config

logger = logging.getLogger(__name__)

_client: Optional[OpenAI] = None


# ══════════════════════════════════════════════════════════════════════════════
#  Typed exceptions
# ══════════════════════════════════════════════════════════════════════════════

class LLMError(RuntimeError):
    """Base for all LLM client errors."""

class LLMAPIError(LLMError):
    """HTTP / auth / rate-limit errors."""

class LLMNetworkError(LLMError):
    """Connection timeout or DNS failures."""

class LLMParseError(LLMError):
    """Empty response or JSON decode failure."""


def _classify_exc(exc: Exception) -> LLMError:
    name = type(exc).__name__.lower()
    msg  = str(exc).lower()
    if any(k in name for k in ("apierror","apiconnectionerror","autherror",
                                "ratelimiterror","apistatuserror")):
        return LLMAPIError(str(exc))
    if any(k in name for k in ("timeout","connectionerror","gaierror","sslerror")):
        return LLMNetworkError(str(exc))
    if "timeout" in msg or "connection refused" in msg or "name or service" in msg:
        return LLMNetworkError(str(exc))
    return LLMAPIError(str(exc))


# ══════════════════════════════════════════════════════════════════════════════
#  Per-call log  (thread-local accumulator)
#
#  Usage:
#    clear_call_log()               — reset before each document
#    get_call_log() -> list[dict]   — retrieve after document is done
#
#  Each entry is a dict with:
#    call_id, agent_name, model, temperature, max_tokens,
#    messages (full list), raw_response, parsed_ok,
#    latency_seconds, attempt, error (or None)
# ══════════════════════════════════════════════════════════════════════════════

_local = threading.local()

def _get_log() -> list:
    if not hasattr(_local, "call_log"):
        _local.call_log = []
    return _local.call_log

def clear_call_log():
    _local.call_log = []

def get_call_log() -> list[dict]:
    return list(_get_log())

def _record_call(entry: dict):
    _get_log().append(entry)


# ══════════════════════════════════════════════════════════════════════════════
#  Client factory
# ══════════════════════════════════════════════════════════════════════════════

def get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client

    backend = config.LLM_BACKEND.lower()

    if backend == "anthropic":
        api_key = config.ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise LLMAPIError("Set ANTHROPIC_API_KEY in config.py or as env var.")
        _client = OpenAI(
            api_key=api_key,
            base_url="https://api.anthropic.com/v1/",
            default_headers={"anthropic-version": "2023-06-01"},
        )
    elif backend == "openai":
        api_key = config.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise LLMAPIError("Set OPENAI_API_KEY in config.py or as env var.")
        _client = OpenAI(api_key=api_key)
    elif backend == "openrouter":
        api_key = config.OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise LLMAPIError("Set OPENROUTER_API_KEY in config.py or as env var.")
        extra = {"HTTP-Referer": config.OPENROUTER_SITE_URL,
                 "X-Title":      config.OPENROUTER_SITE_NAME}
        _client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={k: v for k, v in extra.items() if v},
        )
    else:
        raise ValueError(f"Unknown LLM_BACKEND {config.LLM_BACKEND!r}.")

    return _client


def reset_client():
    global _client
    _client = None


# ══════════════════════════════════════════════════════════════════════════════
#  Core call
# ══════════════════════════════════════════════════════════════════════════════

def chat(
    messages: list[dict],
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
    retries: int = 3,
    json_mode: bool = False,
    agent_name: str = "",
) -> str:
    """
    Call the LLM and return the text response.
    Every attempt is recorded in the thread-local call log.
    """
    model       = model       or config.MODELS["extractor"]
    temperature = temperature if temperature is not None else config.TEMPERATURE
    max_tokens  = max_tokens  or config.MAX_TOKENS
    client      = get_client()

    # Make a local copy; inject JSON instruction if needed
    sent_messages = list(messages)
    if json_mode:
        sent_messages = sent_messages[:-1] + [{
            **sent_messages[-1],
            "content": sent_messages[-1]["content"]
                       + "\n\nReturn ONLY valid JSON. No markdown fences, no explanation.",
        }]

    call_id   = f"{agent_name}_{int(time.time()*1000)}"
    last_exc: Optional[Exception] = None

    for attempt in range(retries):
        t_start = time.perf_counter()
        entry: dict = {
            "call_id":          call_id,
            "agent_name":       agent_name,
            "model":            model,
            "temperature":      temperature,
            "max_tokens":       max_tokens,
            "json_mode":        json_mode,
            "attempt":          attempt + 1,
            "messages":         sent_messages,   # full verbatim copy
            "raw_response":     None,
            "parsed_ok":        None,
            "latency_seconds":  None,
            "error":            None,
        }
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=sent_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content
            entry["latency_seconds"] = round(time.perf_counter() - t_start, 3)
            entry["raw_response"]    = text
            entry["parsed_ok"]       = True

            if not text or not text.strip():
                entry["error"]    = "empty_response"
                entry["parsed_ok"] = False
                _record_call(entry)
                raise LLMParseError(f"[{agent_name}] Empty response from {model}")

            logger.debug(f"[{agent_name or 'llm'}] {model} → {len(text)} chars")
            _record_call(entry)
            return text

        except LLMParseError:
            _record_call(entry)
            raise

        except Exception as exc:
            typed = _classify_exc(exc)
            last_exc = typed
            entry["latency_seconds"] = round(time.perf_counter() - t_start, 3)
            entry["error"]           = f"{type(typed).__name__}: {exc}"
            entry["parsed_ok"]       = False
            _record_call(entry)
            wait = 2 ** attempt
            logger.warning(
                f"[{agent_name or 'llm'}] Attempt {attempt+1}/{retries} "
                f"({type(typed).__name__}): {exc}. "
                + (f"Retrying in {wait}s…" if attempt < retries - 1 else "Giving up.")
            )
            if attempt < retries - 1:
                time.sleep(wait)

    raise last_exc  # type: ignore[misc]


def chat_json(
    messages: list[dict],
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
    agent_name: str = "",
) -> list | dict:
    """Call LLM and parse JSON. Records raw response in call log before parsing."""
    raw = chat(
        messages, model=model, temperature=temperature,
        max_tokens=max_tokens, json_mode=True, agent_name=agent_name,
    )
    # Update the last log entry with parse result
    log = _get_log()
    if log:
        last = log[-1]
        try:
            result = _parse_json(raw, agent_name)
            last["parsed_result"] = result
            last["parsed_ok"]     = True
            return result
        except LLMParseError as exc:
            last["parsed_ok"]    = False
            last["parse_error"]  = str(exc)
            raise
    return _parse_json(raw, agent_name)


def _parse_json(text: str, agent_name: str = "") -> list | dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMParseError(
            f"[{agent_name}] JSON decode failed: {exc}. Raw[:300]: {text[:300]}"
        ) from exc


# ══════════════════════════════════════════════════════════════════════════════
#  Embeddings
# ══════════════════════════════════════════════════════════════════════════════

def embed(texts: list[str]) -> list[list[float]]:
    """Embed via OpenAI or OpenRouter. Returns [] on failure (TF-IDF fallback)."""
    model = config.EMBEDDING_MODEL
    if not model:
        return []

    backend = getattr(config, "EMBEDDING_BACKEND", "openai").lower()
    if backend == "openrouter":
        api_key = config.OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            logger.warning("EMBEDDING_BACKEND=openrouter but no key — using TF-IDF.")
            return []
        extra = {"HTTP-Referer": config.OPENROUTER_SITE_URL,
                 "X-Title":      config.OPENROUTER_SITE_NAME}
        emb_client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={k: v for k, v in extra.items() if v},
        )
    else:
        api_key = config.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning("No OPENAI_API_KEY for embeddings — using TF-IDF.")
            return []
        emb_client = OpenAI(api_key=api_key)

    try:
        resp = emb_client.embeddings.create(model=model, input=texts)
        return [item.embedding for item in resp.data]
    except Exception as exc:
        logger.warning(f"Embedding error ({type(exc).__name__}): {exc}. Using TF-IDF.")
        return []
