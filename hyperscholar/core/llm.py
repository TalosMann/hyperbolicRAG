"""LLM completion functions.

`make_deepseek_complete(cfg)` returns an async function with HyperRAG's
`llm_model_func` contract:

    async fn(prompt, system_prompt=None, history_messages=[], hashing_kv=None, **kw) -> str

It targets DeepSeek's OpenAI-compatible endpoint (or SiliconFlow — both are
reachable under the network constraints; OpenAI itself is TLS-blocked).
The shared LLM cache (`hashing_kv`) is honoured exactly like the upstream
implementation, so expensive extraction calls are never recomputed across
backends.

`StubLLM` is an offline test double.
"""
from __future__ import annotations

import hashlib
import json
import os


def _args_hash(*args) -> str:
    return hashlib.md5(json.dumps(args, sort_keys=True, default=str).encode()).hexdigest()


def make_deepseek_complete(cfg):
    """cfg: core.config.LLMConfig → async completion fn (DeepSeek / SiliconFlow)."""
    from openai import AsyncOpenAI

    api_key = os.environ.get(cfg.api_key_env, "") if cfg.api_key_env else "local"
    client = AsyncOpenAI(base_url=cfg.base_url, api_key=api_key or "local")
    model = cfg.model

    async def deepseek_complete(prompt, system_prompt=None, history_messages=[],
                                hashing_kv=None, **kwargs) -> str:
        kwargs.pop("keyword_extraction", None)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        if hashing_kv is not None:
            args_hash = _args_hash(model, messages)
            cached = await hashing_kv.get_by_id(args_hash)
            if cached is not None:
                return cached["return"]

        resp = await client.chat.completions.create(model=model, messages=messages, **kwargs)
        text = resp.choices[0].message.content

        if hashing_kv is not None:
            await hashing_kv.upsert({args_hash: {"return": text, "model": model}})
        return text

    return deepseek_complete


def _make_provider_complete(provider):
    """Build a completion fn from a LLMProviderConfig."""
    from openai import AsyncOpenAI
    api_key = os.environ.get(provider.api_key_env, "") if provider.api_key_env else "local"
    client = AsyncOpenAI(base_url=provider.base_url, api_key=api_key or "local")
    model = provider.model

    async def _complete(prompt, system_prompt=None, history_messages=[],
                        hashing_kv=None, **kwargs) -> str:
        kwargs.pop("keyword_extraction", None)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        if hashing_kv is not None:
            h = _args_hash(model, messages)
            cached = await hashing_kv.get_by_id(h)
            if cached is not None:
                return cached["return"]
        resp = await client.chat.completions.create(model=model, messages=messages, **kwargs)
        text = resp.choices[0].message.content
        if hashing_kv is not None:
            await hashing_kv.upsert({h: {"return": text, "model": model}})
        return text

    _complete.__name__ = provider.name
    return _complete


def build_llm_func(cfg):
    """Auto-select LLM provider from config.yaml fallback chain.

    Priority:
      1. First provider in `llm.providers` whose API key env var is set
      2. Local providers (Ollama / LM Studio) — tried if base_url is localhost
      3. Legacy single-provider fields (backward compat)

    Prints which provider was selected so you always know what's running.
    """
    # New multi-provider path
    if cfg.providers:
        for p in cfg.providers:
            # Local providers (no key needed) — check if server is reachable
            if not p.api_key_env:
                import socket
                try:
                    host = p.base_url.split("//")[1].split("/")[0].split(":")[0]
                    port = int(p.base_url.split(":")[-1].split("/")[0]) if ":" in p.base_url.split("//")[1] else 80
                    with socket.create_connection((host, port), timeout=1):
                        print(f"[LLM] Using local provider: {p.name} "
                              f"({p.base_url}, model={p.model})")
                        return _make_provider_complete(p)
                except (OSError, ValueError):
                    continue  # server not running, try next

            # Remote providers — check for API key
            key = os.environ.get(p.api_key_env, "")
            if key:
                print(f"[LLM] Using provider: {p.name} "
                      f"({p.base_url}, model={p.model})")
                return _make_provider_complete(p)

        print("[LLM] ⚠ No provider available — no API keys set and no local "
              "server reachable. Set DEEPSEEK_API_KEY, SILICONFLOW_API_KEY, "
              "or start Ollama/LM Studio.")
        raise RuntimeError("No LLM provider available.")

    # Legacy single-provider fallback
    print(f"[LLM] Using legacy provider: {cfg.provider} ({cfg.base_url})")
    return make_deepseek_complete(cfg)


class StubLLM:
    """Deterministic offline LLM for tests.

    - `responses`: optional {substring_of_prompt: canned_response}
    - default behaviour: echo a grounded-looking answer that includes a marker
      plus the first 200 chars of the prompt tail (so tests can assert that
      retrieved context actually reached the LLM).
    """

    def __init__(self, responses: dict | None = None):
        self.responses = responses or {}
        self.calls: list[dict] = []

    async def __call__(self, prompt, system_prompt=None, history_messages=[],
                       hashing_kv=None, **kwargs) -> str:
        self.calls.append({"prompt": prompt, "system": system_prompt})
        if hashing_kv is not None:
            args_hash = _args_hash("stub", prompt, system_prompt)
            cached = await hashing_kv.get_by_id(args_hash)
            if cached is not None:
                return cached["return"]
        text = None
        for key, resp in self.responses.items():
            if key in prompt or (system_prompt and key in system_prompt):
                text = resp
                break
        if text is None:
            text = f"[stub-answer] {prompt[-200:]}"
        if hashing_kv is not None:
            await hashing_kv.upsert({args_hash: {"return": text}})
        return text
