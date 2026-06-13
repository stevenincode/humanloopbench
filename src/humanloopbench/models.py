"""Model runner: a single generate() over Ollama (Cloud or local).

Set these env vars for Ollama Cloud:
    OLLAMA_HOST   e.g. https://ollama.com   (or your hosted endpoint)
    OLLAMA_API_KEY  your key

For a quick offline smoke test, run a tiny local model instead (e.g. llama3.2:1b)
with a local `ollama serve` and leave OLLAMA_HOST unset.
"""

from __future__ import annotations

import os
import time

from ollama import Client, ResponseError

_client: Client | None = None


def _get_client() -> Client:
    global _client
    if _client is None:
        host = os.environ.get("OLLAMA_HOST")  # None -> default localhost
        key = os.environ.get("OLLAMA_API_KEY")
        headers = {"Authorization": f"Bearer {key}"} if key else None
        _client = Client(host=host, headers=headers)
    return _client


def generate(
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.0,
    _retries: int = 5,
) -> str:
    """Return the model's text response. Deterministic by default (temp=0).

    Retries up to _retries times with exponential backoff on 429 rate-limit errors.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(_retries):
        try:
            resp = _get_client().chat(
                model=model,
                messages=messages,
                options={"temperature": temperature},
            )
            return resp["message"]["content"].strip()
        except ResponseError as e:
            if e.status_code == 429 and attempt < _retries - 1:
                time.sleep(2 ** attempt)  # 1, 2, 4, 8 s
                continue
            raise


def smoke_test(model: str = "gpt-oss:20b") -> str:
    """Tiny sanity check — verifies model access."""
    return generate(model, "Reply with exactly: ok")
