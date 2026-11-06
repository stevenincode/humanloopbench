"""Model runner: a single generate() over Ollama (default) or Amazon Bedrock.

Ollama (default) — set for Ollama Cloud:
    OLLAMA_HOST     e.g. https://ollama.com   (or your hosted endpoint)
    OLLAMA_API_KEY  your key
For a quick offline smoke test, run a tiny local model instead (e.g. llama3.2:1b)
with a local `ollama serve` and leave OLLAMA_HOST unset.

Bedrock (`provider="bedrock"`) — uses the standard AWS credential chain
(env vars, AWS_PROFILE / ~/.aws, SSO). Never put keys in code.
    AWS_REGION           e.g. us-east-1
    BEDROCK_MAX_TOKENS   output-token cap per call (default 512; cost guard)
"""

from __future__ import annotations

import os
import time
from typing import Any

from ollama import Client, ResponseError

PROVIDERS = ("ollama", "bedrock")

_client: Client | None = None
_bedrock_client: Any = None

# Bedrock Converse error codes worth retrying with backoff.
_BEDROCK_RETRY_CODES = {
    "ThrottlingException",
    "TooManyRequestsException",
    "ServiceUnavailableException",
    "ModelNotReadyException",
}


def _get_client() -> Client:
    global _client
    if _client is None:
        host = os.environ.get("OLLAMA_HOST")  # None -> default localhost
        key = os.environ.get("OLLAMA_API_KEY")
        headers = {"Authorization": f"Bearer {key}"} if key else None
        _client = Client(host=host, headers=headers)
    return _client


def _get_bedrock_client() -> Any:
    global _bedrock_client
    if _bedrock_client is None:
        try:
            import boto3
        except ImportError as e:  # pragma: no cover - depends on install
            raise RuntimeError("Bedrock provider needs boto3: uv sync --extra bedrock") from e
        from botocore.config import Config

        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        # Disable botocore's own retries; generate() owns the retry loop.
        _bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(retries={"max_attempts": 1, "mode": "standard"}),
        )
    return _bedrock_client


def _generate_ollama(
    model: str, prompt: str, system: str | None, temperature: float, retries: int
) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(retries):
        try:
            resp = _get_client().chat(
                model=model,
                messages=messages,
                options={"temperature": temperature},
            )
            return resp["message"]["content"].strip()
        except ResponseError as e:
            if e.status_code == 429 and attempt < retries - 1:
                time.sleep(2**attempt)  # 1, 2, 4, 8 s
                continue
            raise
    raise RuntimeError("unreachable")  # pragma: no cover


def _generate_bedrock(
    model: str, prompt: str, system: str | None, temperature: float, retries: int
) -> str:
    from botocore.exceptions import ClientError

    kwargs: dict[str, Any] = {
        "modelId": model,
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {
            "temperature": temperature,
            "maxTokens": int(os.environ.get("BEDROCK_MAX_TOKENS", "512")),
        },
    }
    if system:
        kwargs["system"] = [{"text": system}]

    for attempt in range(retries):
        try:
            resp = _get_bedrock_client().converse(**kwargs)
            parts = resp["output"]["message"]["content"]
            return "".join(p.get("text", "") for p in parts).strip()
        except ClientError as e:
            err = e.response.get("Error", {})
            code = err.get("Code", "")
            # Daily token quotas won't clear with backoff; fail fast.
            daily_quota = "per day" in err.get("Message", "").lower()
            if code in _BEDROCK_RETRY_CODES and not daily_quota and attempt < retries - 1:
                time.sleep(2**attempt)  # 1, 2, 4, 8 s
                continue
            raise
    raise RuntimeError("unreachable")  # pragma: no cover


def generate(
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.0,
    provider: str = "ollama",
    _retries: int = 5,
) -> str:
    """Return the model's text response. Deterministic by default (temp=0).

    Retries with exponential backoff on rate-limit / throttling errors.
    """
    if provider == "ollama":
        return _generate_ollama(model, prompt, system, temperature, _retries)
    if provider == "bedrock":
        return _generate_bedrock(model, prompt, system, temperature, _retries)
    raise ValueError(f"unknown provider {provider!r}; expected one of {PROVIDERS}")


def smoke_test(model: str = "gpt-oss:20b", provider: str = "ollama") -> str:
    """Tiny sanity check — verifies model access."""
    return generate(model, "Reply with exactly: ok", provider=provider)
