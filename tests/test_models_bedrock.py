"""Unit tests for the Bedrock path in models.py — boto3 is fully mocked (no AWS calls)."""

from __future__ import annotations

import pytest
from botocore.exceptions import ClientError

from humanloopbench import models


class _FakeBedrock:
    """Stand-in for a boto3 bedrock-runtime client."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        r = self._responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return {"output": {"message": {"role": "assistant", "content": [{"text": r}]}}}


def _client_error(code: str, message: str = "x") -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": message}}, "Converse")


@pytest.fixture
def fake(monkeypatch):
    def _install(*responses):
        client = _FakeBedrock(responses)
        monkeypatch.setattr(models, "_get_bedrock_client", lambda: client)
        monkeypatch.setattr(models.time, "sleep", lambda s: None)
        return client

    return _install


def test_bedrock_returns_text_and_builds_converse_request(fake):
    client = fake("  Paris  ")
    out = models.generate("amazon.nova-micro-v1:0", "Q?", system="Be brief.", provider="bedrock")
    assert out == "Paris"
    req = client.calls[0]
    assert req["modelId"] == "amazon.nova-micro-v1:0"
    assert req["messages"] == [{"role": "user", "content": [{"text": "Q?"}]}]
    assert req["system"] == [{"text": "Be brief."}]
    assert req["inferenceConfig"]["temperature"] == 0.0
    assert req["inferenceConfig"]["maxTokens"] == 512


def test_bedrock_omits_system_and_respects_token_cap(fake, monkeypatch):
    monkeypatch.setenv("BEDROCK_MAX_TOKENS", "64")
    client = fake("ok")
    models.generate("m", "hi", provider="bedrock")
    assert "system" not in client.calls[0]
    assert client.calls[0]["inferenceConfig"]["maxTokens"] == 64


def test_bedrock_retries_on_throttling(fake):
    client = fake(_client_error("ThrottlingException"), _client_error("ThrottlingException"), "ok")
    assert models.generate("m", "hi", provider="bedrock") == "ok"
    assert len(client.calls) == 3


def test_bedrock_gives_up_after_retries(fake):
    fake(*[_client_error("ThrottlingException")] * 3)
    with pytest.raises(ClientError):
        models.generate("m", "hi", provider="bedrock", _retries=3)


def test_bedrock_non_throttle_error_raises_immediately(fake):
    client = fake(_client_error("AccessDeniedException"), "never")
    with pytest.raises(ClientError):
        models.generate("m", "hi", provider="bedrock")
    assert len(client.calls) == 1


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        models.generate("m", "hi", provider="nope")


def test_bedrock_daily_quota_fails_fast(fake):
    err = _client_error("ThrottlingException", "Too many tokens per day, please wait")
    client = fake(err, "never")
    with pytest.raises(ClientError):
        models.generate("m", "hi", provider="bedrock")
    assert len(client.calls) == 1
