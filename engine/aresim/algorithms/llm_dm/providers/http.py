"""Native structured-output REST adapters for the explicitly registered LLM providers.

Uses the standard library with no implicit retries, chat sessions or tool access.
API keys are headers only and never enter normalized results or persisted payloads.
"""

from __future__ import annotations

import os
import socket
import time
from dataclasses import replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from threading import Event
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener

from ..config import CHOICE_PROVIDERS, RoleConfig
from ..contracts import Json, canonical, strict_json
from .base import ProviderRequest, ProviderResult, register_provider


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Credentialed requests must stay on their configured endpoint.
        return None


_PROVIDER_OPEN = build_opener(_RejectRedirects()).open
_MAX_USAGE_COUNT = 2**31 - 1


def _retry_after(value: str | None) -> float:
    if not value:
        return 0
    if len(value) <= 12 and value.isascii() and value.replace(".", "", 1).isdecimal():
        return float(value)
    try:
        target = parsedate_to_datetime(value)
        if target.tzinfo is not None:
            return max(0, (target - datetime.now(UTC)).total_seconds())
    except (ValueError, TypeError, OverflowError):
        pass
    return 0


def _count(raw: object, key: str) -> int:
    value = raw.get(key) if isinstance(raw, dict) else None
    return value if type(value) is int and 0 <= value <= _MAX_USAGE_COUNT else 0


def _provider_label(value: object) -> str | None:
    if isinstance(value, str) and 0 < len(value) <= 256 and value.isprintable():
        return value
    return None


def _anthropic_schema(schema: Json) -> Json:
    """Use explicit branches for unions and standalone enums on the Claude wire."""
    result = dict(schema)
    if "enum" in schema:
        result.pop("type", None)
    elif isinstance(schema.get("type"), list):
        result["anyOf"] = [{"type": kind} for kind in result.pop("type")]
    if "properties" in schema:
        result["properties"] = {
            name: _anthropic_schema(field) for name, field in schema["properties"].items()
        }
    if isinstance(schema.get("items"), dict):
        result["items"] = _anthropic_schema(schema["items"])
    return result


class NativeHTTPProvider:
    """One pinned provider/model profile; a request may not override transport authority."""

    def __init__(
        self,
        config: RoleConfig,
        *,
        api_key: str | None = None,
        allow_env_fallback: bool = True,
    ) -> None:
        self.config = config
        self._api_key = (api_key or "").strip() or None
        self._allow_env_fallback = allow_env_fallback

    def _resolve_key(self) -> str:
        if self._api_key:
            return self._api_key
        if self._allow_env_fallback:
            return os.environ.get(self.config.credential_env, "").strip()
        return ""

    def preflight(self) -> Json:
        if not self._resolve_key():
            if not self._allow_env_fallback:
                raise ValueError(
                    f"API key required for {self.config.provider} ({self.config.credential_env})"
                )
            raise ValueError(
                f"set backend credential variable {self.config.credential_env}"
            )
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "route": self.config.route,
            "native_schema": "requested; endpoint validates on first generation"
            if self.config.provider not in CHOICE_PROVIDERS
            else False,
            "native_choice": self.config.provider in CHOICE_PROVIDERS,
            "transport": "stdlib-https",
        }

    def wire_request(
        self, request: ProviderRequest
    ) -> tuple[str, Json, dict[str, str]]:
        """Translate the common schema without changing its local validation semantics."""
        c = self.config
        key = self._resolve_key()
        if not key:
            raise ValueError("provider credential unavailable")
        messages = [
            {"role": "system", "content": request.system},
            {"role": "user", "content": request.user},
        ]
        schema = {
            "type": "json_schema",
            "name": f"aresim_{request.role}",
            "strict": True,
            "schema": request.schema,
        }
        headers = {"Content-Type": "application/json"}
        if c.provider in {"openai", "deepseek"}:
            url = (
                "https://api.openai.com/v1/responses"
                if c.provider == "openai"
                else "https://api.deepseek.com/responses"
            )
            if c.provider == "deepseek":
                schema = {
                    key: value for key, value in schema.items() if key != "strict"
                }
            payload = {
                "model": c.model,
                "input": messages,
                "text": {"format": schema},
                "max_output_tokens": c.max_output_tokens,
                **c.generation,
            }
            if c.provider == "openai":
                payload["store"] = False
            headers["Authorization"] = f"Bearer {key}"
        elif c.provider in CHOICE_PROVIDERS:
            if request.role != "actor":
                raise ValueError("Decision Models is an Actor-only provider")
            packet = strict_json(request.user)
            goal = packet.get("schema_version") == "aresim.llm_dm.goal_packet.v1"
            question = "goal" if goal else "action"
            options_key = "goals" if goal else "legal_actions"
            id_key = "id" if goal else "name"
            # Choice descriptions belong in criteria once; state carries supporting facts.
            state = (
                {key: value for key, value in packet.items() if key != options_key}
                if goal or packet.get("schema_version") == "aresim.llm_dm.packet.v2"
                else packet
            )
            routed = c.provider == "openrouter_decisions"
            url = (
                "https://openrouter.ai/api/alpha/decisions"
                if routed
                else "https://api.typesafe.ai/v1/systemone"
            )
            criteria = {
                entry[id_key]: {k: v for k, v in entry.items() if k != id_key}
                for entry in packet[options_key]
            }
            if routed:
                criteria = {key: canonical(value) for key, value in criteria.items()}
            payload = {
                "model": c.model,
                "state": state,
                "questions": {
                    question: {
                        "type": "choice",
                        "instructions": request.system,
                        "criteria": criteria,
                    }
                },
            }
            headers["Authorization"] = f"Bearer {key}"
        elif c.provider == "anthropic":
            url = "https://api.anthropic.com/v1/messages"
            payload = {
                "model": c.model,
                "system": request.system,
                "messages": messages[1:],
                "max_tokens": c.max_output_tokens,
                **{k: v for k, v in c.generation.items() if k != "output_config"},
                "output_config": {
                    **c.generation.get("output_config", {}),
                    "format": {
                        "type": "json_schema",
                        "schema": _anthropic_schema(request.schema),
                    },
                },
            }
            headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
        elif c.provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{quote(c.model, safe='')}:generateContent"
            payload = {
                "systemInstruction": {"parts": [{"text": request.system}]},
                "contents": [{"role": "user", "parts": [{"text": request.user}]}],
                "generationConfig": {
                    **c.generation,
                    "maxOutputTokens": c.max_output_tokens,
                    "responseMimeType": "application/json",
                    "responseJsonSchema": request.schema,
                },
            }
            headers["x-goog-api-key"] = key
        else:
            url = "https://openrouter.ai/api/v1/chat/completions"
            payload = {
                "model": c.model,
                "messages": messages,
                "max_tokens": c.max_output_tokens,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {k: v for k, v in schema.items() if k != "type"},
                },
                "provider": {
                    "order": [c.route],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                },
                **c.generation,
            }
            headers["Authorization"] = f"Bearer {key}"
        return url, payload, headers

    def generate(
        self, request: ProviderRequest, timeout: float, cancellation: Event
    ) -> ProviderResult:
        start = time.monotonic()
        if cancellation.is_set():
            return ProviderResult("error", error_code="cancelled")
        url, payload, headers = self.wire_request(request)
        wire = canonical(payload).encode()
        try:
            with _PROVIDER_OPEN(
                Request(url, wire, headers=headers),
                timeout=timeout,
            ) as response:
                body = response.read(8_000_001)
                if len(body) > 8_000_000:
                    return ProviderResult("error", error_code="response_too_large")
                data = strict_json(body)
                result = self.decode(data, request)
                result = replace(
                    result,
                    request_id=result.request_id
                    or _provider_label(response.headers.get("x-request-id"))
                    or _provider_label(response.headers.get("request-id")),
                )
        except HTTPError as error:
            code = error.code
            seconds = _retry_after(error.headers.get("Retry-After"))
            request_id = _provider_label(
                error.headers.get("x-request-id") or error.headers.get("request-id")
            )
            error.close()
            # Do not persist arbitrary error bodies, URLs or authorization headers.
            result = ProviderResult(
                "error",
                request_id=request_id,
                error_code=f"http_{code}",
                retryable=code in {408, 429, 500, 502, 503, 504}
                or (
                    code == 402
                    and seconds > 0
                    and self.config.provider.startswith("openrouter")
                ),
                retry_after_seconds=seconds,
            )
        except (TimeoutError, socket.timeout, URLError, ConnectionError, OSError):
            result = ProviderResult(
                "error", error_code="transport_timeout_or_connection", retryable=True
            )
        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            IndexError,
            OverflowError,
            RecursionError,
        ):
            result = ProviderResult("error", error_code="invalid_provider_envelope")
        return replace(
            result,
            latency_seconds=time.monotonic() - start,
            details={
                **(result.details or {}),
                "transport": {"request_bytes": len(wire)},
            },
        )

    def decode(
        self, body: Json, request: ProviderRequest | None = None
    ) -> ProviderResult:
        """Preserve finish/refusal status before extracting text; usage totals are normalized."""
        if not isinstance(body, dict):
            raise ValueError("provider response must be an object")
        provider = self.config.provider
        raw_usage = body.get("usage", {})
        error_code = None
        if provider in CHOICE_PROVIDERS:
            if request is None or request.role != "actor":
                raise ValueError(
                    "Decision Model normalization requires its Actor request"
                )
            packet = strict_json(request.user)
            goal = packet.get("schema_version") == "aresim.llm_dm.goal_packet.v1"
            answer = body.get("answers", {}).get("goal" if goal else "action")
            if (
                not isinstance(answer, dict)
                or answer.get("type") != "choice"
                or not isinstance(answer.get("choice"), str)
            ):
                return ProviderResult(
                    "error",
                    error_code="missing_or_invalid_choice",
                    usage=choice_usage(raw_usage),
                )
            option = _provider_label(answer["choice"])
            if option is None:
                return ProviderResult(
                    "error",
                    error_code="missing_or_invalid_choice",
                    usage=choice_usage(raw_usage),
                )
            options = packet["goals"] if goal else packet["legal_actions"]
            keys = {entry["id" if goal else "name"] for entry in options}
            probabilities = answer.get("probabilities")
            details = {"choice": option}
            if isinstance(probabilities, dict):
                details["probabilities"] = {
                    key: value
                    for key, value in probabilities.items()
                    if key in keys and type(value) in (int, float) and 0 <= value <= 1
                }
            confidence = answer.get("confidence")
            if type(confidence) in (int, float) and 0 <= confidence <= 1:
                details["confidence"] = confidence
            text = canonical(
                {
                    "schema_version": "aresim.llm_dm.goal.v1"
                    if goal
                    else "aresim.llm.action.v1",
                    **{
                        key: packet[key]
                        for key in ("episode_id", "decision_id", "step_index")
                    },
                    "goal_id" if goal else "action": option,
                }
            )
            return ProviderResult(
                "ok",
                text=text,
                model=_provider_label(body.get("model")) or self.config.model,
                request_id=_provider_label(body.get("id")),
                finish_reason="choice",
                usage=choice_usage(raw_usage),
                details=details,
            )
        if provider in {"openai", "deepseek"}:
            parts = [
                part
                for output in body.get("output", [])
                for part in output.get("content", [])
            ]
            refused = any(part.get("type") == "refusal" for part in parts)
            text = "".join(
                part.get("text", "")
                for part in parts
                if part.get("type") == "output_text"
            )
            status = (
                "refused"
                if refused
                else "ok"
                if body.get("status") == "completed"
                else "incomplete"
            )
            if body.get("status") == "failed":
                status, error_code = "error", "provider_failed"
            usage = normalize_usage(raw_usage, "input_tokens", "output_tokens")
            finish = body.get("status")
        elif provider == "anthropic":
            text = "".join(
                part.get("text", "")
                for part in body.get("content", [])
                if part.get("type") == "text"
            )
            finish = body.get("stop_reason")
            status = (
                "refused"
                if finish == "refusal"
                else "ok"
                if finish == "end_turn"
                else "incomplete"
            )
            usage = normalize_usage(raw_usage, "input_tokens", "output_tokens")
            if usage:
                usage["input_tokens"] += _count(
                    raw_usage, "cache_read_input_tokens"
                ) + _count(raw_usage, "cache_creation_input_tokens")
        elif provider == "gemini":
            candidate = (body.get("candidates") or [{}])[0]
            finish = candidate.get("finishReason")
            if finish == "MAX_TOKENS":
                error_code = "max_output_tokens"
            text = "".join(
                p.get("text", "")
                for p in candidate.get("content", {}).get("parts", [])
                if not p.get("thought")
            )
            status = (
                "ok"
                if finish == "STOP"
                else "refused"
                if finish in {"SAFETY", "PROHIBITED_CONTENT", "RECITATION"}
                or body.get("promptFeedback", {}).get("blockReason")
                else "incomplete"
            )
            raw_usage = body.get("usageMetadata", {})
            usage = normalize_usage(
                raw_usage, "promptTokenCount", "candidatesTokenCount"
            )
            if usage:
                usage["output_tokens"] += _count(raw_usage, "thoughtsTokenCount")
        else:
            choice = (body.get("choices") or [{}])[0]
            message = choice.get("message", {})
            finish = choice.get("finish_reason")
            if finish == "length":
                error_code = "max_output_tokens"
            text = message.get("content") or ""
            status = (
                "refused"
                if message.get("refusal") or finish == "content_filter"
                else "ok"
                if finish == "stop"
                else "incomplete"
            )
            usage = normalize_usage(raw_usage, "prompt_tokens", "completion_tokens")
        if not isinstance(text, str):
            raise ValueError("provider text must be a string")
        text.encode("utf-8")
        if usage is not None:
            incoming = raw_usage.get(
                "input_tokens_details", raw_usage.get("prompt_tokens_details")
            )
            outgoing = raw_usage.get(
                "output_tokens_details", raw_usage.get("completion_tokens_details")
            )
            usage["provider_details"] = {
                "input_tokens_details": {
                    key: _count(incoming, key)
                    for key in ("cached_tokens", "cache_write_tokens")
                },
                "output_tokens_details": {
                    "reasoning_tokens": _count(outgoing, "reasoning_tokens")
                },
            }
            if provider == "gemini":
                usage["provider_details"]["input_tokens_details"]["cached_tokens"] = (
                    _count(raw_usage, "cachedContentTokenCount")
                )
                usage["provider_details"]["output_tokens_details"][
                    "reasoning_tokens"
                ] = _count(raw_usage, "thoughtsTokenCount")
            elif provider == "anthropic":
                usage["provider_details"]["input_tokens_details"] = {
                    "cached_tokens": _count(raw_usage, "cache_read_input_tokens"),
                    "cache_write_tokens": _count(
                        raw_usage, "cache_creation_input_tokens"
                    ),
                }
        return ProviderResult(
            status,
            text,
            _provider_label(body.get("model")) or self.config.model,
            _provider_label(body.get("id", body.get("responseId"))),
            _provider_label(body.get("provider")),
            _provider_label(finish),
            usage,
            error_code=error_code,
        )

    def close(self) -> None:
        pass


def choice_usage(raw: Json) -> Json | None:
    usage = normalize_usage(raw, "input_tokens", "output_tokens")
    if usage is None:
        usage = normalize_usage(raw, "prompt_tokens", "completion_tokens")
    if usage is None and isinstance(raw, dict):
        # Decision models return typed judgments with no billed output tokens.
        return normalize_usage(
            {"input": raw.get("input_tokens", raw.get("prompt_tokens")), "output": 0},
            "input",
            "output",
        )
    return usage


def normalize_usage(raw: Json, input_key: str, output_key: str) -> Json | None:
    if not isinstance(raw, dict):
        return None
    if any(
        type(raw.get(key)) is not int or not 0 <= raw[key] <= _MAX_USAGE_COUNT
        for key in (input_key, output_key)
    ):
        return None
    return {"input_tokens": raw[input_key], "output_tokens": raw[output_key]}


def register_builtin_providers() -> None:
    """Register native wire formats once; the factory owns dispatch, never the engine."""
    for name in (
        "openai",
        "anthropic",
        "gemini",
        "openrouter",
        "deepseek",
        "decision_models",
        "openrouter_decisions",
    ):
        register_provider(name, NativeHTTPProvider)
