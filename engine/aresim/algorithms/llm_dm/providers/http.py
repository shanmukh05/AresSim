"""Native structured-output REST adapters for the explicitly registered LLM providers.

Uses the standard library with no implicit retries, chat sessions or tool access.
API keys are headers only and never enter normalized results or persisted payloads.
"""

from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import replace
from threading import Event
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from ..config import RoleConfig
from ..contracts import Json, canonical, strict_json
from .base import ProviderRequest, ProviderResult, register_provider


class NativeHTTPProvider:
    """One pinned provider/model profile; a request may not override transport authority."""

    def __init__(self, config: RoleConfig) -> None:
        self.config = config

    def preflight(self) -> Json:
        if not os.environ.get(self.config.credential_env, "").strip():
            raise ValueError(
                f"set backend credential variable {self.config.credential_env}"
            )
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "route": self.config.route,
            "native_schema": "requested; endpoint validates on first generation"
            if self.config.provider != "decision_models"
            else False,
            "native_choice": self.config.provider == "decision_models",
            "transport": "stdlib-https",
        }

    def wire_request(
        self, request: ProviderRequest
    ) -> tuple[str, Json, dict[str, str]]:
        """Translate the common schema without changing its local validation semantics."""
        c = self.config
        key = os.environ.get(c.credential_env, "").strip()
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
        elif c.provider == "decision_models":
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
            url = "https://api.typesafe.ai/v1/systemone"
            payload = {
                "model": c.model,
                "state": state,
                "questions": {
                    question: {
                        "type": "choice",
                        "instructions": request.system,
                        "criteria": {
                            entry[id_key]: {
                                key: value
                                for key, value in entry.items()
                                if key != id_key
                            }
                            for entry in packet[options_key]
                        },
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
                    "format": {"type": "json_schema", "schema": request.schema},
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
            with urlopen(
                Request(url, wire, headers=headers),
                timeout=timeout,
            ) as response:
                body = response.read(8_000_001)
                if len(body) > 8_000_000:
                    return ProviderResult("error", error_code="response_too_large")
                data = json.loads(body)
                result = self.decode(data, request)
                result = replace(
                    result,
                    request_id=result.request_id
                    or response.headers.get("x-request-id"),
                )
        except HTTPError as error:
            code = error.code
            retry_after = error.headers.get("Retry-After", "0")
            seconds = (
                float(retry_after) if retry_after.replace(".", "", 1).isdigit() else 0
            )
            # Do not persist arbitrary error bodies, URLs or authorization headers.
            result = ProviderResult(
                "error",
                error_code=f"http_{code}",
                retryable=code in {408, 429, 500, 502, 503, 504},
                retry_after_seconds=seconds,
            )
        except (TimeoutError, socket.timeout, URLError, ConnectionError, OSError):
            result = ProviderResult(
                "error", error_code="transport_timeout_or_connection", retryable=True
            )
        except (UnicodeError, ValueError, KeyError, TypeError):
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
        if provider == "decision_models":
            if request is None or request.role != "actor":
                raise ValueError(
                    "Decision Model normalization requires its Actor request"
                )
            packet = strict_json(request.user)
            goal = packet.get("schema_version") == "aresim.llm_dm.goal_packet.v1"
            answer = body["answers"]["goal" if goal else "action"]
            if (
                not isinstance(answer, dict)
                or answer.get("type") != "choice"
                or not isinstance(answer.get("choice"), str)
            ):
                raise ValueError("Decision Model response requires an action Choice")
            details = {
                key: answer[key]
                for key in ("choice", "confidence", "probabilities")
                if key in answer
            }
            canonical(details)
            text = canonical(
                {
                    "schema_version": "aresim.llm_dm.goal.v1"
                    if goal
                    else "aresim.llm.action.v1",
                    **{
                        key: packet[key]
                        for key in ("episode_id", "decision_id", "step_index")
                    },
                    "goal_id" if goal else "action": answer["choice"],
                }
            )
            return ProviderResult(
                "ok",
                text=text,
                model=body.get("model", self.config.model),
                request_id=body.get("id"),
                finish_reason="choice",
                usage=normalize_usage(raw_usage, "input_tokens", "output_tokens"),
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
                usage["input_tokens"] += int(
                    raw_usage.get("cache_read_input_tokens", 0)
                ) + int(raw_usage.get("cache_creation_input_tokens", 0))
        elif provider == "gemini":
            candidate = (body.get("candidates") or [{}])[0]
            finish = candidate.get("finishReason")
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
                usage["output_tokens"] += int(raw_usage.get("thoughtsTokenCount", 0))
        else:
            choice = (body.get("choices") or [{}])[0]
            message = choice.get("message", {})
            finish = choice.get("finish_reason")
            text = message.get("content") or ""
            status = (
                "refused"
                if message.get("refusal") or finish == "content_filter"
                else "ok"
                if finish == "stop"
                else "incomplete"
            )
            usage = normalize_usage(raw_usage, "prompt_tokens", "completion_tokens")
        if usage is not None:
            usage["provider_details"] = raw_usage
        return ProviderResult(
            status,
            text,
            body.get("model", self.config.model),
            body.get("id", body.get("responseId")),
            body.get("provider"),
            finish,
            usage,
            error_code=error_code,
        )

    def close(self) -> None:
        pass


def normalize_usage(raw: Json, input_key: str, output_key: str) -> Json | None:
    if not isinstance(raw, dict):
        return None
    if any(
        type(raw.get(key)) is not int or raw[key] < 0 for key in (input_key, output_key)
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
    ):
        register_provider(name, NativeHTTPProvider)
