"""Gemini LLM provider for the orchestrator — used when GEMINI_API_KEY is set.

Free-tier-friendly default model is gemini-2.5-flash-lite (250K TPM, 15 RPM,
1000 RPD). Compatible with the orchestrator's LLMProvider protocol so it can
be slotted in next to QwenLLMProvider (HuggingFace) and GroqLLMProvider.
"""
from __future__ import annotations

import base64 as _b64
import json as _json
import logging
from typing import Any

from orchestrator_service.domain.models import ConversationMessage, MessageRole
from orchestrator_service.domain.ports import LLMProvider, LLMResponse

logger = logging.getLogger("orchestrator.gemini_llm")


# ---------------------------------------------------------------------------
# Helpers — convert between OpenAI-style tool format and Gemini's format.
# ---------------------------------------------------------------------------


# Map of OpenAI / JSON-Schema "type" string -> Gemini types.Type enum value.
_TYPE_MAP: dict[str, str] = {
    "string": "STRING",
    "number": "NUMBER",
    "integer": "INTEGER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
    "object": "OBJECT",
}


def _openai_type_to_gemini(type_str: str) -> str:
    return _TYPE_MAP.get(type_str, "STRING")


def _openai_schema_to_gemini(schema: dict[str, Any] | None) -> Any:
    """Recursively convert an OpenAI / JSON-Schema dict to a Gemini Schema.

    Gemini Schema accepts (type, properties, required, items, description,
    enum). We build the tree with `types.Schema(**fields)`.
    """
    if not schema:
        return None
    try:
        from google.genai import types as _gtypes
    except ImportError:
        return None

    if not isinstance(schema, dict):
        return None

    type_str = schema.get("type")
    if type_str is None and "enum" in schema:
        type_str = "string"

    if type_str is None:
        return None

    g_type = _openai_type_to_gemini(type_str)
    fields: dict[str, Any] = {"type": getattr(_gtypes.Type, g_type)}
    if schema.get("description"):
        fields["description"] = schema["description"]
    if schema.get("enum"):
        fields["enum"] = list(schema["enum"])
    if type_str == "object":
        props = schema.get("properties") or {}
        if props:
            converted_props: dict[str, Any] = {}
            for name, sub in props.items():
                child = _openai_schema_to_gemini(sub)
                if child is not None:
                    converted_props[name] = child
            if converted_props:
                fields["properties"] = converted_props
        if schema.get("required"):
            fields["required"] = list(schema["required"])
    if type_str == "array" and schema.get("items"):
        item_schema = _openai_schema_to_gemini(schema["items"])
        if item_schema is not None:
            fields["items"] = item_schema
    return _gtypes.Schema(**fields)


def _openai_tools_to_gemini(tools: list[dict[str, Any]]) -> list[Any] | None:
    """Convert OpenAI-style `tools=[{"type": "function", "function": {...}}]`
    into a list of `google.genai.types.Tool` with function_declarations.
    """
    if not tools:
        return None
    try:
        from google.genai import types as _gtypes
    except ImportError:
        return None

    decls: list[Any] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        parameters = _openai_schema_to_gemini(fn.get("parameters"))
        decls.append(
            _gtypes.FunctionDeclaration(
                name=name,
                description=fn.get("description", ""),
                parameters=parameters,
            )
        )
    if not decls:
        return None
    return [_gtypes.Tool(function_declarations=decls)]


def _is_thought_signature_error(exc: BaseException) -> bool:
    """True iff this looks like the missing-thought_signature 400."""
    text = f"{exc.__class__.__name__}: {exc}".lower()
    return "thought_signature" in text or "thought signature" in text


def _domain_messages_to_gemini_contents(
    messages: list[ConversationMessage],
    *,
    strip_function_calls: bool = False,
) -> tuple[str | None, list[Any]]:
    """Convert the orchestrator's ConversationMessage list to Gemini's
    `(system_instruction, contents)` pair.

    Gemini uses role names "user" and "model" (not "assistant"/"tool"). We
    merge consecutive same-role messages; tool results become a synthetic
    "user" message that includes the function-response part so the model
    can continue the conversation.

    ``strip_function_calls`` drops prior model function calls (and the now
    orphaned tool-result messages) for the degraded one-time retry after a
    thought_signature 400 — used for sessions stored before signatures
    were persisted.
    """
    try:
        from google.genai import types as _gtypes
    except ImportError:
        return None, []

    system_instruction: str | None = None
    raw_contents: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == MessageRole.SYSTEM:
            if system_instruction is None:
                system_instruction = msg.content
            else:
                system_instruction = system_instruction + "\n\n" + msg.content
        elif msg.role == MessageRole.USER:
            raw_contents.append({
                "role": "user",
                "parts": [_gtypes.Part(text=msg.content)],
            })
        elif msg.role == MessageRole.ASSISTANT:
            parts: list[Any] = []
            if msg.content:
                parts.append(_gtypes.Part(text=msg.content))
            tool_calls = [] if strip_function_calls else (msg.tool_calls or [])
            for tc in tool_calls:
                args = tc.arguments
                if isinstance(args, str):
                    try:
                        args = _json.loads(args)
                    except _json.JSONDecodeError:
                        args = {}
                fc_kwargs: dict[str, Any] = {
                    "name": tc.tool_name,
                    "args": args if isinstance(args, dict) else {},
                }
                part_kwargs: dict[str, Any] = {"function_call": _gtypes.FunctionCall(**fc_kwargs)}
                # Replay the model's thought signature verbatim: Gemini 2.5+
                # rejects follow-up turns that echo a function call without
                # it (400 INVALID_ARGUMENT). Stored base64, sent as bytes.
                sig_b64 = getattr(tc, "thought_signature", None)
                if isinstance(sig_b64, str) and sig_b64:
                    try:
                        part_kwargs["thought_signature"] = _b64.b64decode(
                            sig_b64.encode("ascii")
                        )
                    except (ValueError, _b64.binascii.Error):
                        logger.warning(
                            "Dropping undecodable thought_signature for %s",
                            tc.tool_name,
                        )
                parts.append(_gtypes.Part(**part_kwargs))
            if parts:
                raw_contents.append({"role": "model", "parts": parts})
        elif msg.role == MessageRole.TOOL:
            if strip_function_calls:
                # Would dangle without its function call — drop it.
                continue
            content = msg.content
            if not content and msg.metadata.get("result") is not None:
                content = _json.dumps(msg.metadata["result"], default=str)
            tool_name = (
                msg.metadata.get("tool_name")
                if isinstance(msg.metadata, dict)
                else None
            ) or msg.tool_call_id or "tool"
            raw_contents.append({
                "role": "user",
                "parts": [
                    _gtypes.Part(
                        function_response=_gtypes.FunctionResponse(
                            name=tool_name,
                            response={"result": content[:8000] if content else ""},
                        )
                    )
                ],
            })

    # Merge consecutive same-role turns — Gemini rejects them.
    merged: list[Any] = []
    for c in raw_contents:
        if merged and merged[-1].role == c["role"]:
            existing_parts = list(merged[-1].parts) + list(c["parts"])
            merged[-1] = _gtypes.Content(
                role=c["role"], parts=existing_parts
            )
        else:
            merged.append(
                _gtypes.Content(role=c["role"], parts=list(c["parts"]))
            )
    return system_instruction, merged

def _gemini_response_to_llm(response: Any, fallback_model: str) -> LLMResponse:
    """Convert a `models.generate_content` response to the orchestrator's
    LLMResponse. Pulls out text content and any function_call parts.
    """
    content_chunks: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        cand_content = getattr(cand, "content", None)
        parts = getattr(cand_content, "parts", None) or []
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                content_chunks.append(text)
            fc = getattr(part, "function_call", None)
            if fc is not None:
                args = getattr(fc, "args", None) or {}
                if not isinstance(args, str):
                    try:
                        args_str = _json.dumps(args, default=str)
                    except TypeError:
                        args_str = "{}"
                else:
                    args_str = args
                # Preserve Gemini's opaque thought signature (bytes) as
                # base64 text. Dropping it makes the NEXT turn fail: the
                # API rejects replayed function calls without it
                # (400 INVALID_ARGUMENT).
                sig = getattr(part, "thought_signature", None)
                sig_b64 = (
                    _b64.b64encode(sig).decode("ascii")
                    if isinstance(sig, (bytes, bytearray)) and sig
                    else None
                )
                tool_calls.append({
                    "id": f"call_{len(tool_calls)+1}",
                    "type": "function",
                    "function": {
                        "name": getattr(fc, "name", ""),
                        "arguments": args_str,
                    },
                    "thought_signature": sig_b64,
                })

    usage = None
    um = getattr(response, "usage_metadata", None)
    if um is not None:
        usage = {
            "input": getattr(um, "prompt_token_count", 0) or 0,
            "output": getattr(um, "candidates_token_count", 0) or 0,
        }
    model = getattr(response, "model", None) or fallback_model
    return LLMResponse(
        content="\n".join(c for c in content_chunks if c).strip(),
        tool_calls=tool_calls,
        reasoning="",
        model=model,
        token_usage=usage,
    )

class GeminiLLMProvider(LLMProvider):
    """Gemini LLM provider for the orchestrator.

    Default model: gemini-2.5-flash-lite (free tier). Set GEMINI_LLM_MODEL
    to override. Implements the LLMProvider protocol — drop-in alongside
    QwenLLMProvider (HF) and GroqLLMProvider.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model_id: str = "gemini-2.5-flash-lite",
        timeout_seconds: int = 60,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> None:
        self._api_key = api_key
        self._model_id = model_id
        self._timeout = timeout_seconds
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai
            except ImportError:
                raise RuntimeError(
                    "google-genai is required. Install with: pip install google-genai"
                )
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def chat(
        self,
        messages: list[ConversationMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        from google.genai import types as _gtypes

        try:
            client = self._get_client()
        except RuntimeError as exc:
            logger.warning("Gemini LLM client unavailable: %s", exc)
            return LLMResponse(
                content=f"Error: {exc}", tool_calls=[], model=self._model_id
            )

        system_instruction, contents = _domain_messages_to_gemini_contents(messages)
        gemini_tools = _openai_tools_to_gemini(tools or [])

        config_kwargs: dict[str, Any] = {
            "temperature": (
                temperature if temperature is not None else self._temperature
            ),
            "max_output_tokens": max_tokens or self._max_tokens,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if gemini_tools:
            config_kwargs["tools"] = gemini_tools
        config = _gtypes.GenerateContentConfig(**config_kwargs)

        logger.debug(
            "Gemini LLM call: %d contents, %d tool decls, model=%s",
            len(contents),
            len(gemini_tools[0].function_declarations) if gemini_tools else 0,
            self._model_id,
        )

        try:
            response = client.models.generate_content(
                model=self._model_id,
                contents=contents,
                config=config,
            )
            return _gemini_response_to_llm(response, self._model_id)
        except Exception as exc:
            if _is_thought_signature_error(exc):
                # Pre-fix sessions replay signature-less calls. Retry once
                # with prior function calls (and orphaned results) stripped;
                # reduced context beats a hard failure for the user.
                logger.warning(
                    "thought_signature 400 — retrying once without prior "
                    "function calls"
                )
                try:
                    _, stripped = _domain_messages_to_gemini_contents(
                        messages, strip_function_calls=True
                    )
                    response = client.models.generate_content(
                        model=self._model_id,
                        contents=stripped,
                        config=config,
                    )
                    return _gemini_response_to_llm(response, self._model_id)
                except Exception as retry_exc:
                    exc = retry_exc
            logger.exception("Gemini LLM inference failed")
            return LLMResponse(
                content=f"Error: {exc}",
                tool_calls=[],
                model=self._model_id,
            )

    def health(self) -> dict[str, Any]:
        from google.genai import types as _gtypes
        try:
            client = self._get_client()
            client.models.generate_content(
                model=self._model_id,
                contents="Say ok",
                config=_gtypes.GenerateContentConfig(max_output_tokens=5),
            )
            return {
                "status": "healthy",
                "model": self._model_id,
                "provider": "gemini",
            }
        except Exception as exc:
            return {
                "status": "unhealthy",
                "model": self._model_id,
                "error": str(exc),
            }

