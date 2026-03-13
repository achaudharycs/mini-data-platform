"""LLM client protocol and Anthropic implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import anthropic


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    """Unified response from an LLM call."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM backends."""

    def send(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]],
        *,
        stream_text: bool = False,
    ) -> LLMResponse:
        """Send messages to the LLM.

        If stream_text is True, text tokens are printed to stdout as they
        arrive. Tool-use blocks are always collected silently.
        """
        ...


class AnthropicClient:
    """Anthropic SDK client with tool-use and streaming support."""

    def __init__(self, api_key: str, model: str) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def send(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]],
        *,
        stream_text: bool = False,
    ) -> LLMResponse:
        if not stream_text:
            return self._send_blocking(messages, system, tools)
        return self._send_streaming(messages, system, tools)

    def _send_blocking(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=messages,
            tools=tools,
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, input=block.input)
                )

        return LLMResponse(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
        )

    def _send_streaming(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        """Stream response — print all text live, collect tool calls silently."""
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        stop_reason = ""

        current_tool_id: str | None = None
        current_tool_name: str | None = None
        current_tool_json_parts: list[str] = []
        printed_text = False

        with self._client.messages.stream(
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=messages,
            tools=tools,
        ) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        current_tool_id = block.id
                        current_tool_name = block.name
                        current_tool_json_parts = []

                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        print(delta.text, end="", flush=True)
                        text_parts.append(delta.text)
                        printed_text = True
                    elif delta.type == "input_json_delta":
                        current_tool_json_parts.append(delta.partial_json)

                elif event.type == "content_block_stop":
                    if current_tool_id is not None:
                        raw_json = "".join(current_tool_json_parts)
                        tool_input = json.loads(raw_json) if raw_json else {}
                        tool_calls.append(
                            ToolCall(
                                id=current_tool_id,
                                name=current_tool_name or "",
                                input=tool_input,
                            )
                        )
                        current_tool_id = None
                        current_tool_name = None
                        current_tool_json_parts = []

                elif event.type == "message_delta":
                    if hasattr(event, "delta") and hasattr(event.delta, "stop_reason"):
                        stop_reason = event.delta.stop_reason or ""

        if printed_text:
            print()  # newline after streamed text

        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=stop_reason,
        )
