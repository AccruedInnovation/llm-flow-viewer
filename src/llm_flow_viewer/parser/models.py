"""Data models for parsed LLM request/response flows."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolUse:
    """A tool invocation extracted from the SSE response stream.

    Attributes:
        name: The tool name (e.g. "Read", "Execute", "Grep").
        id: The unique tool call identifier (e.g. "call_00_...").
        input: The parsed JSON input parameters for the tool call.
    """
    name: str = ""
    id: str = ""
    input: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedRequest:
    """Structured representation of an LLM request payload.

    Core components:
      - messages : list of user/assistant/tool messages
      - tools    : list of tool definitions
      - system   : list of system prompt messages
    """
    model: str = ""
    max_tokens: int = 0
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tools: List[Dict[str, Any]] = field(default_factory=list)
    system: List[Dict[str, Any]] = field(default_factory=list)

    # Additional top-level fields that may be useful for analysis
    output_config: Optional[Dict[str, Any]] = None
    thinking: Optional[Dict[str, Any]] = None
    stream: bool = False

    # Timestamps
    timestamp_end: Optional[float] = None
    timestamp_start: Optional[float] = None
    request_id: str = ""


@dataclass
class ParsedResponse:
    """Complete structured data from a DeepSeek SSE stream."""
    # Core content
    thinking: str = ""
    text: str = ""
    tool_uses: List[ToolUse] = field(default_factory=list)

    # Message metadata (from message_start)
    message_id: Optional[str] = None
    model: Optional[str] = None
    role: Optional[str] = None

    # Usage statistics (from message_start and message_delta)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None
    service_tier: Optional[str] = None

    # Termination info (from message_delta)
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None

    # Raw SSE event stream (for TUI display)
    raw_sse_events: List[Dict[str, Any]] = field(default_factory=list)

    # HTTP response info
    status_code: int = 200
    error_message: str = ""

    # Parse warning count (JSON decode errors in SSE stream that were skipped)
    sse_parse_warnings: int = 0

    # Timestamps
    timestamp_end: Optional[float] = None
    timestamp_start: Optional[float] = None
    request_id: str = ""


# ---------------------------------------------------------------------------
# Higher-level data models
# ---------------------------------------------------------------------------


@dataclass
class TokenUsage:
    """Token usage statistics for a single LLM call.

    Attributes:
        prompt_tokens: Number of tokens in the prompt (input).
        completion_tokens: Number of tokens generated (output).
        total_tokens: Total tokens used (prompt + completion).
    """
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@dataclass
class Timing:
    """Timestamps for a single LLM API call.

    Attributes:
        request_start: Unix timestamp when the request was sent.
        request_end: Unix timestamp when the request finished sending.
        response_start: Unix timestamp when the response started arriving.
        response_end: Unix timestamp when the response finished arriving.
    """
    request_start: Optional[float] = None
    request_end: Optional[float] = None
    response_start: Optional[float] = None
    response_end: Optional[float] = None


@dataclass
class ToolResult:
    """The result of a tool execution sent back to the LLM.

    Attributes:
        tool_use_id: The ID of the tool_use block this result corresponds to.
        content: The text content of the tool execution result.
        is_error: Whether the tool execution resulted in an error.
    """
    tool_use_id: str = ""
    content: str = ""
    is_error: bool = False


@dataclass
class ConnectionTiming:
    """Connection-level timing for a single LLM API call.

    Attributes:
        conn_id: Unique identifier for the server connection (flows sharing
            this ID reused the same TCP+TLS connection).
        timestamp_start: Unix timestamp when the TCP connection was initiated.
        timestamp_tls_setup: Unix timestamp when TLS handshake completed
            (None if no TLS).
        timestamp_end: Unix timestamp when the connection was closed.
    """
    conn_id: str = ""
    timestamp_start: Optional[float] = None
    timestamp_tls_setup: Optional[float] = None
    timestamp_end: Optional[float] = None


@dataclass
class LLMCall:
    """A single LLM API call, pairing a request with its response.

    Attributes:
        request_id: ULID string uniquely identifying this call.
        request: The parsed request payload.
        response: The parsed response payload.
        timing: Timing information for the call.
        token_usage: Token usage counts for the call.
        connection_timing: Connection-level timing data for the call.
    """
    request_id: str = ""
    request: Optional[ParsedRequest] = None
    response: Optional[ParsedResponse] = None
    timing: Optional[Timing] = None
    token_usage: Optional[TokenUsage] = None
    connection_timing: Optional[ConnectionTiming] = None


@dataclass
class Session:
    """A logical session containing multiple LLM calls.

    Attributes:
        index: The session index number (1-7).
        task_name: The human-readable task name (e.g. "analyze_codebase").
        model: The primary model used in this session.
        calls: List of LLMCall objects belonging to this session.
        flow_errors: List of error messages from flow reading (e.g.
            FlowReadException entries) that prevented some flows from
            being parsed.
    """
    index: int = 0
    task_name: str = ""
    model: str = ""
    calls: List[LLMCall] = field(default_factory=list)
    flow_errors: List[str] = field(default_factory=list)
