"""SSE response stream parser for DeepSeek Anthropic-compatible API.

Parses text/event-stream content blocks into structured data:
- Concatenates text_delta events into text output
- Concatenates thinking_delta events into thinking content
- Captures tool_use blocks (name, id, accumulated JSON input from input_json_delta)
- Extracts token usage from message_start and message_delta events
- Extracts stop_reason from message_delta
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from llm_flow_viewer.parser.models import ParsedResponse, ToolUse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level SSE parsing
# ---------------------------------------------------------------------------


def _parse_sse_events(content: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Split a raw SSE text body into (event_type, data_dict) pairs.

    Each SSE event block is separated by ``\\n\\n``.  Lines that start with
    ``event:`` carry the event type; lines that start with ``data:`` carry
    the JSON payload.  Empty and comment-only blocks are skipped.

    JSON decode errors are logged with a warning.  The caller can obtain
    the number of skipped blocks via the returned metadata.
    """
    events: List[Tuple[str, Dict[str, Any]]] = []
    blocks = content.split("\n\n")

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        event_type: Optional[str] = None
        data_str: List[str] = []

        for line in block.split("\n"):
            line = line.strip()
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_str.append(line[len("data:"):].strip())

        # Skip comment-only or unknown event blocks with no data
        if not data_str:
            continue

        full_data = "".join(data_str)
        if event_type is None:
            event_type = "message"  # default event type per SSE spec

        try:
            parsed = json.loads(full_data)
        except json.JSONDecodeError:
            logger.warning("Failed to decode SSE data as JSON: %s", full_data[:200])
            continue

        events.append((event_type, parsed))

    return events


def _parse_sse_events_with_warnings(
    content: str,
) -> Tuple[List[Tuple[str, Dict[str, Any]]], int]:
    """Same as :func:`_parse_sse_events` but also returns a warning count.

    Args:
        content: The raw SSE text body.

    Returns:
        A tuple of (events, warning_count) where *warning_count* is the
        number of data blocks that failed to decode as JSON.
    """
    events: List[Tuple[str, Dict[str, Any]]] = []
    blocks = content.split("\n\n")
    warning_count = 0

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        event_type: Optional[str] = None
        data_str: List[str] = []

        for line in block.split("\n"):
            line = line.strip()
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_str.append(line[len("data:"):].strip())

        if not data_str:
            continue

        full_data = "".join(data_str)
        if event_type is None:
            event_type = "message"

        try:
            parsed = json.loads(full_data)
        except json.JSONDecodeError:
            logger.warning("Failed to decode SSE data as JSON: %s", full_data[:200])
            warning_count += 1
            continue

        events.append((event_type, parsed))

    return events, warning_count


# ---------------------------------------------------------------------------
# High-level response parsing
# ---------------------------------------------------------------------------


def parse_response(
    raw: bytes | None,
    status_code: int = 200,
) -> ParsedResponse:
    """Parse an SSE response body into a :class:`ParsedResponse`.

    Args:
        raw: The raw byte payload from the HTTP response.
        status_code: The HTTP status code of the response.

    Returns:
        A :class:`ParsedResponse` with fields populated from the SSE stream.
    """
    response = ParsedResponse(status_code=status_code)

    if not raw:
        return response

    # Decode the response body.
    try:
        body = raw.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning("Response body is not valid UTF-8; returning empty response")
        return response

    # Parse raw SSE events, tracking JSON decode warnings.
    events, warning_count = _parse_sse_events_with_warnings(body)

    # Store raw events for TUI display (event-by-event view)
    response.raw_sse_events = [
        {"event_type": event_type, "data": data}
        for event_type, data in events
    ]

    # Record how many SSE data blocks had JSON decode failures
    response.sse_parse_warnings = warning_count

    # State accumulators for content blocks.
    text_parts: List[str] = []
    thinking_parts: List[str] = []
    tool_uses: List[ToolUse] = []

    # Current tool-use accumulator (while we are receiving input_json_delta fragments).
    current_tool_json: str = ""
    current_tool_name: str = ""
    current_tool_id: str = ""

    for event_type, data in events:
        if event_type == "message_start":
            _handle_message_start(response, data)

        elif event_type == "content_block_start":
            block = data.get("content_block", {})
            block_type = block.get("type")
            if block_type == "tool_use":
                current_tool_name = block.get("name", "")
                current_tool_id = block.get("id", "")
                current_tool_json = ""
            elif block_type == "thinking":
                pass  # Will be filled by content_block_delta events
            elif block_type == "text":
                # Some content_block_start events for text may include initial text.
                initial_text = block.get("text", "")
                if initial_text:
                    text_parts.append(initial_text)

        elif event_type == "content_block_delta":
            delta = data.get("delta", {})
            delta_type = delta.get("type")

            if delta_type == "text_delta":
                text_parts.append(delta.get("text", ""))
            elif delta_type == "thinking_delta":
                thinking_parts.append(delta.get("thinking", ""))
            elif delta_type == "input_json_delta":
                current_tool_json += delta.get("partial_json", "")
            elif delta_type == "signature_delta":
                # DeepSeek sends signature_delta for thinking blocks; ignore.
                pass

        elif event_type == "content_block_stop":
            if current_tool_id and current_tool_name:
                # Parse the accumulated JSON input.
                parsed_input: Dict[str, Any] = {}
                if current_tool_json.strip():
                    try:
                        parsed_input = json.loads(current_tool_json)
                    except json.JSONDecodeError:
                        logger.warning(
                            "Failed to parse tool input JSON for %s (%s): %s",
                            current_tool_name,
                            current_tool_id,
                            current_tool_json[:200],
                        )
                tool_uses.append(
                    ToolUse(
                        name=current_tool_name,
                        id=current_tool_id,
                        input=parsed_input,
                    )
                )
                # Reset tool accumulators.
                current_tool_name = ""
                current_tool_id = ""
                current_tool_json = ""

        elif event_type == "message_delta":
            _handle_message_delta(response, data)

        elif event_type in ("message_stop", "ping"):
            pass  # No data to extract from these events.

        else:
            logger.debug("Unknown SSE event type: %s", event_type)

    # Assemble the final response.
    response.text = "".join(text_parts)
    response.thinking = "".join(thinking_parts)
    response.tool_uses = tool_uses

    return response


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------


def _handle_message_start(response: ParsedResponse, data: Dict[str, Any]) -> None:
    """Extract message metadata and initial usage from ``message_start``."""
    message = data.get("message", {})
    response.message_id = message.get("id")
    response.model = message.get("model")
    response.role = message.get("role")

    usage = data.get("usage", {})
    if usage:
        _update_usage(response, usage, is_message_delta=False)


def _handle_message_delta(response: ParsedResponse, data: Dict[str, Any]) -> None:
    """Extract final usage and termination info from ``message_delta``.

    ``message_delta`` usage values (output_tokens, etc.) override any values
    that were previously set by ``message_start``.
    """
    delta = data.get("delta", {})
    response.stop_reason = delta.get("stop_reason")
    response.stop_sequence = delta.get("stop_sequence")

    usage = data.get("usage", {})
    if usage:
        _update_usage(response, usage, is_message_delta=True)


def _update_usage(
    response: ParsedResponse,
    usage: Dict[str, Any],
    *,
    is_message_delta: bool,
) -> None:
    """Update token usage fields from a usage dict.

    Every field present in *usage* is written to *response*.  Because
    ``message_delta`` events arrive after ``message_start`` events, the
    delta values naturally override the start values — satisfying the
    requirement that ``message_delta`` counts take precedence.

    ``output_tokens`` starts at 0 in ``message_start`` and is corrected in
    ``message_delta``, so we only update it from ``message_delta`` usage
    to avoid storing a temporary-zero and potentially missing the real value
    if the stream is truncated at ``message_start``.
    """
    if is_message_delta:
        response.output_tokens = usage.get("output_tokens")

    # These fields are present in both message_start and message_delta.
    # message_delta will overwrite message_start by virtue of arriving later.
    if "input_tokens" in usage:
        response.input_tokens = usage["input_tokens"]
    if "cache_creation_input_tokens" in usage:
        response.cache_creation_input_tokens = usage["cache_creation_input_tokens"]
    if "cache_read_input_tokens" in usage:
        response.cache_read_input_tokens = usage["cache_read_input_tokens"]
    if "service_tier" in usage:
        response.service_tier = usage.get("service_tier")
