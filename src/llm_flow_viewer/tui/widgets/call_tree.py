"""Collapsible call tree widget.

Displays a session's LLM API calls in a Tree widget.  Supports three states:
placeholder (no session selected), loading (session being parsed), and
populated (session data displayed with expandable section nodes).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from rich.text import Text
from textual.binding import Binding
from textual.widgets import Tree

from llm_flow_viewer.parser.models import (
    ConnectionTiming,
    LLMCall,
    Session,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_CONTENT_PREVIEW_LEN = 60
"""Maximum character length for content previews in tree node labels."""

_ROLE_ICONS: dict[str, str] = {
    "user": "\U0001f464",       # 👤
    "assistant": "\U0001f916",   # 🤖
    "system": "\u2699\ufe0f",    # ⚙️
    "tool": "\U0001f527",        # 🔧
}
"""Icons used for message roles in the tree."""

_CONTENT_BLOCK_TYPES: dict[str, str] = {
    "text": "text",
    "thinking": "thinking",
    "tool_use": "tool_use",
    "tool_result": "tool_result",
    "image_url": "image_url",
}
"""Recognized content block types."""

_RESULT_PREVIEW_LEN = 120
"""Maximum character length for tool result previews in tree node labels."""

_ERROR_ICON = "\u274c"  # ❌
"""Icon used for error tool results."""

_SUCCESS_ICON = "\u2705"  # ✅
"""Icon used for successful tool results."""

_WARNING_ICON = "\u26a0\ufe0f"  # ⚠️
"""Icon used for warning indicators (e.g. SSE parse warnings)."""


# ---------------------------------------------------------------------------
# Tree node data
# ---------------------------------------------------------------------------


@dataclass
class CallTreeNodeData:
    """Data associated with a node in the CallTree.

    Attributes:
        node_type: One of ``"session"``, ``"call"``, ``"section"``,
            ``"field"``, ``"messages_header"``, ``"message"``,
            ``"content_block"``, ``"tools_header"``, ``"tool"``,
            ``"tool_input_schema"``, ``"system_header"``,
            ``"system_message"``, ``"system_text"``, ``"raw_request"``,
            ``"output_config"``, ``"tool_call_node"``,
            ``"tool_call_input"``, ``"tool_result_node"``,
            ``"placeholder"``, ``"loading"``, ``"error"``.
        call: The :class:`LLMCall` associated with this node (only for
            ``"call"`` nodes).
        call_index: The 0-based index of this call within the session
            (only for ``"call"`` nodes).
        section_type: The type of section (only for ``"section"`` nodes):
            ``"request_details"``, ``"response_details"``, ``"tool_calls"``,
            ``"timing"``, ``"token_usage"``.
        summary: A short summary string for display purposes.
        field_key: Key name for ``"field"`` nodes (e.g. ``"Model"``).
        field_value: String value for ``"field"`` nodes.
        message_role: Role string for ``"message"`` nodes.
        content_block_type: Type string for ``"content_block"`` nodes.
        content_preview: Truncated preview text for display in tree label.
        full_content: Full content string (for detail panel rendering).
        tool_name: Tool name for ``"tool"`` nodes.
        tool_description: Tool description for ``"tool"`` nodes.
        tool_call_id: The tool call ID (e.g. ``"call_00_UcPPI..."``) for
            ``"tool_call_node"`` and ``"tool_result_node"`` nodes.
        is_error: Whether this node represents an error state (used for
            ``"tool_result_node"`` nodes with error results).
        message_index: Index of the message within the messages list.
        block_index: Index of the content block within a message.
    """

    node_type: str = "placeholder"
    call: Optional[LLMCall] = None
    call_index: int = 0
    section_type: Optional[str] = None
    summary: str = ""
    field_key: str = ""
    field_value: str = ""
    message_role: str = ""
    content_block_type: str = ""
    content_preview: str = ""
    full_content: str = ""
    tool_name: str = ""
    tool_description: str = ""
    tool_call_id: str = ""
    is_error: bool = False
    message_index: int = 0
    block_index: int = 0


# ---------------------------------------------------------------------------
# Section configuration
# ---------------------------------------------------------------------------

_SECTION_LABELS = {
    "request_details": "Request Details",
    "response_details": "Response Details",
    "tool_calls": "Tool Calls & Results",
    "timing": "Timing",
    "token_usage": "Token Usage",
}


def _should_include_section(section_type: str, call: LLMCall) -> bool:
    """Determine whether a section should be shown for a given call.

    Sections are conditionally rendered based on data availability:

    * ``request_details`` — shown when the call has a parsed request.
    * ``response_details`` — shown when the call has a parsed response
      with non-empty content.
    * ``tool_calls`` — shown when the response has one or more tool uses.
    * ``timing`` — shown when timing data is present.
    * ``token_usage`` — shown when token usage data is present.

    Args:
        section_type: The section type identifier.
        call: The LLMCall to check.

    Returns:
        ``True`` if the section should be included.
    """
    if section_type == "request_details":
        return call.request is not None

    if section_type == "response_details":
        if call.response is None:
            return False
        # Show if there's any response content (text, thinking, tool uses,
        # raw SSE events, or a non-200 error status)
        resp = call.response
        if resp.status_code != 200 or resp.error_message:
            return True
        return bool(resp.text or resp.thinking or resp.tool_uses or resp.raw_sse_events)

    if section_type == "tool_calls":
        if call.response is None:
            return False
        return len(call.response.tool_uses) > 0

    if section_type == "timing":
        return call.timing is not None

    if section_type == "token_usage":
        return call.token_usage is not None

    return False


def _call_node_label(call: LLMCall, index: int) -> str:
    """Generate a human-readable label for a call node.

    Args:
        call: The LLMCall to label.
        index: The 0-based call index (displayed as 1-based).

    Returns:
        A label string such as ``"Call #1 — deepseek-v4-flash"``.
    """
    model = ""
    if call.request and call.request.model:
        model = call.request.model

    if model:
        return f"Call #{index + 1} — {model}"
    return f"Call #{index + 1}"


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _format_thousands(value: int) -> str:
    """Format an integer with thousands-separator commas.

    Args:
        value: The integer to format.

    Returns:
        A string like ``"131,072"``.
    """
    return f"{value:,}"


def _truncate(text: str, max_len: int = _MAX_CONTENT_PREVIEW_LEN) -> str:
    """Truncate *text* to *max_len* characters, appending ``...`` if needed.

    Args:
        text: The text to truncate.
        max_len: Maximum character count before truncation.

    Returns:
        The truncated string or the original if shorter than *max_len*.
    """
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _get_role_icon(role: str) -> str:
    """Return the icon character for a given message role.

    Args:
        role: The message role (``"user"``, ``"assistant"``, etc.).

    Returns:
        An icon character.
    """
    return _ROLE_ICONS.get(role, "\U0001f4ac")  # default speech balloon 💬


def _content_preview_from_messages(
    messages: List[Dict[str, Any]],
) -> str:
    """Build a preview string for the messages list showing role and content hint.

    Args:
        messages: The messages list from the parsed request.

    Returns:
        A preview string.
    """
    count = len(messages)
    if count == 0:
        return "Messages (0)"
    first_role = messages[0].get("role", "unknown")
    return f"Messages ({count}) — first: {_get_role_icon(first_role)} {first_role}"


def _get_message_content_preview(message: Dict[str, Any]) -> str:
    """Get a truncated content preview from a message dict.

    Handles both plain-string content and content-block arrays.

    Args:
        message: A message dict with ``role`` and ``content`` keys.

    Returns:
        A truncated content preview string.
    """
    content = message.get("content", "")
    if isinstance(content, str):
        return _truncate(content)
    if isinstance(content, list):
        if not content:
            return "(empty)"
        return _get_content_block_preview(content[0])
    return ""


def _get_content_block_preview(block: Dict[str, Any]) -> str:
    """Get a preview string for a single content block.

    Args:
        block: A content block dict with a ``type`` key.

    Returns:
        A short description of the block.
    """
    block_type = block.get("type", "unknown")
    if block_type == "text":
        return _truncate(block.get("text", ""))
    if block_type == "thinking":
        return _truncate(block.get("thinking", ""))
    if block_type == "tool_use":
        name = block.get("name", "?")
        # Show a brief input preview if available
        inp = block.get("input", {})
        if isinstance(inp, dict) and inp:
            first_key = next(iter(inp.keys()))
            first_val = inp[first_key]
            val_str = str(first_val)
            if len(val_str) > 30:
                val_str = val_str[:30] + "..."
            return f"{name}({first_key}={val_str})"
        return name
    if block_type == "tool_result":
        content = block.get("content", "")
        if isinstance(content, str):
            return _truncate(content)
        if isinstance(content, list):
            # tool_result content can be a list of text blocks
            parts = [c.get("text", "") if isinstance(c, dict) else str(c) for c in content]
            return _truncate(" ".join(parts))
        return "(tool result)"
    if block_type == "image_url":
        url = block.get("image_url", {})
        if isinstance(url, dict):
            return _truncate(url.get("url", ""))
        return _truncate(str(url))
    return f"[{block_type}]"


def _get_content_type_label(block_type: str) -> str:
    """Return a display label for a content block type.

    Square brackets are escaped for Rich markup compatibility.

    Args:
        block_type: The content block type string.

    Returns:
        A formatted type label like ``"[text]"``.
    """
    # Escape brackets to prevent Rich from interpreting them as markup
    return f"\\[{block_type}\\]"


def _should_show_content_blocks(message: Dict[str, Any]) -> bool:
    """Check if a message's content is a list of content blocks.

    Args:
        message: The message dict.

    Returns:
        ``True`` if content is a list (content blocks).
    """
    return isinstance(message.get("content"), list)


# ---------------------------------------------------------------------------
# Tool result helpers
# ---------------------------------------------------------------------------


def _flatten_tool_result_content(content: Any) -> str:
    """Flatten a tool_result content value into a string.

    Handles both plain strings and lists of text blocks.

    Args:
        content: The tool result content (string, list, or other).

    Returns:
        A flattened string representation.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return " ".join(parts)
    if content is None:
        return ""
    return str(content)


def _find_tool_result(
    tool_use_id: str,
    session: Session,
    call_idx: int,
) -> Optional[Dict[str, Any]]:
    """Find a matching tool_result content block in subsequent calls.

    Searches through user messages in requests of calls after *call_idx*
    for a ``tool_result`` content block whose ``tool_use_id`` matches
    *tool_use_id*.

    Args:
        tool_use_id: The tool call ID to match (e.g. ``"call_00_..."``).
        session: The session containing all calls.
        call_idx: The index of the current call in ``session.calls``.

    Returns:
        The matching tool_result content block dict, or ``None``.
    """
    # Look through subsequent calls' request messages for matching tool_result
    for next_idx in range(call_idx + 1, len(session.calls)):
        next_call = session.calls[next_idx]
        if next_call.request is None:
            continue
        for message in next_call.request.messages:
            content = message.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result" and block.get("tool_use_id") == tool_use_id:
                    return block
    return None


def _find_tool_results_for_call(
    call: LLMCall,
    session: Session,
    call_idx: int,
) -> Dict[str, Dict[str, Any]]:
    """Find all tool results matching the tool_uses in a given call.

    Maps each ``ToolUse.id`` to its corresponding ``tool_result`` block
    from subsequent calls.

    Args:
        call: The LLMCall whose tool_uses to look up.
        session: The session containing all calls.
        call_idx: The index of *call* in ``session.calls``.

    Returns:
        A dict mapping ``tool_use_id`` → tool_result block dict.
    """
    if call.response is None:
        return {}
    results: Dict[str, Dict[str, Any]] = {}
    for tool_use in call.response.tool_uses:
        result = _find_tool_result(tool_use.id, session, call_idx)
        if result is not None:
            results[tool_use.id] = result
    return results


# ---------------------------------------------------------------------------
# Tool Calls section population
# ---------------------------------------------------------------------------


def _add_tool_calls(
    section_node,
    call: LLMCall,
    session: Session | None,
    call_idx: int,
) -> None:
    """Populate a Tool Calls & Results section node with children.

    For each ``ToolUse`` in the call's response, creates an expandable
    tool call node labeled with the tool name and call ID.  Expanding
    a tool call node reveals:

    * **Input Parameters** — the tool's input arguments rendered as
      syntax-highlighted JSON (leaf node; select to view in detail panel).
    * **Result** — the corresponding tool_result content block matched
      by ``tool_use.id`` ↔ ``tool_result.tool_use_id`` from a subsequent
      call's request messages.  The result label shows a truncated
      preview (~120 chars) with a character-count indicator for long
      results.  Error results (``is_error: true``) are displayed with
      a red ``❌`` icon and include exit code, failed command, and
      error message details.

    When *session* is ``None`` (during incremental streaming), result
    lookups are skipped — each tool use shows
    ``"→ (no result recorded)"``.

    Args:
        section_node: The ``TreeNode`` for the Tool Calls & Results
            section.
        call: The :class:`LLMCall` whose tool_uses to display.
        session: The :class:`Session` containing all calls (used to
            look up tool results in subsequent calls), or ``None``
            during streaming.
        call_idx: The index of *call* in ``session.calls``.
    """
    if call.response is None:
        return

    tool_uses = call.response.tool_uses
    if not tool_uses:
        return

    # Pre-find all matching tool results for efficiency
    matched_results = _find_tool_results_for_call(call, session, call_idx) if session is not None else {}

    for tool_use in tool_uses:
        tool_call_id = tool_use.id or "?"
        tool_name = tool_use.name or "?"

        # -- Tool call node --
        icon = _ROLE_ICONS.get("tool", "\U0001f527")  # 🔧
        call_label = f"{icon} {tool_name} ({tool_call_id})"
        tool_call_data = CallTreeNodeData(
            node_type="tool_call_node",
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            summary=call_label,
        )
        tool_call_node = section_node.add(
            call_label,
            data=tool_call_data,
            allow_expand=True,
        )

        # -- Input Parameters child --
        input_json = json.dumps(tool_use.input, indent=2) if tool_use.input else "{}"
        input_data = CallTreeNodeData(
            node_type="tool_call_input",
            full_content=input_json,
            summary="Input Parameters (JSON)",
            tool_name=tool_name,
            tool_call_id=tool_call_id,
        )
        tool_call_node.add(
            "Input Parameters (JSON)",
            data=input_data,
            allow_expand=False,
        )

        # -- Result child (if matching tool_result found) --
        result_block = matched_results.get(tool_use.id)
        if result_block is not None:
            # Extract content
            raw_content = result_block.get("content", "")
            flat_content = _flatten_tool_result_content(raw_content)
            char_count = len(flat_content)
            is_error = bool(result_block.get("is_error", False))

            # Build preview
            preview = _truncate(flat_content, max_len=_RESULT_PREVIEW_LEN)
            if char_count > _RESULT_PREVIEW_LEN:
                length_str = f" ({_format_thousands(char_count)} characters)"
            else:
                length_str = ""

            # Build result label
            if is_error:
                error_icon = _ERROR_ICON
                # Extract error details if available
                exit_code = result_block.get("exit_code", "")
                command = result_block.get("command", "")
                error_msg = result_block.get("error", "")
                error_detail_parts = []
                if exit_code:
                    error_detail_parts.append(f"Exit Code: {exit_code}")
                if command:
                    error_detail_parts.append(f"Command: {command}")
                if error_msg:
                    error_detail_parts.append(f"Error: {error_msg}")
                result_preview = f"{error_icon} {tool_name} failed"
                if error_detail_parts:
                    result_preview += " — " + " | ".join(error_detail_parts)
                # Full content includes error details + actual content
                error_details = "\n".join(error_detail_parts)
                if error_details and flat_content:
                    result_full = f"{error_details}\n\n---\n\n{flat_content}"
                elif error_details:
                    result_full = error_details
                else:
                    result_full = flat_content
            else:
                error_icon = ""
                result_preview = f"{_SUCCESS_ICON} {preview}{length_str}"  # ✅ with truncated content
                result_full = flat_content

            result_data = CallTreeNodeData(
                node_type="tool_result_node",
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                summary=result_preview,
                full_content=result_full,
                content_preview=preview,
                is_error=is_error,
            )

            if is_error:
                # Use styled Text for error results
                styled_label = Text(f"{_ERROR_ICON} Result: {preview}{length_str}", style="bold red")
                tool_call_node.add(styled_label, data=result_data, allow_expand=False)
            else:
                tool_call_node.add(result_preview, data=result_data, allow_expand=False)
        else:
            # No matching result found — show pending indicator
            pending_data = CallTreeNodeData(
                node_type="tool_result_node",
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                summary="\u2192 (no result recorded)",
                full_content="",
            )
            tool_call_node.add(
                "\u2192 (no result recorded)",
                data=pending_data,
                allow_expand=False,
            )


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string.

    Durations < 1 second are shown in milliseconds (e.g. ``"847ms"``).
    Durations >= 1 second are shown with one decimal place (e.g. ``"2.1s"``).

    Args:
        seconds: Duration in seconds.

    Returns:
        A human-readable duration string.
    """
    if seconds < 1.0:
        ms = int(round(seconds * 1000))
        return f"{ms}ms"
    return f"{seconds:.1f}s"


def _format_timestamp(unix_ts: float) -> str:
    """Format a Unix timestamp as a human-readable datetime.

    Returns a string combining the Unix timestamp and the ISO-style datetime,
    e.g. ``"1000000000.000 | 2001-09-09 01:46:40.000"``.

    Args:
        unix_ts: Unix timestamp in seconds (with optional fractional part).

    Returns:
        A formatted string.
    """
    try:
        dt = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
        dt_str = dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d}"
        return f"{unix_ts:.3f} | {dt_str}"
    except (OSError, ValueError, OverflowError):
        return f"{unix_ts:.3f}"


def _format_connection_timing(
    conn_start: Optional[float],
    conn_tls_setup: Optional[float],
    conn_end: Optional[float],
    shared_flow_count: int,
) -> str:
    """Format connection-level timing information for display.

    Args:
        conn_start: Timestamp of TCP connection initiation (or None).
        conn_tls_setup: Timestamp of TLS handshake completion (or None).
        conn_end: Timestamp of connection close (or None).
        shared_flow_count: Number of flows sharing this connection.

    Returns:
        A formatted string with connection timing details, or an empty string
        if no connection data is available.
    """
    if conn_start is None:
        return ""

    parts = []

    # TLS setup time
    if conn_tls_setup is not None:
        tls_time = conn_tls_setup - conn_start
        parts.append(f"TLS setup: {_format_duration(tls_time)}")

    # Total connection lifetime
    if conn_end is not None:
        lifetime = conn_end - conn_start
        parts.append(f"Lifetime: {_format_duration(lifetime)}")

    # Connection reuse
    if shared_flow_count > 1:
        parts.append(f"Shared across {shared_flow_count} calls")
    elif shared_flow_count == 1:
        parts.append("Single use")

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Timing section population
# ---------------------------------------------------------------------------


def _add_timing_section(section_node, call: LLMCall) -> None:
    """Populate a Timing section node with children.

    Displays raw timestamps (both Unix and human-readable) and derived
    timing metrics.

    Raw timestamps shown:
    * Request Start — when the HTTP request began sending
    * Request End — when the request body finished sending
    * Response Start — when the first byte of the response arrived
    * Response End — when the response body finished arriving

    Derived metrics:
    * Request Duration — time from request start to request end (ms)
    * Response Duration — time from response start to response end (s)
    * TTFB (Time to First Byte) — time from request end to response start (ms)
    * Total RTT (Round-Trip Time) — time from request start to response end (s)

    Args:
        section_node: The ``TreeNode`` for the Timing section.
        call: The :class:`LLMCall` containing timing data.
    """
    timing = call.timing
    if timing is None:
        section_node.add(
            "No timing data available",
            data=CallTreeNodeData(node_type="field", summary="No timing data available"),
            allow_expand=False,
        )
        return

    # -- Raw timestamp cluster --
    timestamp_header_data = CallTreeNodeData(
        node_type="field",
        field_key="Timestamps",
        summary="Raw Timestamps",
    )
    ts_header = section_node.add(
        "Raw Timestamps",
        data=timestamp_header_data,
        allow_expand=True,
    )

    if timing.request_start is not None:
        ts_str = _format_timestamp(timing.request_start)
        ts_data = CallTreeNodeData(
            node_type="field",
            field_key="Request Start",
            field_value=ts_str,
            summary=f"Request Start: {ts_str}",
            full_content=ts_str,
        )
        ts_header.add(f"Request Start: {ts_str}", data=ts_data, allow_expand=False)

    if timing.request_end is not None:
        ts_str = _format_timestamp(timing.request_end)
        ts_data = CallTreeNodeData(
            node_type="field",
            field_key="Request End",
            field_value=ts_str,
            summary=f"Request End: {ts_str}",
            full_content=ts_str,
        )
        ts_header.add(f"Request End: {ts_str}", data=ts_data, allow_expand=False)

    if timing.response_start is not None:
        ts_str = _format_timestamp(timing.response_start)
        ts_data = CallTreeNodeData(
            node_type="field",
            field_key="Response Start (First Byte)",
            field_value=ts_str,
            summary=f"Response Start: {ts_str}",
            full_content=ts_str,
        )
        ts_header.add(f"Response Start: {ts_str}", data=ts_data, allow_expand=False)

    if timing.response_end is not None:
        ts_str = _format_timestamp(timing.response_end)
        ts_data = CallTreeNodeData(
            node_type="field",
            field_key="Response End",
            field_value=ts_str,
            summary=f"Response End: {ts_str}",
            full_content=ts_str,
        )
        ts_header.add(f"Response End: {ts_str}", data=ts_data, allow_expand=False)

    # -- Derived metrics --
    metrics_header_data = CallTreeNodeData(
        node_type="field",
        field_key="Metrics",
        summary="Derived Metrics",
    )
    metrics_header = section_node.add(
        "Derived Metrics",
        data=metrics_header_data,
        allow_expand=True,
    )

    # Request Duration (request_end - request_start) in ms
    if timing.request_start is not None and timing.request_end is not None:
        req_dur = timing.request_end - timing.request_start
        req_dur_str = _format_duration(req_dur)
        req_dur_data = CallTreeNodeData(
            node_type="field",
            field_key="Request Duration",
            field_value=req_dur_str,
            summary=f"Request Duration: {req_dur_str}",
            full_content=req_dur_str,
        )
        metrics_header.add(
            f"Request Duration: {req_dur_str}",
            data=req_dur_data,
            allow_expand=False,
        )

    # Response Duration (response_end - response_start) in s
    if timing.response_start is not None and timing.response_end is not None:
        resp_dur = timing.response_end - timing.response_start
        resp_dur_str = _format_duration(resp_dur)
        resp_dur_data = CallTreeNodeData(
            node_type="field",
            field_key="Response Duration",
            field_value=resp_dur_str,
            summary=f"Response Duration: {resp_dur_str}",
            full_content=resp_dur_str,
        )
        metrics_header.add(
            f"Response Duration: {resp_dur_str}",
            data=resp_dur_data,
            allow_expand=False,
        )

    # TTFB (response_start - request_end) in ms
    if timing.request_end is not None and timing.response_start is not None:
        ttfb = timing.response_start - timing.request_end
        ttfb_str = _format_duration(ttfb)
        ttfb_data = CallTreeNodeData(
            node_type="field",
            field_key="TTFB",
            field_value=ttfb_str,
            summary=f"TTFB: {ttfb_str}",
            full_content=ttfb_str,
        )
        metrics_header.add(
            f"TTFB: {ttfb_str}",
            data=ttfb_data,
            allow_expand=False,
        )

    # Total RTT (response_end - request_start) in s
    if timing.request_start is not None and timing.response_end is not None:
        rtt = timing.response_end - timing.request_start
        rtt_str = _format_duration(rtt)
        rtt_data = CallTreeNodeData(
            node_type="field",
            field_key="Total RTT",
            field_value=rtt_str,
            summary=f"Total RTT: {rtt_str}",
            full_content=rtt_str,
        )
        metrics_header.add(
            f"Total RTT: {rtt_str}",
            data=rtt_data,
            allow_expand=False,
        )

    # -- Connection-level timing (per-call) --
    conn_timing = call.connection_timing
    if conn_timing is not None and conn_timing.timestamp_start is not None:
        conn_str = _format_connection_timing(
            conn_timing.timestamp_start,
            conn_timing.timestamp_tls_setup,
            conn_timing.timestamp_end,
            shared_flow_count=0,  # Per-call: no sharing info at this level
        )
        if conn_str:
            conn_header_data = CallTreeNodeData(
                node_type="field",
                field_key="Connection",
                summary="Connection",
            )
            conn_header = section_node.add(
                "Connection",
                data=conn_header_data,
                allow_expand=True,
            )
            conn_data = CallTreeNodeData(
                node_type="field",
                field_key="Connection Timing",
                field_value=conn_str,
                summary=conn_str,
                full_content=conn_str,
            )
            conn_header.add(conn_str, data=conn_data, allow_expand=False)


# ---------------------------------------------------------------------------
# Token Usage section
# ---------------------------------------------------------------------------


def _compute_cache_efficiency(
    cache_read: Optional[int],
    input_tokens: Optional[int],
) -> Optional[float]:
    """Compute cache efficiency as a percentage.

    Cache efficiency is calculated as::

        cache_read / (cache_read + input_tokens) × 100

    If both values are 0 (or None), returns ``None``.
    If ``cache_read > 0`` but ``input_tokens`` is 0 or None, returns 100.0.

    Args:
        cache_read: Number of tokens read from cache.
        input_tokens: Number of fresh input tokens.

    Returns:
        The efficiency percentage (0.0–100.0), or ``None`` if not computable.
    """
    if cache_read is None:
        return None
    cache_read = max(cache_read, 0)
    if input_tokens is None:
        input_tokens = 0
    input_tokens = max(input_tokens, 0)

    total = cache_read + input_tokens
    if total == 0:
        return None
    return (cache_read / total) * 100.0


def _render_cache_efficiency(efficiency_pct: Optional[float]) -> Optional[Text]:
    """Render a cache efficiency value as a styled :class:`Text` object.

    Applies color coding:
    * **>90%** — bold green
    * **<50%** — bold yellow
    * 50–90% — default (no colour)

    Args:
        efficiency_pct: The efficiency percentage (or ``None``).

    Returns:
        A styled :class:`Text` object, or ``None`` if the input is ``None``.
    """
    if efficiency_pct is None:
        return None

    pct_str = f"{efficiency_pct:.1f}%"
    if efficiency_pct > 90.0:
        return Text(pct_str, style="bold green")
    elif efficiency_pct < 50.0:
        return Text(pct_str, style="bold yellow")
    else:
        return Text(pct_str)


def _add_token_usage_section(section_node, call: LLMCall) -> None:
    """Populate a Token Usage section node with children.

    Displays detailed token counts and cache efficiency metrics.

    **Fields shown:**
    * **Input Tokens** — fresh input tokens (``response.input_tokens``)
    * **Output Tokens** — generated output tokens (``response.output_tokens``)
    * **Cache Created** — tokens written to cache (``response.cache_creation_input_tokens``)
    * **Cache Read** — tokens read from cache (``response.cache_read_input_tokens``)
    * **Cache Efficiency** — hit rate as percentage with colour coding
    * **Service Tier** — service tier identifier (``response.service_tier``)

    All token counts ≥ 1000 are displayed with thousands separators
    (e.g. ``"12,800"``). Cache efficiency is colour-coded green for >90%
    and yellow for <50%.

    Args:
        section_node: The ``TreeNode`` for the Token Usage section.
        call: The :class:`LLMCall` containing token usage data.
    """
    resp = call.response
    tu = call.token_usage

    # Use response-level token data as primary source
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_read: Optional[int] = None
    cache_creation: Optional[int] = None

    if resp is not None:
        input_tokens = resp.input_tokens
        output_tokens = resp.output_tokens
        cache_read = resp.cache_read_input_tokens
        cache_creation = resp.cache_creation_input_tokens
    elif tu is not None:
        input_tokens = tu.prompt_tokens
        output_tokens = tu.completion_tokens

    # -- Input Tokens --
    if input_tokens is not None:
        _add_field_node(
            section_node,
            "Input Tokens",
            _format_thousands(input_tokens),
        )
    else:
        # Missing optional field — show N/A placeholder
        _add_field_node(section_node, "Input Tokens", "\u2014")

    # -- Output Tokens --
    if output_tokens is not None:
        _add_field_node(
            section_node,
            "Output Tokens",
            _format_thousands(output_tokens),
        )
    else:
        _add_field_node(section_node, "Output Tokens", "\u2014")

    # -- Cache Created --
    if cache_creation is not None:
        _add_field_node(
            section_node,
            "Cache Created",
            _format_thousands(cache_creation),
        )
    else:
        _add_field_node(section_node, "Cache Created", "\u2014")

    # -- Cache Read + Efficiency --
    if cache_read is not None:
        _add_field_node(
            section_node,
            "Cache Read",
            _format_thousands(cache_read),
        )

        # Cache efficiency (colour-coded)
        efficiency = _compute_cache_efficiency(cache_read, input_tokens)
        styled_eff = _render_cache_efficiency(efficiency)
        if styled_eff is not None:
            label_str = str(styled_eff)
            eff_data = CallTreeNodeData(
                node_type="field",
                field_key="Cache Efficiency",
                field_value=label_str,
                summary=f"Cache Efficiency: {label_str}",
                full_content=label_str,
            )
            # Use the styled Text object as the label
            section_node.add(
                Text.assemble("Cache Efficiency: ", styled_eff),
                data=eff_data,
                allow_expand=False,
            )
    else:
        _add_field_node(section_node, "Cache Read", "\u2014")

    # -- Service Tier --
    if resp is not None and resp.service_tier:
        _add_field_node(
            section_node,
            "Service Tier",
            resp.service_tier,
        )


def _compute_session_token_info(session: Session) -> str:
    """Compute and format aggregate token totals for a session.

    Sums input_tokens, output_tokens, and total cache tokens (cache_read +
    cache_creation) across all calls in the session.  Token counts ≥ 1000
    are formatted with thousands separators.

    Args:
        session: The session whose calls to aggregate.

    Returns:
        A formatted string like ``"Input: 12,800 | Output: 1,050 | Cache: 248,320"``
        or an empty string if no token data is available across all calls.
    """
    total_input = 0
    total_output = 0
    total_cache = 0
    found_any = False

    for call in session.calls:
        resp = call.response
        if resp is not None:
            if resp.input_tokens is not None:
                total_input += resp.input_tokens
                found_any = True
            if resp.output_tokens is not None:
                total_output += resp.output_tokens
                found_any = True
            if resp.cache_read_input_tokens is not None:
                total_cache += resp.cache_read_input_tokens
                found_any = True
            if resp.cache_creation_input_tokens is not None:
                total_cache += resp.cache_creation_input_tokens
                found_any = True
        elif call.token_usage is not None:
            if call.token_usage.prompt_tokens is not None:
                total_input += call.token_usage.prompt_tokens
                found_any = True
            if call.token_usage.completion_tokens is not None:
                total_output += call.token_usage.completion_tokens
                found_any = True

    if not found_any:
        return ""

    parts = []
    parts.append(f"Input: {_format_thousands(total_input)}")
    parts.append(f"Output: {_format_thousands(total_output)}")
    parts.append(f"Cache: {_format_thousands(total_cache)}")
    return " | ".join(parts)


def _compute_session_connection_info(
    session: Session,
) -> str:
    """Compute and format session-level connection timing information.

    Looks at all calls in the session to determine:
    * Connection lifetime (earliest start to latest end across shared
      connections)
    * TLS setup time (from the first call with TLS data)
    * Number of flows sharing the most common connection

    Args:
        session: The session containing calls with connection timing data.

    Returns:
        A formatted string with connection info, or empty string if no
        connection data is available.
    """
    conn_ids: Dict[str, int] = {}
    first_conn_start: Optional[float] = None
    first_conn_tls: Optional[float] = None
    last_conn_end: Optional[float] = None

    for call in session.calls:
        ct = call.connection_timing
        if ct is None or not ct.conn_id:
            continue

        # Count connection reuse
        conn_ids[ct.conn_id] = conn_ids.get(ct.conn_id, 0) + 1

        # Track overall connection timing
        if ct.timestamp_start is not None:
            if first_conn_start is None or ct.timestamp_start < first_conn_start:
                first_conn_start = ct.timestamp_start
            if ct.timestamp_tls_setup is not None:
                if first_conn_tls is None:
                    first_conn_tls = ct.timestamp_tls_setup
        if ct.timestamp_end is not None:
            if last_conn_end is None or ct.timestamp_end > last_conn_end:
                last_conn_end = ct.timestamp_end

    if not conn_ids:
        return ""

    # Find the most shared connection
    most_shared_conn_id = max(conn_ids, key=conn_ids.get)
    shared_count = conn_ids[most_shared_conn_id]

    return _format_connection_timing(
        first_conn_start,
        first_conn_tls,
        last_conn_end,
        shared_count,
    )


# ---------------------------------------------------------------------------
# CallTree widget
# ---------------------------------------------------------------------------


class CallTree(Tree[CallTreeNodeData]):
    """A collapsible tree widget that displays the session's LLM API calls.

    **States:**

    * **Placeholder** — shown when no session is loaded.  The tree root
      displays a message like ``"Select a session to begin"``.
    * **Loading** — shown while a session is being parsed.  The root label
      changes to ``"Loading <session_name>..."``.
    * **Populated** — shown after a session has been successfully loaded.
      The tree root shows the session name, with child nodes for each
      LLM API call.  Each call node is expandable to reveal detail
      sections (Request Details, Response Details, Tool Calls & Results,
      Timing, Token Usage) — sections are conditionally included based on
      data availability.
    """

    BINDINGS = [
        Binding("enter", "select_cursor", "Select", show=True),
        Binding("space", "toggle_node", "Expand", show=True),
        Binding("shift+space", "toggle_all_nodes", "Expand/Collapse All", show=False),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("tab", "focus_next", "Next Panel", show=False),
    ]

    DEFAULT_CSS = """
    CallTree {
        overflow-y: auto;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(
            "Select a session from the sidebar to view its API calls",
            data=CallTreeNodeData(
                node_type="placeholder",
                summary="Select a session from the sidebar to view its API calls",
            ),
            **kwargs,
        )
        self.show_root = True
        self.guide_depth = 5

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_placeholder(self, text: str = "Select a session from the sidebar to view its API calls") -> None:
        """Reset the tree to the placeholder state.

        Clears all children and sets the root label to a placeholder message.

        Args:
            text: The placeholder text to display.
        """
        self.clear()
        self.root.set_label(text)
        self.root.data = CallTreeNodeData(node_type="placeholder", summary=text)
        self.root.expand()

    def show_loading(self, session_name: str = "") -> None:
        """Show a loading indicator in the tree.

        Clears any existing children and updates the root label to indicate
        that a session is being loaded.

        Args:
            session_name: The name of the session being loaded (shown in
                the label for context).
        """
        self.clear()
        label = f"Loading {session_name}..." if session_name else "Loading..."
        self.root.set_label(label)
        self.root.data = CallTreeNodeData(
            node_type="loading",
            summary=f"Loading {session_name}..." if session_name else "Loading...",
        )

    def populate(self, session: Session) -> None:
        """Populate the tree with session and call nodes.

        Clears any existing state and builds a tree with a session root node,
        call child nodes (sorted chronologically), and expandable section
        sub-nodes per call.

        Args:
            session: The parsed session data containing zero or more
                :class:`~llm_flow_viewer.parser.models.LLMCall` objects.
        """
        self.clear()

        if not session.calls:
            # Empty session — show informational message
            self.root.set_label(f"{session.task_name} — No API calls found")
            self.root.data = CallTreeNodeData(
                node_type="session",
                summary=f"{session.task_name} — No API calls found",
            )
            return

        # Build the session root node
        call_count = len(session.calls)
        session_label = f"Session: {session.task_name} [{call_count} call{'s' if call_count != 1 else ''}]"
        self.root.set_label(session_label)
        self.root.data = CallTreeNodeData(
            node_type="session",
            summary=session_label,
        )

        # Add call nodes in chronological order (already sorted by session)
        for idx, call in enumerate(session.calls):
            label = _call_node_label(call, idx)
            call_data = CallTreeNodeData(
                node_type="call",
                call=call,
                call_index=idx,
                summary=label,
            )
            call_node = self.root.add(label, data=call_data, allow_expand=True)

            # Add section nodes (conditionally)
            _add_sections(call_node, call, session, idx)

        # Add error nodes for corrupt flow data
        for err_idx, err_msg in enumerate(session.flow_errors):
            error_label = f"{_WARNING_ICON} Error reading flow: {err_msg[:80]}{'...' if len(err_msg) > 80 else ''}"
            error_data = CallTreeNodeData(
                node_type="error",
                summary=error_label,
                full_content=err_msg,
            )
            self.root.add(
                Text(error_label, style="bold yellow"),
                data=error_data,
                allow_expand=False,
            )

        # Add session-level connection timing node (if connection data available)
        conn_info = _compute_session_connection_info(session)
        if conn_info:
            conn_data = CallTreeNodeData(
                node_type="field",
                field_key="Connection",
                field_value=conn_info,
                summary=f"Connection: {conn_info}",
                full_content=conn_info,
            )
            self.root.add(
                f"\U0001f310 Connection: {conn_info}",
                data=conn_data,
                allow_expand=False,
            )

        # Add session-level aggregate token totals (if token data available)
        token_info = _compute_session_token_info(session)
        if token_info:
            token_data = CallTreeNodeData(
                node_type="field",
                field_key="Session Tokens",
                field_value=token_info,
                summary=f"\U0001f4ca {token_info}",
                full_content=token_info,
            )
            self.root.add(
                f"\U0001f4ca {token_info}",
                data=token_data,
                allow_expand=False,
            )

        # Expand the root to show call nodes
        self.root.expand()

    def append_calls(
        self,
        calls: List[LLMCall],
        session: Session | None = None,
    ) -> None:
        """Append call nodes for incremental (streaming) population.

        Adds each :class:`LLMCall` in *calls* as a new child of the root
        node, building its full expandable section structure (Request
        Details, Response Details, Tool Calls, Timing, Token Usage).

        When *session* is ``None`` (the common case during streaming),
        cross-call tool result lookups are skipped — each tool use will
        show ``"→ (no result recorded)"``.  The caller may pass a
        partial or complete session to enable result linking.

        This method is designed for use in the streaming load path:
        batch-fetched calls are appended incrementally rather than
        waiting for the full parse to complete.

        Args:
            calls: :class:`LLMCall` objects to add as tree nodes.
            session: Optional :class:`Session` for cross-call tool
                result lookups.  ``None`` during streaming.
        """
        # Determine starting index from existing call children
        existing_call_count = sum(
            1 for c in self.root.children
            if c.data is not None and c.data.node_type == "call"
        )

        for offset, call in enumerate(calls):
            call_idx = existing_call_count + offset
            label = _call_node_label(call, call_idx)
            call_data = CallTreeNodeData(
                node_type="call",
                call=call,
                call_index=call_idx,
                summary=label,
            )
            call_node = self.root.add(
                label, data=call_data, allow_expand=True,
            )

            # Add expandable section nodes (conditionally)
            _add_sections(call_node, call, session, call_idx)

        self.root.expand()

    def add_error_node(self, task_name: str, error_message: str) -> None:
        """Append an error indicator node to the existing tree root.

        Unlike :meth:`show_placeholder`, this method does **not** clear
        existing nodes — it adds the error as a new child of the root,
        preserving any partially-loaded call nodes already present.

        The root label is updated to show the task name with an error
        indicator, and a child node with the error details is added.

        Args:
            task_name: The name of the session that failed.
            error_message: The error message to display.
        """
        # Update root label to show the session name + error indicator
        current_label = str(self.root.label)
        if "Error" not in current_label:
            self.root.set_label(f"{task_name} [Error]")

        # Update root data to session type if it was placeholder/loading
        root_data = self.root.data
        if root_data and root_data.node_type in ("placeholder", "loading"):
            self.root.data = CallTreeNodeData(
                node_type="session",
                summary=f"{task_name} [Error]",
            )

        # Add error child node
        error_label = (
            f"\u274c Parse error: {error_message[:80]}"
            f"{'...' if len(error_message) > 80 else ''}"
        )
        error_data = CallTreeNodeData(
            node_type="error",
            summary=error_label,
            full_content=f"Error loading {task_name}: {error_message}",
        )
        self.root.add(
            error_label,
            data=error_data,
            allow_expand=False,
        )

        self.root.expand()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_toggle_all_nodes(self) -> None:
        """Toggle expand/collapse all children of the cursor node recursively.

        If the current cursor node's immediate children are all collapsed,
        expand all descendants recursively. Otherwise, collapse all
        descendants recursively.
        """
        cursor_node = self.cursor_node
        if cursor_node is None or not cursor_node.children:
            return

        # Determine current state: if any child is expanded, we collapse all;
        # if all are collapsed, we expand all.
        any_expanded = any(
            child.allow_expand and child.is_expanded
            for child in cursor_node.children
        )

        if any_expanded:
            self._collapse_all_recursive(cursor_node)
        else:
            self._expand_all_recursive(cursor_node)

    @staticmethod
    def _expand_all_recursive(node) -> None:
        """Recursively expand a node and all its expandable descendants.

        Args:
            node: The ``TreeNode`` to expand (and whose children will
                also be expanded).
        """
        for child in node.children:
            if child.allow_expand:
                child.expand()
                CallTree._expand_all_recursive(child)

    @staticmethod
    def _collapse_all_recursive(node) -> None:
        """Recursively collapse all expandable descendants of a node.

        The node itself is not collapsed; only its children and their
        descendants are collapsed. This preserves the cursor position
        while hiding nested detail.

        Args:
            node: The ``TreeNode`` whose descendants should be collapsed.
        """
        for child in node.children:
            if child.allow_expand:
                CallTree._collapse_all_recursive(child)
                child.collapse()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        """Set up the tree after the widget is mounted."""
        self.root.expand()


# ---------------------------------------------------------------------------
# HTTP status / Stop reason helpers
# ---------------------------------------------------------------------------


_HTTP_STATUS_LABELS: dict[int, str] = {
    100: "Continue",
    101: "Switching Protocols",
    200: "OK",
    201: "Created",
    202: "Accepted",
    204: "No Content",
    301: "Moved Permanently",
    302: "Found",
    304: "Not Modified",
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    408: "Request Timeout",
    409: "Conflict",
    415: "Unsupported Media Type",
    422: "Unprocessable Entity",
    429: "Too Many Requests",
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout",
}

_THINKING_ICON = "\U0001f9e0"  # 🧠

_SSE_EVENT_TYPES: set[str] = {
    "message_start", "content_block_start", "content_block_delta",
    "content_block_stop", "message_delta", "message_stop", "ping",
}


def _http_status_label(status_code: int) -> str:
    """Return a human-readable HTTP status label.

    Args:
        status_code: The numeric HTTP status code.

    Returns:
        A string like ``"200 OK"`` or ``"429 Too Many Requests"``.
    """
    label = _HTTP_STATUS_LABELS.get(status_code, "Unknown")
    return f"{status_code} {label}"


def _human_readable_stop_reason(stop_reason: Optional[str]) -> str:
    """Convert a raw stop_reason to a human-readable form.

    Args:
        stop_reason: The raw stop reason from the SSE stream (or ``None``).

    Returns:
        Human-readable string:

        * ``"end_turn"`` → ``"Turn completed"``
        * ``"tool_use"`` → ``"Requesting tool execution"``
        * ``None`` → ``"N/A"``
        * Anything else → as-is.
    """
    mapping = {
        "end_turn": "Turn completed",
        "tool_use": "Requesting tool execution",
    }
    if stop_reason is None:
        return "N/A"
    return mapping.get(stop_reason, stop_reason)


# ---------------------------------------------------------------------------
# Section population helpers
# ---------------------------------------------------------------------------


def _add_field_node(
    parent_node,
    key: str,
    value: str,
    summary: str = "",
) -> None:
    """Add a simple key-value field child node.

    Args:
        parent_node: The parent ``TreeNode``.
        key: The field key (e.g. ``"Model"``).
        value: The field value as a string.
        summary: Alternate display label (if empty, ``"{key}: {value}"``).
    """
    label = summary or f"{key}: {value}"
    data = CallTreeNodeData(
        node_type="field",
        field_key=key,
        field_value=value,
        summary=label,
        full_content=value,
    )
    parent_node.add(label, data=data, allow_expand=False)


def _add_messages_section(parent_node, messages: List[Dict[str, Any]]) -> None:
    """Add a Messages sub-section with role icons and content previews.

    Creates an expandable "Messages (N)" node, then adds each message
    as an expandable child with role icon, content preview, and content
    block children.

    Args:
        parent_node: The parent ``TreeNode`` (the Request Details section).
        messages: The list of message dicts from the request.
    """
    count = len(messages)
    header_data = CallTreeNodeData(
        node_type="messages_header",
        summary=f"Messages ({count})",
    )
    header_node = parent_node.add(
        f"Messages ({count})",
        data=header_data,
        allow_expand=True,
    )

    for msg_idx, message in enumerate(messages):
        role = message.get("role", "unknown")
        icon = _get_role_icon(role)
        preview = _get_message_content_preview(message)
        # Store the full message as JSON for detail panel rendering
        message_full_content = json.dumps(message, indent=2)
        message_data = CallTreeNodeData(
            node_type="message",
            message_role=role,
            content_preview=preview,
            summary=f"{icon} {role}: {preview}",
            full_content=message_full_content,
            message_index=msg_idx,
        )
        # Messages with content blocks are expandable
        has_content_blocks = _should_show_content_blocks(message)
        msg_node = header_node.add(
            f"{icon} {role}: {preview}",
            data=message_data,
            allow_expand=has_content_blocks,
        )

        # Add content block children if message has blocks
        if has_content_blocks:
            blocks = message.get("content", [])
            for blk_idx, block in enumerate(blocks):
                blk_type = block.get("type", "unknown")
                type_label = _get_content_type_label(blk_type)
                blk_preview = _get_content_block_preview(block)
                blk_data = CallTreeNodeData(
                    node_type="content_block",
                    content_block_type=blk_type,
                    content_preview=blk_preview,
                    summary=f"{type_label} {blk_preview}",
                    message_index=msg_idx,
                    block_index=blk_idx,
                    full_content=_get_full_block_content(block),
                )
                # content blocks are leaf nodes (not expandable in tree,
                # but they can be selected to show in detail panel)
                msg_node.add(
                    f"{type_label} {blk_preview}",
                    data=blk_data,
                    allow_expand=False,
                )


def _get_full_block_content(block: Dict[str, Any]) -> str:
    """Get the full content of a content block as a display string.

    Args:
        block: The content block dict.

    Returns:
        The full content string (JSON for complex types).
    """
    block_type = block.get("type", "unknown")
    if block_type == "text":
        return block.get("text", "")
    if block_type == "thinking":
        return block.get("thinking", "")
    if block_type == "tool_use":
        # Return structured JSON for tool_use blocks
        return json.dumps({
            "type": "tool_use",
            "name": block.get("name", ""),
            "id": block.get("id", ""),
            "input": block.get("input", {}),
        }, indent=2)
    if block_type == "tool_result":
        return json.dumps(block, indent=2)
    if block_type == "image_url":
        return json.dumps(block, indent=2)
    return json.dumps(block, indent=2)


def _add_tools_section(parent_node, tools: List[Dict[str, Any]]) -> None:
    """Add a Tools sub-section with tool definitions.

    Creates an expandable "Tools (N)" node, then adds each tool with
    its name, truncated description, and expandable input_schema JSON.

    Args:
        parent_node: The parent ``TreeNode``.
        tools: The list of tool definition dicts.
    """
    count = len(tools)
    header_data = CallTreeNodeData(
        node_type="tools_header",
        summary=f"Tools ({count})",
    )
    header_node = parent_node.add(
        f"Tools ({count})",
        data=header_data,
        allow_expand=True,
    )

    for tool in tools:
        name = tool.get("name", "?")
        description = tool.get("description", "")
        truncated_desc = _truncate(description, max_len=60)
        tool_data = CallTreeNodeData(
            node_type="tool",
            tool_name=name,
            tool_description=description,
            summary=f"{name}: {truncated_desc}",
            full_content=description,
        )
        tool_node = header_node.add(
            f"{name}: {truncated_desc}",
            data=tool_data,
            allow_expand=True,
        )

        # Add input_schema as a child node (expandable to show JSON)
        input_schema = tool.get("input_schema", {})
        schema_json = json.dumps(input_schema, indent=2) if input_schema else "{}"
        schema_data = CallTreeNodeData(
            node_type="tool_input_schema",
            full_content=schema_json,
            summary="input_schema (JSON)",
        )
        tool_node.add(
            "input_schema (JSON)",
            data=schema_data,
            allow_expand=False,
        )


def _add_system_section(parent_node, system: List[Dict[str, Any]]) -> None:
    """Add a System sub-section with system prompt messages.

    Creates an expandable "System (N)" node with each system message
    showing a preview; expanding shows the full text.

    Args:
        parent_node: The parent ``TreeNode``.
        system: The list of system prompt dicts (each has ``type``, ``text``).
    """
    count = len(system)
    header_data = CallTreeNodeData(
        node_type="system_header",
        summary=f"System ({count})",
    )
    header_node = parent_node.add(
        f"System ({count})",
        data=header_data,
        allow_expand=True,
    )

    for sys_idx, sys_msg in enumerate(system):
        text = sys_msg.get("text", "") if isinstance(sys_msg, dict) else str(sys_msg)
        preview = _truncate(text, max_len=60)
        sys_msg_data = CallTreeNodeData(
            node_type="system_message",
            summary=f"System Prompt #{sys_idx + 1}: {preview}",
            content_preview=preview,
            full_content=text,
            message_index=sys_idx,
        )
        sys_node = header_node.add(
            f"System Prompt #{sys_idx + 1}: {preview}",
            data=sys_msg_data,
            allow_expand=True,
        )

        # Full text child node (leaf, shows in detail panel)
        full_data = CallTreeNodeData(
            node_type="system_text",
            summary="Full text",
            full_content=text,
        )
        sys_node.add(
            "Full text",
            data=full_data,
            allow_expand=False,
        )


def _add_raw_request_node(parent_node, raw_json: str) -> None:
    """Add a Raw Request node that shows the full request as JSON.

    Args:
        parent_node: The parent ``TreeNode``.
        raw_json: The pretty-printed JSON string of the full request body.
    """
    data = CallTreeNodeData(
        node_type="raw_request",
        summary="Raw Request (JSON)",
        full_content=raw_json,
    )
    parent_node.add(
        "Raw Request (JSON)",
        data=data,
        allow_expand=False,
    )


# ---------------------------------------------------------------------------
# Section population
# ---------------------------------------------------------------------------


def _add_request_details(call_node, call: LLMCall) -> None:
    """Populate a Request Details section node with children.

    Adds field nodes for top-level metadata (model, max_tokens, stream,
    output_config), then expandable sub-sections for Messages, Tools,
    System prompts, and a Raw Request leaf node.

    Args:
        call_node: The ``TreeNode`` for the Request Details section.
        call: The :class:`LLMCall` containing request data.
    """
    req = call.request
    if req is None:
        return

    # -- Top-level fields --
    # Model
    if req.model:
        _add_field_node(call_node, "Model", req.model)

    # Max tokens
    if req.max_tokens > 0:
        _add_field_node(call_node, "Max Tokens", _format_thousands(req.max_tokens))

    # Stream flag
    _add_field_node(call_node, "Stream", "True" if req.stream else "False")

    # Output config
    if req.output_config:
        oc_json = json.dumps(req.output_config, indent=2)
        oc_data = CallTreeNodeData(
            node_type="output_config",
            summary="Output Config",
            full_content=oc_json,
        )
        call_node.add("Output Config", data=oc_data, allow_expand=False)

    # -- Messages --
    if req.messages:
        _add_messages_section(call_node, req.messages)

    # -- Tools --
    if req.tools:
        _add_tools_section(call_node, req.tools)

    # -- System prompts --
    if req.system:
        _add_system_section(call_node, req.system)

    # -- Raw Request JSON --
    # Build the raw JSON from the request object fields
    raw_body = _build_raw_request_json(req)
    _add_raw_request_node(call_node, raw_body)


def _build_raw_request_json(req) -> str:
    """Build a pretty-printed JSON string representing the full request body.

    Args:
        req: The :class:`ParsedRequest` object.

    Returns:
        A syntax-highlightable JSON string.
    """
    body = {
        "model": req.model,
        "max_tokens": req.max_tokens,
        "stream": req.stream,
    }
    if req.messages:
        body["messages"] = req.messages
    if req.tools:
        body["tools"] = req.tools
    if req.system:
        body["system"] = req.system
    if req.output_config:
        body["output_config"] = req.output_config
    if req.thinking:
        body["thinking"] = req.thinking
    return json.dumps(body, indent=2)


def _add_response_details(call_node, call: LLMCall) -> None:
    """Populate a Response Details section node with children.

    Adds field nodes for HTTP status code (with human-readable label,
    red styling for non-200), model name, message_id, stop reason
    (human-readable), then expandable sub-nodes for text output,
    thinking/reasoning content (with thinking icon), and raw SSE events.

    Args:
        call_node: The ``TreeNode`` for the Response Details section.
        call: The :class:`LLMCall` containing response data.
    """
    resp = call.response
    if resp is None:
        return

    # -- HTTP status code --
    status_label = _http_status_label(resp.status_code)
    if resp.status_code != 200:
        # Use a styled Text object for red/warning highlighting
        status_value = Text(status_label, style="bold red")
        status_data = CallTreeNodeData(
            node_type="field",
            field_key="Status",
            field_value=status_label,
            summary=f"Status: {status_label}",
            full_content=status_label,
        )
        call_node.add(
            Text.assemble("Status: ", status_value),
            data=status_data,
            allow_expand=False,
        )
    else:
        _add_field_node(call_node, "Status", status_label)

    # -- SSE parse warnings --
    if resp.sse_parse_warnings > 0:
        warning_text = f"{_WARNING_ICON} {resp.sse_parse_warnings} SSE data line(s) had invalid JSON"
        warning_data = CallTreeNodeData(
            node_type="field",
            field_key="SSE Parse Warnings",
            field_value=str(resp.sse_parse_warnings),
            summary=warning_text,
            full_content=(
                f"{resp.sse_parse_warnings} SSE data block(s) contained invalid JSON "
                f"and were skipped. Valid data blocks were retained."
            ),
        )
        call_node.add(
            Text(warning_text, style="bold yellow"),
            data=warning_data,
            allow_expand=False,
        )

    # -- Model --
    if resp.model:
        _add_field_node(call_node, "Model", resp.model)

    # -- Message ID --
    if resp.message_id:
        _add_field_node(call_node, "Message ID", resp.message_id)

    # -- Stop reason --
    readable_reason = _human_readable_stop_reason(resp.stop_reason)
    _add_field_node(call_node, "Stop Reason", readable_reason)

    # -- Text Output --
    text = resp.text or ""
    text_char_count = len(text)
    text_label = f"Text Output ({_format_thousands(text_char_count)} char{'' if text_char_count == 1 else 's'})"
    text_data = CallTreeNodeData(
        node_type="response_text",
        summary=text_label,
        full_content=text,
        content_preview=text[:_MAX_CONTENT_PREVIEW_LEN] if text else "(empty)",
    )
    call_node.add(text_label, data=text_data, allow_expand=False)

    # -- Thinking / Reasoning --
    thinking = resp.thinking or ""
    if thinking:
        thinking_char_count = len(thinking)
        thinking_label = (
            f"{_THINKING_ICON} Thinking "
            f"({_format_thousands(thinking_char_count)} char{'' if thinking_char_count == 1 else 's'})"
        )
        thinking_data = CallTreeNodeData(
            node_type="response_thinking",
            summary=thinking_label,
            full_content=thinking,
            content_preview=thinking[:_MAX_CONTENT_PREVIEW_LEN],
        )
        call_node.add(thinking_label, data=thinking_data, allow_expand=False)

    # -- Raw SSE Events --
    raw_events = resp.raw_sse_events
    if raw_events:
        event_count = len(raw_events)
        sse_label = f"Raw SSE Events ({event_count})"
        sse_header_data = CallTreeNodeData(
            node_type="response_raw_sse_header",
            summary=sse_label,
        )
        sse_header_node = call_node.add(
            sse_label,
            data=sse_header_data,
            allow_expand=True,
        )

        for ev_idx, event in enumerate(raw_events):
            event_type = event.get("event_type", "unknown")
            event_data_dict = event.get("data", {})
            event_json = json.dumps(event_data_dict, indent=2)
            event_label = f"#{ev_idx + 1} {event_type}"
            event_node_data = CallTreeNodeData(
                node_type="response_sse_event",
                summary=event_label,
                full_content=event_json,
                content_preview=event_type,
                block_index=ev_idx,
            )
            sse_header_node.add(
                event_label,
                data=event_node_data,
                allow_expand=False,
            )


def _add_sections(
    call_node,
    call: LLMCall,
    session: Session | None,
    call_idx: int,
) -> None:
    """Add expandable section nodes to a call node.

    Sections are added in a fixed order and conditionally included based
    on data availability in the call. Sections are populated eagerly.

    When *session* is ``None`` (during incremental streaming via
    :meth:`CallTree.append_calls`), cross-call tool result lookups are
    skipped.

    Args:
        call_node: The ``TreeNode`` for the call.
        call: The :class:`LLMCall` whose data determines which sections
            to include.
        session: The :class:`Session` containing all calls (used to
            look up tool results in subsequent calls), or ``None``
            during streaming.
        call_idx: The index of *call* in ``session.calls``.
    """
    section_order = [
        "request_details",
        "response_details",
        "tool_calls",
        "timing",
        "token_usage",
    ]

    for section_type in section_order:
        if _should_include_section(section_type, call):
            label = _SECTION_LABELS[section_type]
            section_data = CallTreeNodeData(
                node_type="section",
                section_type=section_type,
                call=call,
                summary=label,
            )
            section_node = call_node.add(label, data=section_data, allow_expand=True)

            # Populate section eagerly
            if section_type == "request_details":
                _add_request_details(section_node, call)
            elif section_type == "response_details":
                _add_response_details(section_node, call)
            elif section_type == "tool_calls":
                _add_tool_calls(section_node, call, session, call_idx)
            elif section_type == "timing":
                _add_timing_section(section_node, call)
            elif section_type == "token_usage":
                _add_token_usage_section(section_node, call)
