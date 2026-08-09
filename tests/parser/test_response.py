"""Tests for the SSE response stream parser module."""

import json
import logging

import pytest

from llm_flow_viewer.parser.models import ParsedResponse, ToolUse
from llm_flow_viewer.parser.response import parse_response


# ---------------------------------------------------------------------------
# Helper — build SSE event blocks
# ---------------------------------------------------------------------------


def _sse_event(event_type: str, data: dict) -> str:
    """Format a single SSE event block (event + data lines)."""
    return f"event: {event_type}\ndata: {json.dumps(data, separators=(',', ':'))}\n"


def _sse_body(*events: tuple[str, dict]) -> str:
    """Build a full SSE body from (event_type, data_dict) pairs."""
    return "\n".join(_sse_event(t, d) for t, d in events)


# ===================================================================
# VAL-PARSE-005: Empty response content
# ===================================================================

class TestEmptyContent:
    """Flows with empty response bodies should not crash the parser."""

    def test_empty_bytes_returns_default_response(self):
        """Empty bytes content produces a default ParsedResponse."""
        result = parse_response(b"")
        assert isinstance(result, ParsedResponse)
        assert result.text == ""
        assert result.thinking == ""
        assert result.tool_uses == []
        assert result.stop_reason is None

    def test_none_content_returns_default_response(self):
        """None content produces a default ParsedResponse."""
        result = parse_response(None)
        assert isinstance(result, ParsedResponse)
        assert result.text == ""

    def test_blank_string_returns_default_response(self):
        """Whitespace-only content produces a default ParsedResponse."""
        result = parse_response(b"   \n\n  ")
        assert isinstance(result, ParsedResponse)
        assert result.text == ""


# ===================================================================
# VAL-PARSE-015: Text content concatenation
# ===================================================================

class TestTextContent:
    """text_delta events are concatenated into parsed_response.text."""

    SSE_TEXT_ONLY = _sse_body(
        ("message_start", {
            "type": "message_start",
            "message": {
                "id": "msg_001",
                "type": "message",
                "role": "assistant",
                "model": "deepseek-v4-flash",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 100,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 500,
                    "output_tokens": 0,
                    "service_tier": "standard",
                },
            },
        }),
        ("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello, "},
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "world!"},
        }),
        ("content_block_stop", {
            "type": "content_block_stop",
            "index": 0,
        }),
        ("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {
                "input_tokens": 100,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 500,
                "output_tokens": 5,
                "service_tier": "standard",
            },
        }),
        ("message_stop", {"type": "message_stop"}),
    )

    def test_concatenates_text_deltas(self):
        """Multiple text_delta events are concatenated in order."""
        result = parse_response(self.SSE_TEXT_ONLY.encode("utf-8"))
        assert result.text == "Hello, world!"
        assert result.thinking == ""

    def test_stop_reason_is_end_turn(self):
        """A text-only response has stop_reason=end_turn."""
        result = parse_response(self.SSE_TEXT_ONLY.encode("utf-8"))
        assert result.stop_reason == "end_turn"

    def test_initial_text_in_content_block_start(self):
        """text in content_block_start is included in final text."""
        sse = _sse_body(
            ("content_block_start", {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": "Initial "},
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "delta"},
            }),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 2},
            }),
        ).encode("utf-8")
        result = parse_response(sse)
        assert result.text == "Initial delta"


# ===================================================================
# VAL-PARSE-016: Thinking content concatenation
# ===================================================================

class TestThinkingContent:
    """thinking_delta events are concatenated into parsed_response.thinking."""

    SSE_WITH_THINKING = _sse_body(
        ("message_start", {
            "type": "message_start",
            "message": {
                "id": "msg_002",
                "type": "message",
                "role": "assistant",
                "model": "deepseek-v4-flash",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 50, "output_tokens": 0},
            },
        }),
        ("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Let me "},
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "think about "},
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "this."},
        }),
        ("content_block_stop", {
            "type": "content_block_stop",
            "index": 0,
        }),
        ("content_block_start", {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "text", "text": ""},
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "Here is my answer."},
        }),
        ("content_block_stop", {
            "type": "content_block_stop",
            "index": 1,
        }),
        ("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 12},
        }),
        ("message_stop", {"type": "message_stop"}),
    )

    def test_concatenates_thinking_deltas(self):
        """thinking_delta events are concatenated into the thinking field."""
        result = parse_response(self.SSE_WITH_THINKING.encode("utf-8"))
        assert result.thinking == "Let me think about this."

    def test_text_is_separate_from_thinking(self):
        """Text output is separate from thinking content."""
        result = parse_response(self.SSE_WITH_THINKING.encode("utf-8"))
        assert result.text == "Here is my answer."
        assert result.thinking == "Let me think about this."


# ===================================================================
# VAL-PARSE-017: Tool use blocks (name, id, input)
# ===================================================================

class TestToolUseBlocks:
    """Tool use blocks captured with name, id, and accumulated JSON input."""

    SSE_TOOL_USE = _sse_body(
        ("message_start", {
            "type": "message_start",
            "message": {
                "id": "msg_003",
                "type": "message",
                "role": "assistant",
                "model": "deepseek-v4-flash",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 200, "output_tokens": 0},
            },
        }),
        ("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "call_00_abc123",
                "name": "Read",
                "input": {},
            },
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": "{"},
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '"file_path"'},
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": ':"test.txt"}'},
        }),
        ("content_block_stop", {
            "type": "content_block_stop",
            "index": 0,
        }),
        ("content_block_start", {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "call_01_def456",
                "name": "Grep",
                "input": {},
            },
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"pattern"'},
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": ':"search","path":"."}'},
        }),
        ("content_block_stop", {
            "type": "content_block_stop",
            "index": 1,
        }),
        ("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 30},
        }),
        ("message_stop", {"type": "message_stop"}),
    )

    def test_tool_use_name_and_id(self):
        """Tool use blocks have correct name and id fields."""
        result = parse_response(self.SSE_TOOL_USE.encode("utf-8"))
        assert len(result.tool_uses) == 2
        assert result.tool_uses[0].name == "Read"
        assert result.tool_uses[0].id == "call_00_abc123"
        assert result.tool_uses[1].name == "Grep"
        assert result.tool_uses[1].id == "call_01_def456"

    def test_tool_use_input_accumulated(self):
        """Tool input is accumulated from input_json_delta fragments."""
        result = parse_response(self.SSE_TOOL_USE.encode("utf-8"))
        assert result.tool_uses[0].input == {"file_path": "test.txt"}
        assert result.tool_uses[1].input == {"pattern": "search", "path": "."}

    def test_tool_use_stop_reason(self):
        """A response with tool calls has stop_reason=tool_use."""
        result = parse_response(self.SSE_TOOL_USE.encode("utf-8"))
        assert result.stop_reason == "tool_use"

    def test_malformed_tool_json_does_not_crash(self, caplog):
        """Invalid input_json_delta produces empty input dict and warning."""
        sse = _sse_body(
            ("content_block_start", {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "call_bad",
                    "name": "Read",
                    "input": {},
                },
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": "not valid json"},
            }),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            }),
        ).encode("utf-8")
        caplog.set_level(logging.WARNING)
        result = parse_response(sse)
        assert len(result.tool_uses) == 1
        assert result.tool_uses[0].input == {}
        assert result.tool_uses[0].name == "Read"


# ===================================================================
# VAL-PARSE-018: Token usage
# ===================================================================

class TestTokenUsage:
    """Token counts extracted from message_start and message_delta."""

    SSE_TOKEN_USAGE = _sse_body(
        ("message_start", {
            "type": "message_start",
            "message": {
                "id": "msg_004",
                "type": "message",
                "role": "assistant",
                "model": "deepseek-v4-flash",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 1259,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 1792,
                    "output_tokens": 0,
                    "service_tier": "standard",
                },
            },
        }),
        ("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Some output."},
        }),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {
                "input_tokens": 1259,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 1792,
                "output_tokens": 67,
                "service_tier": "standard",
            },
        }),
        ("message_stop", {"type": "message_stop"}),
    )

    def test_extracts_input_tokens(self):
        """input_tokens extracted from message_start usage."""
        result = parse_response(self.SSE_TOKEN_USAGE.encode("utf-8"))
        assert result.input_tokens == 1259

    def test_extracts_output_tokens_from_message_delta(self):
        """output_tokens extracted from message_delta (overrides message_start 0)."""
        result = parse_response(self.SSE_TOKEN_USAGE.encode("utf-8"))
        assert result.output_tokens == 67

    def test_extracts_cache_tokens(self):
        """cache_creation and cache_read tokens extracted."""
        result = parse_response(self.SSE_TOKEN_USAGE.encode("utf-8"))
        assert result.cache_creation_input_tokens == 0
        assert result.cache_read_input_tokens == 1792

    def test_message_delta_overrides_message_start(self):
        """message_delta usage values override message_start values when present."""
        sse = _sse_body(
            ("message_start", {
                "type": "message_start",
                "message": {"id": "m1", "content": []},
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 0,
                },
            }),
            ("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {
                    "input_tokens": 999,  # Different from message_start
                    "output_tokens": 50,
                },
            }),
        ).encode("utf-8")
        result = parse_response(sse)
        # message_delta overrides input_tokens
        assert result.input_tokens == 999
        assert result.output_tokens == 50

    def test_service_tier_extracted(self):
        """service_tier field extracted from usage."""
        result = parse_response(self.SSE_TOKEN_USAGE.encode("utf-8"))
        assert result.service_tier == "standard"

    def test_fallback_to_message_start_when_message_delta_has_no_usage(self):
        """Token usage falls back to message_start when message_delta has no usage field."""
        sse = _sse_body(
            ("message_start", {
                "type": "message_start",
                "message": {
                    "id": "msg_fallback",
                    "type": "message",
                    "role": "assistant",
                    "model": "deepseek-v4-flash",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                },
                "usage": {
                    "input_tokens": 500,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 1000,
                    "output_tokens": 0,
                    "service_tier": "standard",
                },
            }),
            ("content_block_start", {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello!"},
            }),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            # message_delta has NO usage field — fall back to message_start
            ("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            }),
            ("message_stop", {"type": "message_stop"}),
        ).encode("utf-8")
        result = parse_response(sse)
        # input_tokens and cache values come from message_start
        assert result.input_tokens == 500
        assert result.cache_creation_input_tokens == 0
        assert result.cache_read_input_tokens == 1000
        assert result.service_tier == "standard"
        # output_tokens is not set from message_start (always 0 placeholder);
        # since message_delta has no usage, it remains None
        assert result.output_tokens is None


# ===================================================================
# VAL-PARSE-019: Stop reason
# ===================================================================

class TestStopReason:
    """stop_reason extracted from message_delta."""

    def test_stop_reason_end_turn(self):
        """stop_reason=end_turn correctly extracted."""
        sse = _sse_body(
            ("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 10},
            }),
        ).encode("utf-8")
        result = parse_response(sse)
        assert result.stop_reason == "end_turn"

    def test_stop_reason_tool_use(self):
        """stop_reason=tool_use correctly extracted."""
        sse = _sse_body(
            ("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 10},
            }),
        ).encode("utf-8")
        result = parse_response(sse)
        assert result.stop_reason == "tool_use"

    def test_stop_sequence_extracted(self):
        """stop_sequence extracted from message_delta."""
        sse = _sse_body(
            ("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": "something"},
                "usage": {"output_tokens": 1},
            }),
        ).encode("utf-8")
        result = parse_response(sse)
        assert result.stop_sequence == "something"


# ===================================================================
# Combined: thinking + text + tool_use in one stream
# ===================================================================

class TestCombinedContent:
    """A single SSE stream with thinking, text, and tool_use blocks."""

    SSE_COMBINED = _sse_body(
        ("message_start", {
            "type": "message_start",
            "message": {
                "id": "msg_005",
                "type": "message",
                "role": "assistant",
                "model": "deepseek-v4-flash",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 3091,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 16384,
                    "output_tokens": 0,
                    "service_tier": "standard",
                },
            },
        }),
        # Thinking block (index 0)
        ("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "I need to "},
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "read the file."},
        }),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        # Text block (index 1)
        ("content_block_start", {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "text", "text": ""},
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "Let me look at "},
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "the file."},
        }),
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        # Tool use block (index 2)
        ("content_block_start", {
            "type": "content_block_start",
            "index": 2,
            "content_block": {
                "type": "tool_use",
                "id": "call_00_tool1",
                "name": "Read",
                "input": {},
            },
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": '{"file_path":'},
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": '"main.go"}'},
        }),
        ("content_block_stop", {"type": "content_block_stop", "index": 2}),
        # message_delta
        ("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {
                "input_tokens": 3091,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 16384,
                "output_tokens": 45,
                "service_tier": "standard",
            },
        }),
        ("message_stop", {"type": "message_stop"}),
    )

    def test_combined_thinking_text_and_tool(self):
        """All three content types extracted from a combined stream."""
        result = parse_response(self.SSE_COMBINED.encode("utf-8"))
        assert result.thinking == "I need to read the file."
        assert result.text == "Let me look at the file."
        assert len(result.tool_uses) == 1
        assert result.tool_uses[0].name == "Read"
        assert result.tool_uses[0].id == "call_00_tool1"
        assert result.tool_uses[0].input == {"file_path": "main.go"}
        assert result.stop_reason == "tool_use"

    def test_combined_token_counts(self):
        """Token counts correctly set from the combined stream."""
        result = parse_response(self.SSE_COMBINED.encode("utf-8"))
        assert result.input_tokens == 3091
        assert result.output_tokens == 45
        assert result.cache_read_input_tokens == 16384


# ===================================================================
# Message metadata
# ===================================================================

class TestMessageMetadata:
    """Message metadata extracted from message_start event."""

    def test_message_id_model_and_role(self):
        """message_id, model, and role extracted from message_start."""
        sse = _sse_body(
            ("message_start", {
                "type": "message_start",
                "message": {
                    "id": "msg_abc123",
                    "type": "message",
                    "role": "assistant",
                    "model": "deepseek-v4-pro",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 5, "output_tokens": 0},
                },
            }),
            ("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            }),
        ).encode("utf-8")
        result = parse_response(sse)
        assert result.message_id == "msg_abc123"
        assert result.model == "deepseek-v4-pro"
        assert result.role == "assistant"


# ===================================================================
# ping events are silently ignored
# ===================================================================

class TestPingEvents:
    """ping events are ignored without errors."""

    def test_ping_events_do_not_crash(self):
        """ping events interspersed with content are handled gracefully."""
        sse = _sse_body(
            ("message_start", {
                "type": "message_start",
                "message": {"id": "m1", "content": []},
                "usage": {"input_tokens": 10, "output_tokens": 0},
            }),
            ("ping", {"type": "ping"}),
            ("ping", {"type": "ping"}),
            ("content_block_start", {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Works."},
            }),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 2},
            }),
        ).encode("utf-8")
        result = parse_response(sse)
        assert result.text == "Works."
        assert result.stop_reason == "end_turn"


# ===================================================================
# status_code handling
# ===================================================================

class TestStatusCode:
    """status_code is recorded on the ParsedResponse."""

    def test_default_status_200(self):
        """Default status_code is 200."""
        result = parse_response(b"")
        assert result.status_code == 200

    def test_custom_status_code(self):
        """Provided status_code is preserved."""
        result = parse_response(b"", status_code=429)
        assert result.status_code == 429
