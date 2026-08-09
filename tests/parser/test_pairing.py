"""Tests for request/response pairing and timing extraction.

Covers:
- VAL-PARSE-020: Pairs requests and responses by shared request_id (ULID)
- VAL-PARSE-021: Extracts request timing (start/end)
- VAL-PARSE-022: Extracts response timing (start/end)
- Error cases: malformed request JSON, empty response content
"""

from __future__ import annotations

import json
import logging
from unittest.mock import patch

import pytest
from mitmproxy import http

from llm_flow_viewer.parser.models import (
    LLMCall,
    ParsedRequest,
    ParsedResponse,
    Timing,
    TokenUsage,
)
from llm_flow_viewer.parser.pairing import (
    generate_request_id,
    pair_flow_to_llm_call,
    pair_flows,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_http_flow(
    method: str = "POST",
    host: str = "api.deepseek.com",
    path: str = "/anthropic/v1/messages",
    req_content: bytes | None = None,
    status_code: int = 200,
    resp_content: bytes | None = None,
    has_response: bool = True,
    req_ts_start: float = 1000.0,
    req_ts_end: float = 1000.05,
    resp_ts_start: float = 1000.1,
    resp_ts_end: float = 1000.5,
) -> http.HTTPFlow:
    """Build a mock HTTPFlow with tunable request/response content and timestamps."""
    if req_content is None:
        req_content = json.dumps({
            "model": "deepseek-v4-flash",
            "max_tokens": 131072,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }).encode("utf-8")

    if resp_content is None:
        resp_content = _make_sse_body(
            ("message_start", {
                "type": "message_start",
                "message": {
                    "id": "msg_001",
                    "type": "message",
                    "role": "assistant",
                    "model": "deepseek-v4-flash",
                    "content": [],
                },
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 0,
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
            ("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 5,
                },
            }),
            ("message_stop", {"type": "message_stop"}),
        ).encode("utf-8")

    req = http.Request.make(method, f"https://{host}{path}", req_content)
    flow = http.HTTPFlow(client_conn=None, server_conn=None, live=False)
    flow.request = req
    flow.request.timestamp_start = req_ts_start
    flow.request.timestamp_end = req_ts_end

    if has_response:
        resp = http.Response.make(status_code, resp_content)
        resp.timestamp_start = resp_ts_start
        resp.timestamp_end = resp_ts_end
        flow.response = resp
    else:
        flow.response = None

    return flow


def _make_sse_body(*events: tuple[str, dict]) -> str:
    """Build a full SSE body from (event_type, data_dict) pairs."""
    parts = []
    for event_type, data in events:
        parts.append(f"event: {event_type}\ndata: {json.dumps(data, separators=(',', ':'))}\n")
    return "\n".join(parts)


# ===================================================================
# generate_request_id
# ===================================================================

class TestGenerateRequestId:
    """ULID generation for request-response pairs."""

    def test_generates_unique_ulid_string(self):
        """generate_request_id returns a non-empty string."""
        rid = generate_request_id()
        assert isinstance(rid, str)
        assert len(rid) > 0

    def test_ulid_is_26_characters(self):
        """ULID string is exactly 26 characters (standard Crockford base32)."""
        rid = generate_request_id()
        assert len(rid) == 26

    def test_ulid_contains_only_valid_characters(self):
        """ULID uses Crockford base32 characters (0-9, A-Z excluding I, L, O, U)."""
        rid = generate_request_id()
        valid_chars = set("0123456789ABCDEFGHJKMNPQRSTVWXYZ")
        assert all(c in valid_chars for c in rid), f"Invalid ULID characters in: {rid}"

    def test_generates_unique_ids(self):
        """Multiple calls produce different ULIDs."""
        ids = {generate_request_id() for _ in range(100)}
        assert len(ids) == 100


# ===================================================================
# pair_flow_to_llm_call — single flow
# ===================================================================

class TestPairFlowToLLMCall:
    """VAL-PARSE-020: Pair a single flow into an LLMCall."""

    def test_pairs_complete_flow(self):
        """A complete flow with valid request and response produces an LLMCall."""
        flow = _make_http_flow()
        call = pair_flow_to_llm_call(flow)
        assert call is not None
        assert isinstance(call, LLMCall)

    def test_ulid_is_valid(self):
        """The LLMCall has a valid 26-char ULID as request_id."""
        flow = _make_http_flow()
        call = pair_flow_to_llm_call(flow)
        assert call is not None
        assert len(call.request_id) == 26
        valid_chars = set("0123456789ABCDEFGHJKMNPQRSTVWXYZ")
        assert all(c in valid_chars for c in call.request_id)

    def test_both_objects_present(self):
        """LLMCall contains both a ParsedRequest and a ParsedResponse."""
        flow = _make_http_flow()
        call = pair_flow_to_llm_call(flow)
        assert call is not None
        assert isinstance(call.request, ParsedRequest)
        assert isinstance(call.response, ParsedResponse)

    def test_request_id_matches_both_objects(self):
        """ParsedRequest.request_id == ParsedResponse.request_id == LLMCall.request_id."""
        flow = _make_http_flow()
        call = pair_flow_to_llm_call(flow)
        assert call is not None
        assert call.request_id == call.request.request_id
        assert call.request_id == call.response.request_id

    def test_all_four_timestamps_populated(self):
        """Timing object has all four timestamps as non-None floats with correct order."""
        flow = _make_http_flow(
            req_ts_start=1000.0,
            req_ts_end=1000.05,
            resp_ts_start=1000.1,
            resp_ts_end=1000.5,
        )
        call = pair_flow_to_llm_call(flow)
        assert call is not None
        timing = call.timing
        assert timing is not None
        assert timing.request_start == 1000.0
        assert timing.request_end == 1000.05
        assert timing.response_start == 1000.1
        assert timing.response_end == 1000.5

    def test_timestamps_chronological_order(self):
        """request_start <= request_end < response_start <= response_end."""
        flow = _make_http_flow(
            req_ts_start=1000.0,
            req_ts_end=1000.05,
            resp_ts_start=1000.1,
            resp_ts_end=1000.5,
        )
        call = pair_flow_to_llm_call(flow)
        assert call is not None
        t = call.timing
        assert t.request_start <= t.request_end
        assert t.request_end <= t.response_start
        assert t.response_start <= t.response_end

    def test_request_timestamps_on_parsed_request(self):
        """ParsedRequest has timestamp_start and timestamp_end set from flow."""
        flow = _make_http_flow(req_ts_start=1000.0, req_ts_end=1000.05)
        call = pair_flow_to_llm_call(flow)
        assert call is not None
        assert call.request.timestamp_start == 1000.0
        assert call.request.timestamp_end == 1000.05

    def test_response_timestamps_on_parsed_response(self):
        """ParsedResponse has timestamp_start and timestamp_end set from flow."""
        flow = _make_http_flow(resp_ts_start=1000.1, resp_ts_end=1000.5)
        call = pair_flow_to_llm_call(flow)
        assert call is not None
        assert call.response.timestamp_start == 1000.1
        assert call.response.timestamp_end == 1000.5

    def test_token_usage_populated(self):
        """LLMCall.token_usage has prompt_tokens and completion_tokens from response."""
        flow = _make_http_flow()
        call = pair_flow_to_llm_call(flow)
        assert call is not None
        tu = call.token_usage
        assert tu is not None
        assert tu.prompt_tokens == 100
        assert tu.completion_tokens == 5

    def test_request_body_parsed_correctly(self):
        """ParsedRequest contains the expected model and messages."""
        flow = _make_http_flow()
        call = pair_flow_to_llm_call(flow)
        assert call is not None
        assert call.request.model == "deepseek-v4-flash"
        assert call.request.max_tokens == 131072
        assert len(call.request.messages) == 1
        assert call.request.messages[0]["role"] == "user"

    def test_response_body_parsed_correctly(self):
        """ParsedResponse contains the expected text output."""
        flow = _make_http_flow()
        call = pair_flow_to_llm_call(flow)
        assert call is not None
        assert call.response.text == "Hello!"
        assert call.response.stop_reason == "end_turn"


# ===================================================================
# Error handling — request parsing failures
# ===================================================================

class TestPairFlowRequestErrors:
    """VAL-PARSE-021 (verification): Request parsing failure is handled gracefully."""

    def test_skips_malformed_json_request(self, caplog):
        """Flow with malformed JSON in request body is skipped, warning logged."""
        flow = _make_http_flow(req_content=b"this is not valid json")
        caplog.set_level(logging.WARNING)
        call = pair_flow_to_llm_call(flow)
        assert call is None
        combined = " ".join(caplog.messages).lower()
        assert any(kw in combined for kw in ("malformed", "skip", "unparseable", "empty"))

    def test_skips_empty_request_content(self, caplog):
        """Flow with empty request content is skipped, warning logged."""
        flow = _make_http_flow(req_content=b"")
        caplog.set_level(logging.WARNING)
        call = pair_flow_to_llm_call(flow)
        assert call is None
        assert any("skip" in msg.lower() or "empty" in msg.lower()
                   for msg in caplog.messages)

    def test_skips_none_request_content(self, caplog):
        """Flow with None request content is skipped, warning logged."""
        flow = _make_http_flow()
        # Override request content to None after creation
        flow.request.content = None
        caplog.set_level(logging.WARNING)
        call = pair_flow_to_llm_call(flow)
        assert call is None
        assert any("skip" in msg.lower() or "empty" in msg.lower() or "none" in msg.lower()
                   for msg in caplog.messages)


# ===================================================================
# Error handling — response parsing failures
# ===================================================================

class TestPairFlowResponseErrors:
    """VAL-PARSE-022 (verification): Empty/missing response is handled gracefully."""

    def test_skips_flow_with_no_response_object(self, caplog):
        """Flow with response=None should not produce an LLMCall (caught by filter)."""
        flow = _make_http_flow(has_response=False)
        caplog.set_level(logging.WARNING)
        call = pair_flow_to_llm_call(flow)
        # is_llm_call_flow rejects flows without response, so pair_flow_to_llm_call
        # should return None or the calling code skips it
        # The function currently only handles cases that pass the filter
        assert call is None

    def test_skips_flow_with_empty_response_content(self, caplog):
        """Flow with empty response content should not produce an LLMCall."""
        flow = _make_http_flow(resp_content=b"")
        caplog.set_level(logging.WARNING)
        call = pair_flow_to_llm_call(flow)
        assert call is None


# ===================================================================
# pair_flows — end-to-end
# ===================================================================

class TestPairFlows:
    """pair_flows processes multiple flows and yields LLMCalls."""

    def test_non_existent_file_yields_empty(self, caplog):
        """pair_flows handles non-existent file gracefully (empty result)."""
        caplog.set_level(logging.WARNING)
        calls = list(pair_flows(["D:\\nonexistent\\file.flow"]))
        assert calls == []


# ===================================================================
# TokenUsage mapping from response
# ===================================================================

class TestTokenUsageMapping:
    """TokenUsage object is populated from ParsedResponse token fields."""

    def test_token_usage_from_response(self):
        """TokenUsage.prompt_tokens and completion_tokens match response fields."""
        flow = _make_http_flow()
        call = pair_flow_to_llm_call(flow)
        assert call is not None
        assert call.token_usage.prompt_tokens == call.response.input_tokens
        assert call.token_usage.completion_tokens == call.response.output_tokens
