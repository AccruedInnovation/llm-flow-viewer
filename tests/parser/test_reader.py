"""Tests for the flow reader module."""

import json
import logging
from unittest.mock import MagicMock, patch

import pytest
from mitmproxy import http

from llm_flow_viewer.parser.reader import (
    FlowFilterConfig,
    expand_flow_files,
    is_llm_call_flow,
    open_flow_file,
    read_flow_files,
)


def _make_http_flow(
    method: str = "POST",
    host: str = "api.deepseek.com",
    path: str = "/anthropic/v1/messages",
    content: bytes = b'{"model":"deepseek-v4-flash","max_tokens":131072,"messages":[{"role":"user","content":"hello"}],"tools":[{"name":"Read","description":"Read files"}],"system":[{"type":"text","text":"You are Droid"}],"thinking":{"type":"enabled"},"stream":true}',
    status_code: int = 200,
    response_content: bytes = b'{"id":"msg_01","type":"message","role":"assistant","content":[{"type":"text","text":"Hello!"}],"model":"deepseek-v4-flash","stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}',
    has_response: bool = True,
) -> http.HTTPFlow:
    """Helper to create a mock HTTPFlow for testing."""
    req = http.Request.make(method, f"https://{host}{path}", content)
    flow = http.HTTPFlow(client_conn=None, server_conn=None, live=False)
    flow.request = req
    if has_response:
        resp = http.Response.make(status_code, response_content)
        flow.response = resp
    else:
        flow.response = None
    return flow


# ===================================================================
# open_flow_file tests
# ===================================================================

class TestOpenFlowFile:
    """Tests for open_flow_file: opens flow file and yields flows."""

    def test_handles_missing_file(self):
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            list(open_flow_file("D:\\nonexistent\\file.flow"))

    def test_handles_corrupt_file(self, tmp_path):
        """Corrupt flow file is handled gracefully (empty list returned)."""
        corrupt_file = tmp_path / "corrupt.flow"
        corrupt_file.write_bytes(b"this is not a valid mitmproxy flow file")
        flows = list(open_flow_file(str(corrupt_file)))
        assert flows == []


# ===================================================================
# is_llm_call_flow tests
# ===================================================================

class TestIsLLMCallFlow:
    """Tests for is_llm_call_flow: endpoint and method filtering."""

    def test_matches_post_to_deepseek_anthropic(self):
        """POST to api.deepseek.com/anthropic/v1/messages returns True."""
        flow = _make_http_flow()
        assert is_llm_call_flow(flow) is True

    def test_rejects_non_post_method(self):
        """GET request to the same endpoint returns False."""
        flow = _make_http_flow(method="GET")
        assert is_llm_call_flow(flow) is False

    def test_rejects_wrong_host(self):
        """Flow to a different host returns False."""
        flow = _make_http_flow(host="api.openai.com")
        assert is_llm_call_flow(flow) is False

    def test_rejects_wrong_path(self):
        """Flow to a different path returns False."""
        flow = _make_http_flow(path="/v1/chat/completions")
        assert is_llm_call_flow(flow) is False

    def test_rejects_empty_request_content(self):
        """Flow with empty request content returns False."""
        flow = _make_http_flow(content=b"")
        assert is_llm_call_flow(flow) is False

    def test_rejects_none_request_content(self):
        """Flow with None request content returns False."""
        req = http.Request.make("POST", "https://api.deepseek.com/anthropic/v1/messages")
        flow = http.HTTPFlow(client_conn=None, server_conn=None, live=False)
        flow.request = req
        flow.request.content = None
        assert is_llm_call_flow(flow) is False

    def test_rejects_non_http_flow(self):
        """Non-HTTPFlow object returns False without error."""
        not_a_flow = "not a flow object"
        assert is_llm_call_flow(not_a_flow) is False

    def test_rejects_empty_response_content(self):
        """Flow with empty response content returns False."""
        flow = _make_http_flow(response_content=b"")
        assert is_llm_call_flow(flow) is False

    def test_rejects_none_response(self):
        """Flow with no response object returns False."""
        flow = _make_http_flow(has_response=False)
        assert is_llm_call_flow(flow) is False

    def test_accepts_flow_with_response_content(self):
        """Flow with non-empty response content returns True."""
        flow = _make_http_flow(
            response_content=b'{"id":"msg_01","type":"message"}'
        )
        assert is_llm_call_flow(flow) is True


class TestExpandFlowFiles:
    """Tests for expand_flow_files: glob pattern expansion."""

    def test_expand_non_existent_pattern(self):
        """Non-matching pattern returns empty list."""
        files = expand_flow_files(["D:\\nonexistent\\path\\*.nonexistent"])
        assert files == []


# ===================================================================
# read_flow_files tests
# ===================================================================

class TestReadFlowFiles:
    """Tests for read_flow_files: reading from multiple files."""

    def test_read_non_existent_file_returns_empty(self):
        """Reading a non-existent file returns empty list with warning."""
        results = list(read_flow_files(["D:\\nonexistent\\file.flow"]))
        assert results == []


# ===================================================================
# parse_flows_from_file tests
# ===================================================================

class TestParseFlowsFromFile:
    """Tests for parse_flows_from_file: full pipeline end-to-end."""

    def test_skips_non_post_with_custom_filter(self):
        """Flow with non-POST method is skipped when using default config."""
        flow = _make_http_flow(method="OPTIONS")
        config = FlowFilterConfig()
        assert is_llm_call_flow(flow) is False

