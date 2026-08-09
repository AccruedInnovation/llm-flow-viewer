"""Tests for the data model classes (TokenUsage, Timing, ToolResult, LLMCall, Session)."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime

import pytest

from llm_flow_viewer.parser.models import (
    LLMCall,
    ParsedRequest,
    ParsedResponse,
    Session,
    Timing,
    TokenUsage,
    ToolResult,
    ToolUse,
)


# ===================================================================
# TokenUsage
# ===================================================================

class TestTokenUsage:
    """TokenUsage holds prompt, completion, and total token counts."""

    def test_can_instantiate_with_required_fields(self):
        """TokenUsage can be created with no arguments, all fields default to None."""
        tu = TokenUsage()
        assert tu.prompt_tokens is None
        assert tu.completion_tokens is None
        assert tu.total_tokens is None

    def test_can_instantiate_with_all_fields(self):
        """TokenUsage can be created with all fields specified."""
        tu = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert tu.prompt_tokens == 100
        assert tu.completion_tokens == 50
        assert tu.total_tokens == 150

    def test_optional_fields_default_to_none(self):
        """Each optional field defaults to None when not provided."""
        tu = TokenUsage(prompt_tokens=10)
        assert tu.prompt_tokens == 10
        assert tu.completion_tokens is None
        assert tu.total_tokens is None

    def test_json_serializable_via_asdict(self):
        """TokenUsage is JSON-serializable via dataclasses.asdict."""
        tu = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        d = asdict(tu)
        assert d == {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        # Verify it can be round-tripped through JSON
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        assert restored == d


# ===================================================================
# Timing
# ===================================================================

class TestTiming:
    """Timing holds request and response timestamps."""

    def test_can_instantiate_with_required_fields(self):
        """Timing can be created with no arguments, all fields default to None."""
        t = Timing()
        assert t.request_start is None
        assert t.request_end is None
        assert t.response_start is None
        assert t.response_end is None

    def test_can_instantiate_with_all_fields(self):
        """Timing can be created with all fields specified."""
        t = Timing(
            request_start=1000.0,
            request_end=1000.1,
            response_start=1000.2,
            response_end=1000.5,
        )
        assert t.request_start == 1000.0
        assert t.request_end == 1000.1
        assert t.response_start == 1000.2
        assert t.response_end == 1000.5

    def test_optional_fields_default_to_none(self):
        """Each optional field defaults to None when not provided."""
        t = Timing(request_start=1.0)
        assert t.request_start == 1.0
        assert t.request_end is None
        assert t.response_start is None
        assert t.response_end is None

    def test_json_serializable_via_asdict(self):
        """Timing is JSON-serializable via dataclasses.asdict."""
        t = Timing(
            request_start=1000.0,
            request_end=1000.1,
            response_start=1000.2,
            response_end=1000.5,
        )
        d = asdict(t)
        assert d == {
            "request_start": 1000.0,
            "request_end": 1000.1,
            "response_start": 1000.2,
            "response_end": 1000.5,
        }
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        assert restored == d


# ===================================================================
# ToolResult
# ===================================================================

class TestToolResult:
    """ToolResult holds tool_use_id and content."""

    def test_can_instantiate_with_required_fields(self):
        """ToolResult can be created with no arguments."""
        tr = ToolResult()
        assert tr.tool_use_id == ""
        assert tr.content == ""

    def test_can_instantiate_with_all_fields(self):
        """ToolResult can be created with tool_use_id and content."""
        tr = ToolResult(tool_use_id="call_00_abc", content='{"result": "success"}')
        assert tr.tool_use_id == "call_00_abc"
        assert tr.content == '{"result": "success"}'

    def test_defaults_are_empty_string(self):
        """Optional fields default to empty string."""
        tr = ToolResult(tool_use_id="call_01")
        assert tr.tool_use_id == "call_01"
        assert tr.content == ""

    def test_json_serializable_via_asdict(self):
        """ToolResult is JSON-serializable via dataclasses.asdict."""
        tr = ToolResult(tool_use_id="call_00_abc", content="file content here")
        d = asdict(tr)
        assert d == {"tool_use_id": "call_00_abc", "content": "file content here", "is_error": False}
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        assert restored == d


# ===================================================================
# LLMCall
# ===================================================================

class TestLLMCall:
    """LLMCall pairs a ParsedRequest with a ParsedResponse plus ULID, timing, token usage."""

    def test_can_instantiate_with_required_fields(self):
        """LLMCall can be created with no arguments, all optional fields default to None/empty."""
        call = LLMCall()
        assert call.request_id == ""
        assert call.request is None
        assert call.response is None
        assert call.timing is None
        assert call.token_usage is None

    def test_can_instantiate_with_all_fields(self):
        """LLMCall can be created with all fields specified."""
        req = ParsedRequest(model="deepseek-v4-flash", max_tokens=131072)
        resp = ParsedResponse(text="Hello!", stop_reason="end_turn")
        timing = Timing(request_start=1.0, request_end=2.0, response_start=2.0, response_end=5.0)
        token_usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        call = LLMCall(
            request_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            request=req,
            response=resp,
            timing=timing,
            token_usage=token_usage,
        )
        assert call.request_id == "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        assert call.request is req
        assert call.response is resp
        assert call.timing is timing
        assert call.token_usage is token_usage

    def test_fields_default_to_none(self):
        """Optional fields (request, response, timing, token_usage) default to None."""
        call = LLMCall(request_id="test-id")
        assert call.request_id == "test-id"
        assert call.request is None
        assert call.response is None
        assert call.timing is None
        assert call.token_usage is None

    def test_json_serializable_via_asdict(self):
        """LLMCall is JSON-serializable via dataclasses.asdict."""
        req = ParsedRequest(model="deepseek-v4-flash", max_tokens=131072)
        resp = ParsedResponse(text="Hello!", stop_reason="end_turn")
        timing = Timing(request_start=1.0, request_end=2.0)
        token_usage = TokenUsage(prompt_tokens=100, completion_tokens=50)

        call = LLMCall(
            request_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            request=req,
            response=resp,
            timing=timing,
            token_usage=token_usage,
        )
        d = asdict(call)
        assert d["request_id"] == "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        assert d["request"]["model"] == "deepseek-v4-flash"
        assert d["response"]["text"] == "Hello!"
        assert d["timing"]["request_start"] == 1.0
        assert d["token_usage"]["prompt_tokens"] == 100

        # Round-trip through JSON
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        assert restored["request_id"] == d["request_id"]
        assert restored["request"]["model"] == "deepseek-v4-flash"


# ===================================================================
# Session
# ===================================================================

class TestSession:
    """Session holds a list of LLMCall objects and metadata."""

    def test_can_instantiate_with_required_fields(self):
        """Session can be created with no arguments, defaults to empty list and zeros."""
        session = Session()
        assert session.index == 0
        assert session.task_name == ""
        assert session.model == ""
        assert session.calls == []

    def test_can_instantiate_with_all_fields(self):
        """Session can be created with index, task_name, model, and calls."""
        call1 = LLMCall(request_id="call-1")
        call2 = LLMCall(request_id="call-2")
        session = Session(
            index=1,
            task_name="analyze_codebase",
            model="deepseek-v4-flash",
            calls=[call1, call2],
        )
        assert session.index == 1
        assert session.task_name == "analyze_codebase"
        assert session.model == "deepseek-v4-flash"
        assert len(session.calls) == 2
        assert session.calls[0].request_id == "call-1"
        assert session.calls[1].request_id == "call-2"

    def test_calls_defaults_to_empty_list(self):
        """calls field defaults to empty list when not provided."""
        session = Session(index=2, task_name="test", model="deepseek-v4-pro")
        assert session.calls == []

    def test_json_serializable_via_asdict(self):
        """Session is JSON-serializable via dataclasses.asdict."""
        call = LLMCall(request_id="call-1")
        session = Session(
            index=1,
            task_name="analyze_codebase",
            model="deepseek-v4-flash",
            calls=[call],
        )
        d = asdict(session)
        assert d["index"] == 1
        assert d["task_name"] == "analyze_codebase"
        assert d["model"] == "deepseek-v4-flash"
        assert len(d["calls"]) == 1
        assert d["calls"][0]["request_id"] == "call-1"

        # Round-trip through JSON
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        assert restored["index"] == 1
        assert restored["task_name"] == "analyze_codebase"


# ===================================================================
# Edge cases and validation
# ===================================================================

class TestDataclassDefaults:
    """All dataclasses should have proper defaults for optional fields."""

    def test_all_optional_fields_default_to_none(self):
        """Verify all @dataclass fields annotated as Optional default to None."""
        # TokenUsage
        tu = TokenUsage()
        assert tu.prompt_tokens is None
        assert tu.completion_tokens is None
        assert tu.total_tokens is None

        # Timing
        t = Timing()
        assert t.request_start is None
        assert t.request_end is None
        assert t.response_start is None
        assert t.response_end is None

    def test_all_list_fields_default_to_empty_list(self):
        """Verify list fields default to empty list, not None."""
        session = Session()
        assert session.calls == []

    def test_all_str_fields_default_to_empty(self):
        """Verify str fields default to empty string, not None."""
        tr = ToolResult()
        assert tr.tool_use_id == ""
        assert tr.content == ""

        call = LLMCall()
        assert call.request_id == ""

        session = Session()
        assert session.task_name == ""
        assert session.model == ""


class TestTypeHints:
    """Verify type hints are correct for all fields."""

    def test_token_usage_fields_are_optional_int(self):
        """TokenUsage fields should accept None or int values."""
        tu = TokenUsage(prompt_tokens=None, completion_tokens=None, total_tokens=None)
        assert tu.prompt_tokens is None
        tu.prompt_tokens = 42
        assert tu.prompt_tokens == 42

    def test_timing_fields_are_optional_float(self):
        """Timing fields should accept None or float values."""
        t = Timing(request_start=None, request_end=None)
        assert t.request_start is None
        t.request_start = 1234.56
        assert isinstance(t.request_start, float)

    def test_session_calls_are_list_of_llmcall(self):
        """Session.calls should be a list of LLMCall objects."""
        session = Session()
        session.calls.append(LLMCall(request_id="call-1"))
        session.calls.append(LLMCall(request_id="call-2"))
        assert len(session.calls) == 2
        assert all(isinstance(c, LLMCall) for c in session.calls)

    def test_llmcall_holds_parsed_objects(self):
        """LLMCall should accept ParsedRequest, ParsedResponse objects."""
        req = ParsedRequest(model="test-model")
        resp = ParsedResponse(text="test response")
        call = LLMCall(request=req, response=resp)
        assert isinstance(call.request, ParsedRequest)
        assert isinstance(call.response, ParsedResponse)
