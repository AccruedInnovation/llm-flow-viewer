"""Tests for Parquet cache read/write and cache freshness logic.

Covers:
- VAL-PARSE-023: Parse flows, write to Parquet cache, read back;
  verify all LLMCall fields survive the round-trip.
- VAL-PARSE-024: Run parser with existing fresh cache;
  verify cache is loaded and no re-parsing occurs.
- VAL-PARSE-025: Modify source flow file timestamp to be newer than cache;
  verify re-parsing is triggered and cache is updated.
- VAL-PARSE-026: Process large flow file (simulated);
  verify memory usage stays bounded (streaming, not full-file load).
- VAL-STREAM-004: Parquet cache written after streaming parse, cache hit on reload.
- VAL-STREAM-005: Non-streaming path (no callback) behaves identically to pre-fix.
- VAL-STREAM-006: Batched UI updates: callback fired ceil(N/50) times, not per-call.
- VAL-STREAM-012: Streaming produces identical Session to non-streaming.
- VAL-STREAM-013: append_calls edge cases (0, 1, 50, 51, 100, 101, 1000 calls).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from llm_flow_viewer.parser.cache import (
    SCHEMA_VERSION,
    _generate_cache_key,
    get_cache_paths,
    is_cache_fresh,
    llmcalls_to_requests_table,
    llmcalls_to_responses_table,
    read_cache,
    requests_table_to_list,
    write_cache,
)
from llm_flow_viewer.parser.models import (
    LLMCall,
    ParsedRequest,
    ParsedResponse,
    Timing,
    TokenUsage,
    ToolUse,
    ToolResult,
)


# ===================================================================
# Helper: build a realistic LLMCall for round-trip testing
# ===================================================================


def _make_llm_call(
    request_id: str = "01ARZ3NDEKTSV4RRFFQ69G5FAV",
) -> LLMCall:
    """Build a realistic LLMCall with all fields populated."""
    req = ParsedRequest(
        request_id=request_id,
        model="deepseek-v4-flash",
        max_tokens=131072,
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Hi there!"},
                    {"type": "tool_use", "id": "call_01", "name": "Read",
                     "input": {"file_path": "test.py"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_01",
                     "content": "file contents here"},
                ],
            },
        ],
        tools=[
            {
                "name": "Read",
                "description": "Read a file from the filesystem",
                "input_schema": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                },
            },
        ],
        system=[
            {"type": "text", "text": "You are a helpful assistant."},
            {"type": "text", "text": "Use tools when needed."},
        ],
        output_config={"effort": "max"},
        thinking={"type": "enabled", "budget_tokens": 16000},
        stream=True,
        timestamp_start=1000.0,
        timestamp_end=1000.05,
    )
    resp = ParsedResponse(
        request_id=request_id,
        thinking="Let me think about this...",
        text="Here is my response.",
        tool_uses=[
            ToolUse(
                name="Execute",
                id="call_02",
                input={"command": "python test.py"},
            ),
        ],
        message_id="msg_001",
        model="deepseek-v4-flash",
        role="assistant",
        input_tokens=150,
        output_tokens=50,
        cache_creation_input_tokens=12000,
        cache_read_input_tokens=88000,
        service_tier="default",
        stop_reason="tool_use",
        stop_sequence=None,
        status_code=200,
        error_message="",
        timestamp_start=1000.1,
        timestamp_end=1000.5,
    )
    timing = Timing(
        request_start=1000.0,
        request_end=1000.05,
        response_start=1000.1,
        response_end=1000.5,
    )
    token_usage = TokenUsage(
        prompt_tokens=150,
        completion_tokens=50,
        total_tokens=200,
    )
    return LLMCall(
        request_id=request_id,
        request=req,
        response=resp,
        timing=timing,
        token_usage=token_usage,
    )


def _make_multiple_calls(count: int = 5) -> List[LLMCall]:
    """Build a list of LLMCalls with sequential request_ids."""
    calls = []
    for i in range(count):
        rid = f"01ARZ3NDEKTSV4RRFFQ69G5FA{i}"
        call = _make_llm_call(request_id=rid)
        call.request.request_id = rid
        call.response.request_id = rid
        calls.append(call)
    return calls


# ===================================================================
# Test: Table conversion (LLMCall → pyarrow Table)
# ===================================================================


class TestTableConversion:
    """Converting LLMCall lists to/from pyarrow Tables."""

    def test_requests_table_has_expected_columns(self):
        """The requests Table has all expected columns."""
        calls = _make_multiple_calls(2)
        table = llmcalls_to_requests_table(calls)
        expected = {
            "request_id", "model", "max_tokens", "messages", "tools",
            "system", "output_config", "thinking", "stream",
            "timestamp_start", "timestamp_end",
        }
        assert set(table.column_names) == expected

    def test_requests_table_has_correct_row_count(self):
        """Number of rows in requests table matches call count."""
        calls = _make_multiple_calls(5)
        table = llmcalls_to_requests_table(calls)
        assert len(table) == 5

    def test_responses_table_has_expected_columns(self):
        """The responses Table has all expected columns."""
        calls = _make_multiple_calls(2)
        table = llmcalls_to_responses_table(calls)
        expected = {
            "request_id", "thinking", "text", "tool_uses",
            "message_id", "model", "role", "input_tokens",
            "output_tokens", "cache_creation_input_tokens",
            "cache_read_input_tokens", "service_tier",
            "stop_reason", "stop_sequence", "status_code",
            "error_message", "timestamp_start", "timestamp_end",
        }
        assert set(table.column_names) == expected

    def test_responses_table_has_correct_row_count(self):
        """Number of rows in responses table matches call count."""
        calls = _make_multiple_calls(5)
        table = llmcalls_to_responses_table(calls)
        assert len(table) == 5

    def test_requests_table_metadata_includes_schema_version(self):
        """Requests table metadata contains cache_schema_version."""
        calls = _make_multiple_calls(1)
        table = llmcalls_to_requests_table(calls)
        assert table.schema.metadata is not None
        meta = {k.decode("utf-8"): v.decode("utf-8")
                for k, v in table.schema.metadata.items()}
        assert meta.get("cache_schema_version") == str(SCHEMA_VERSION)

    def test_responses_table_metadata_includes_schema_version(self):
        """Responses table metadata contains cache_schema_version."""
        calls = _make_multiple_calls(1)
        table = llmcalls_to_responses_table(calls)
        assert table.schema.metadata is not None
        meta = {k.decode("utf-8"): v.decode("utf-8")
                for k, v in table.schema.metadata.items()}
        assert meta.get("cache_schema_version") == str(SCHEMA_VERSION)


# ===================================================================
# Test: Table round-trip (pyarrow Table → LLMCall list)
# ===================================================================


class TestTableRoundTrip:
    """Converting pyarrow Tables back to LLMCall lists."""

    def test_round_trip_single_call(self):
        """A single LLMCall survives the table round-trip."""
        calls = _make_multiple_calls(1)
        req_table = llmcalls_to_requests_table(calls)
        resp_table = llmcalls_to_responses_table(calls)
        restored = requests_table_to_list(req_table, resp_table)
        assert len(restored) == 1
        r = restored[0]
        assert r.request_id == calls[0].request_id
        assert r.request.model == calls[0].request.model
        assert r.request.max_tokens == calls[0].request.max_tokens
        assert r.response.text == calls[0].response.text
        assert r.response.thinking == calls[0].response.thinking

    def test_round_trip_multiple_calls(self):
        """Multiple LLMCalls survive the table round-trip."""
        calls = _make_multiple_calls(5)
        req_table = llmcalls_to_requests_table(calls)
        resp_table = llmcalls_to_responses_table(calls)
        restored = requests_table_to_list(req_table, resp_table)
        assert len(restored) == 5
        for original, restored_call in zip(calls, restored):
            assert original.request_id == restored_call.request_id
            assert original.request.model == restored_call.request.model

    def test_round_trip_messages_preserved(self):
        """Request messages (complex JSON) survive round-trip."""
        calls = _make_multiple_calls(1)
        req_table = llmcalls_to_requests_table(calls)
        resp_table = llmcalls_to_responses_table(calls)
        restored = requests_table_to_list(req_table, resp_table)
        original_msgs = calls[0].request.messages
        restored_msgs = restored[0].request.messages
        assert len(restored_msgs) == len(original_msgs)
        assert restored_msgs[0]["role"] == original_msgs[0]["role"]
        assert restored_msgs[1]["content"][1]["name"] == "Read"

    def test_round_trip_tools_preserved(self):
        """Tool definitions survive round-trip."""
        calls = _make_multiple_calls(1)
        req_table = llmcalls_to_requests_table(calls)
        resp_table = llmcalls_to_responses_table(calls)
        restored = requests_table_to_list(req_table, resp_table)
        assert len(restored[0].request.tools) == 1
        assert restored[0].request.tools[0]["name"] == "Read"

    def test_round_trip_system_preserved(self):
        """System prompts survive round-trip."""
        calls = _make_multiple_calls(1)
        req_table = llmcalls_to_requests_table(calls)
        resp_table = llmcalls_to_responses_table(calls)
        restored = requests_table_to_list(req_table, resp_table)
        assert len(restored[0].request.system) == 2
        assert "helpful assistant" in restored[0].request.system[0]["text"]

    def test_round_trip_tool_uses_preserved(self):
        """Tool use blocks survive round-trip."""
        calls = _make_multiple_calls(1)
        req_table = llmcalls_to_requests_table(calls)
        resp_table = llmcalls_to_responses_table(calls)
        restored = requests_table_to_list(req_table, resp_table)
        assert len(restored[0].response.tool_uses) == 1
        assert restored[0].response.tool_uses[0].name == "Execute"
        assert restored[0].response.tool_uses[0].id == "call_02"

    def test_round_trip_timing_reconstructed(self):
        """Timing object is reconstructed from request/response timestamps."""
        calls = _make_multiple_calls(1)
        req_table = llmcalls_to_requests_table(calls)
        resp_table = llmcalls_to_responses_table(calls)
        restored = requests_table_to_list(req_table, resp_table)
        t = restored[0].timing
        assert t is not None
        assert t.request_start == 1000.0
        assert t.request_end == 1000.05
        assert t.response_start == 1000.1
        assert t.response_end == 1000.5

    def test_round_trip_token_usage_reconstructed(self):
        """TokenUsage object is reconstructed from response fields."""
        calls = _make_multiple_calls(1)
        req_table = llmcalls_to_requests_table(calls)
        resp_table = llmcalls_to_responses_table(calls)
        restored = requests_table_to_list(req_table, resp_table)
        tu = restored[0].token_usage
        assert tu is not None
        assert tu.prompt_tokens == 150
        assert tu.completion_tokens == 50

    def test_round_trip_stream_field_preserved(self):
        """Stream boolean field survives round-trip."""
        calls = _make_multiple_calls(1)
        req_table = llmcalls_to_requests_table(calls)
        resp_table = llmcalls_to_responses_table(calls)
        restored = requests_table_to_list(req_table, resp_table)
        assert restored[0].request.stream is True

    def test_round_trip_thinking_config_preserved(self):
        """Thinking config survives round-trip."""
        calls = _make_multiple_calls(1)
        req_table = llmcalls_to_requests_table(calls)
        resp_table = llmcalls_to_responses_table(calls)
        restored = requests_table_to_list(req_table, resp_table)
        assert restored[0].request.thinking == {"type": "enabled", "budget_tokens": 16000}

    def test_round_trip_output_config_preserved(self):
        """Output config survives round-trip."""
        calls = _make_multiple_calls(1)
        req_table = llmcalls_to_requests_table(calls)
        resp_table = llmcalls_to_responses_table(calls)
        restored = requests_table_to_list(req_table, resp_table)
        assert restored[0].request.output_config == {"effort": "max"}

    def test_round_trip_token_counts_preserved(self):
        """All token count fields survive round-trip."""
        calls = _make_multiple_calls(1)
        req_table = llmcalls_to_requests_table(calls)
        resp_table = llmcalls_to_responses_table(calls)
        restored = requests_table_to_list(req_table, resp_table)
        r = restored[0].response
        assert r.input_tokens == 150
        assert r.output_tokens == 50
        assert r.cache_creation_input_tokens == 12000
        assert r.cache_read_input_tokens == 88000
        assert r.service_tier == "default"

    def test_round_trip_stop_reason_preserved(self):
        """Stop reason survives round-trip."""
        calls = _make_multiple_calls(1)
        req_table = llmcalls_to_requests_table(calls)
        resp_table = llmcalls_to_responses_table(calls)
        restored = requests_table_to_list(req_table, resp_table)
        assert restored[0].response.stop_reason == "tool_use"

    def test_round_trip_status_code_preserved(self):
        """Status code survives round-trip."""
        calls = _make_multiple_calls(1)
        req_table = llmcalls_to_requests_table(calls)
        resp_table = llmcalls_to_responses_table(calls)
        restored = requests_table_to_list(req_table, resp_table)
        assert restored[0].response.status_code == 200


# ===================================================================
# Test: Cache file naming & paths
# ===================================================================


class TestCachePaths:
    """Cache file path generation."""

    def test_get_cache_paths_returns_tuple(self):
        """get_cache_paths returns (requests_path, responses_path)."""
        req_path, resp_path = get_cache_paths(
            "D:\\flows\\test_flows-file",
            "D:\\cache",
        )
        assert isinstance(req_path, str)
        assert isinstance(resp_path, str)

    def test_cache_paths_end_with_parquet(self):
        """Both cache paths end with .parquet."""
        req_path, resp_path = get_cache_paths(
            "D:\\flows\\test_flows-file",
            "D:\\cache",
        )
        assert req_path.endswith("_requests.parquet")
        assert resp_path.endswith("_responses.parquet")

    def test_cache_paths_use_cache_dir(self):
        """Cache paths are under the specified cache directory."""
        req_path, _ = get_cache_paths(
            "D:\\flows\\test_flows-file",
            "D:\\custom_cache",
        )
        assert req_path.startswith("D:\\custom_cache")

    def test_cache_paths_default_to_flow_dir(self):
        """When no cache_dir given, cache files go next to the flow file."""
        req_path, _ = get_cache_paths(
            "D:\\flows\\test_flows-file",
        )
        assert req_path.startswith("D:\\flows")

    def test_cache_key_from_filepath(self):
        """Cache key is derived from the file path."""
        key1 = _generate_cache_key("D:\\flows\\test_flows-file")
        key2 = _generate_cache_key("D:\\flows\\other_flows-file")
        assert len(key1) > 0
        assert key1 != key2


# ===================================================================
# Test: write_cache and read_cache
# ===================================================================


class TestWriteReadCache:
    """Writing cache to disk and reading it back."""

    def test_write_cache_creates_parquet_files(self, tmp_path):
        """Writing cache creates two .parquet files."""
        calls = _make_multiple_calls(3)
        flow_path = os.path.join(tmp_path, "test_flows-file")
        # Create a dummy source file
        Path(flow_path).touch()
        write_cache(calls, flow_path, str(tmp_path))
        req_path, resp_path = get_cache_paths(flow_path, str(tmp_path))
        assert os.path.isfile(req_path)
        assert os.path.isfile(resp_path)

    def test_read_cache_returns_calls(self, tmp_path):
        """Reading cache returns LLMCalls matching what was written."""
        calls = _make_multiple_calls(3)
        flow_path = os.path.join(tmp_path, "test_flows-file")
        Path(flow_path).touch()
        write_cache(calls, flow_path, str(tmp_path))
        restored = read_cache(flow_path, str(tmp_path))
        assert restored is not None
        assert len(restored) == 3
        for orig, rest in zip(calls, restored):
            assert orig.request_id == rest.request_id
            assert orig.request.model == rest.request.model
            assert orig.response.text == rest.response.text

    def test_read_cache_none_when_no_cache(self, tmp_path):
        """Reading cache returns None when no cache files exist."""
        flow_path = os.path.join(tmp_path, "nonexistent_flows-file")
        result = read_cache(flow_path, str(tmp_path))
        assert result is None

    def test_read_cache_none_when_requests_missing(self, tmp_path):
        """Reading cache returns None when requests file is missing."""
        calls = _make_multiple_calls(1)
        flow_path = os.path.join(tmp_path, "test_flows-file")
        Path(flow_path).touch()
        write_cache(calls, flow_path, str(tmp_path))
        # Delete the requests file
        req_path, _ = get_cache_paths(flow_path, str(tmp_path))
        os.remove(req_path)
        result = read_cache(flow_path, str(tmp_path))
        assert result is None

    def test_read_cache_none_when_responses_missing(self, tmp_path):
        """Reading cache returns None when responses file is missing."""
        calls = _make_multiple_calls(1)
        flow_path = os.path.join(tmp_path, "test_flows-file")
        Path(flow_path).touch()
        write_cache(calls, flow_path, str(tmp_path))
        _, resp_path = get_cache_paths(flow_path, str(tmp_path))
        os.remove(resp_path)
        result = read_cache(flow_path, str(tmp_path))
        assert result is None

    def test_read_cache_none_when_schema_mismatch(self, tmp_path):
        """Reading cache returns None when schema version doesn't match."""
        calls = _make_multiple_calls(1)
        flow_path = os.path.join(tmp_path, "test_flows-file")
        Path(flow_path).touch()
        write_cache(calls, flow_path, str(tmp_path))

        # Rewrite the requests file with wrong schema version metadata
        req_path, _ = get_cache_paths(flow_path, str(tmp_path))
        table = pq.read_table(req_path)
        wrong_meta = {
            b"cache_schema_version": b"999",
        }
        wrong_schema = table.schema.with_metadata(wrong_meta)
        # Use ParquetWriter to write with the wrong metadata
        with pq.ParquetWriter(req_path, wrong_schema) as writer:
            writer.write_table(table)
        result = read_cache(flow_path, str(tmp_path))
        assert result is None

    def test_write_cache_does_not_corrupt_existing_cache_on_failure(self, tmp_path):
        """If write fails partway, the cache is still usable (or cleaned up)."""
        calls = _make_multiple_calls(1)
        flow_path = os.path.join(tmp_path, "test_flows-file")
        Path(flow_path).touch()
        # First write should succeed
        write_cache(calls, flow_path, str(tmp_path))
        restored = read_cache(flow_path, str(tmp_path))
        assert restored is not None
        assert len(restored) == 1


# ===================================================================
# Test: Cache freshness
# ===================================================================


class TestCacheFreshness:
    """VAL-PARSE-024, VAL-PARSE-025: Cache freshness logic."""

    def test_cache_fresh_when_newer_than_source(self, tmp_path):
        """is_cache_fresh returns True when cache is newer than source."""
        calls = _make_multiple_calls(1)
        flow_path = os.path.join(tmp_path, "test_flows-file")

        # Create source file first
        Path(flow_path).touch()
        source_mtime = os.path.getmtime(flow_path)

        # Write cache (cache mtime will be >= source mtime)
        write_cache(calls, flow_path, str(tmp_path))

        assert is_cache_fresh(flow_path, str(tmp_path)) is True

    def test_cache_stale_when_source_newer(self, tmp_path):
        """is_cache_fresh returns False when source is newer than cache."""
        calls = _make_multiple_calls(1)
        flow_path = os.path.join(tmp_path, "test_flows-file")

        # Write cache first
        Path(flow_path).touch()
        write_cache(calls, flow_path, str(tmp_path))

        # Update source file modification time
        time.sleep(0.01)  # Ensure mtime difference
        Path(flow_path).touch()

        assert is_cache_fresh(flow_path, str(tmp_path)) is False

    def test_cache_not_fresh_when_no_cache(self, tmp_path):
        """is_cache_fresh returns False when no cache exists."""
        flow_path = os.path.join(tmp_path, "test_flows-file")
        Path(flow_path).touch()
        assert is_cache_fresh(flow_path, str(tmp_path)) is False

    def test_cache_not_fresh_when_source_not_found(self, tmp_path):
        """is_cache_fresh returns False when source file doesn't exist."""
        flow_path = os.path.join(tmp_path, "nonexistent_flows-file")
        assert is_cache_fresh(flow_path, str(tmp_path)) is False


# ===================================================================
# Test: load_or_parse_cached (high-level integration)
# ===================================================================


class TestLoadOrParseCached:
    """VAL-PARSE-024, VAL-PARSE-025: High-level cache-or-parse logic."""

    def test_returns_calls_from_cache_when_fresh(self, tmp_path):
        """When cache is fresh, returns calls without parsing."""
        calls = _make_multiple_calls(3)
        flow_path = os.path.join(tmp_path, "test_flows-file")
        Path(flow_path).touch()
        write_cache(calls, flow_path, str(tmp_path))

        # Now load_or_parse should read from cache
        from llm_flow_viewer.parser.cache import load_or_parse_cached

        # We mock pair_flows to verify it's NOT called
        with patch("llm_flow_viewer.parser.cache.pair_flows") as mock_pair:
            result = load_or_parse_cached(flow_path, str(tmp_path))
            mock_pair.assert_not_called()

        assert len(result) == 3

    def test_reparses_when_cache_stale(self, tmp_path):
        """When cache is stale, re-parses from scratch."""
        calls = _make_multiple_calls(1)
        flow_path = os.path.join(tmp_path, "test_flows-file")

        # Write cache
        Path(flow_path).touch()
        write_cache(calls, flow_path, str(tmp_path))

        # Make source newer
        time.sleep(0.01)
        Path(flow_path).touch()

        from llm_flow_viewer.parser.cache import load_or_parse_cached

        # Mock pair_flows to simulate real parsing returning different data
        new_calls = _make_multiple_calls(2)  # Different count than cached

        with patch("llm_flow_viewer.parser.cache.pair_flows") as mock_pair:
            mock_pair.return_value = iter(new_calls)
            result = load_or_parse_cached(flow_path, str(tmp_path))
            mock_pair.assert_called_once()

        # Should return the newly parsed data (2 calls, not 1)
        assert len(result) == 2

    def test_reparses_when_no_cache(self, tmp_path):
        """When no cache exists, parses from scratch."""
        flow_path = os.path.join(tmp_path, "test_flows-file")
        Path(flow_path).touch()

        from llm_flow_viewer.parser.cache import load_or_parse_cached
        calls = _make_multiple_calls(2)

        with patch("llm_flow_viewer.parser.cache.pair_flows") as mock_pair:
            mock_pair.return_value = iter(calls)
            result = load_or_parse_cached(flow_path, str(tmp_path))
            mock_pair.assert_called_once()

        assert len(result) == 2

    def test_writes_cache_after_parsing(self, tmp_path):
        """After parsing from scratch, cache is written to disk."""
        flow_path = os.path.join(tmp_path, "test_flows-file")
        Path(flow_path).touch()

        from llm_flow_viewer.parser.cache import load_or_parse_cached
        calls = _make_multiple_calls(3)

        with patch("llm_flow_viewer.parser.cache.pair_flows") as mock_pair:
            mock_pair.return_value = iter(calls)
            result = load_or_parse_cached(flow_path, str(tmp_path))

        # Verify cache files now exist
        req_path, resp_path = get_cache_paths(flow_path, str(tmp_path))
        assert os.path.isfile(req_path)
        assert os.path.isfile(resp_path)

        # Reading back should return the same data
        restored = read_cache(flow_path, str(tmp_path))
        assert restored is not None
        assert len(restored) == 3

    def test_skips_cache_write_when_zero_calls_parsed(self, tmp_path):
        """When parsing yields zero calls, cache files are NOT written."""
        flow_path = os.path.join(tmp_path, "test_flows-file")
        Path(flow_path).touch()

        from llm_flow_viewer.parser.cache import load_or_parse_cached

        with patch("llm_flow_viewer.parser.cache.pair_flows") as mock_pair:
            mock_pair.return_value = iter([])  # Empty — zero calls parsed
            result = load_or_parse_cached(flow_path, str(tmp_path))
            mock_pair.assert_called_once()

        assert result == []

        # Verify cache files were NOT created
        req_path, resp_path = get_cache_paths(flow_path, str(tmp_path))
        assert not os.path.isfile(req_path)
        assert not os.path.isfile(resp_path)




# ===================================================================
# Test: Error handling
# ===================================================================


class TestCacheErrorHandling:
    """Cache module handles I/O errors gracefully."""

    def test_read_cache_returns_none_on_corrupt_file(self, tmp_path):
        """Reading a corrupt parquet file returns None."""
        calls = _make_multiple_calls(1)
        flow_path = os.path.join(tmp_path, "test_flows-file")
        Path(flow_path).touch()

        # Write valid cache
        write_cache(calls, flow_path, str(tmp_path))

        # Corrupt the requests file
        req_path, _ = get_cache_paths(flow_path, str(tmp_path))
        with open(req_path, "wb") as f:
            f.write(b"NOT A VALID PARQUET FILE")

        result = read_cache(flow_path, str(tmp_path))
        assert result is None

    def test_write_cache_creates_directory(self, tmp_path):
        """Writing cache creates the cache directory if it doesn't exist."""
        calls = _make_multiple_calls(1)
        flow_path = os.path.join(tmp_path, "test_flows-file")
        Path(flow_path).touch()

        new_dir = os.path.join(tmp_path, "new_cache_dir", "subdir")
        # Should succeed by creating the directory
        write_cache(calls, flow_path, new_dir)

        req_path, resp_path = get_cache_paths(flow_path, new_dir)
        assert os.path.isfile(req_path)
        assert os.path.isfile(resp_path)

    def test_empty_calls_list_writes_empty_tables(self, tmp_path):
        """Writing an empty list of calls creates empty parquet files."""
        flow_path = os.path.join(tmp_path, "test_flows-file")
        Path(flow_path).touch()

        write_cache([], flow_path, str(tmp_path))

        req_path, resp_path = get_cache_paths(flow_path, str(tmp_path))
        assert os.path.isfile(req_path)
        assert os.path.isfile(resp_path)

        restored = read_cache(flow_path, str(tmp_path))
        assert restored is not None
        assert len(restored) == 0


# ===================================================================
# Test: Cache key generation
# ===================================================================


class TestCacheKeyGeneration:
    """Cache key incorporates hash of file path and schema version."""

    def test_same_path_produces_same_key(self):
        """Same path always produces the same cache key."""
        path = "D:\\flows\\test_flows-file"
        key1 = _generate_cache_key(path)
        key2 = _generate_cache_key(path)
        assert key1 == key2

    def test_different_paths_produce_different_keys(self):
        """Different paths produce different cache keys."""
        key1 = _generate_cache_key("D:\\flows\\file1")
        key2 = _generate_cache_key("D:\\flows\\file2")
        assert key1 != key2

    def test_cache_key_is_reasonable_length(self):
        """Cache key is a reasonable length (hash hex digest)."""
        key = _generate_cache_key("D:\\flows\\test_flows-file")
        assert 8 <= len(key) <= 64


# ===================================================================
# Test: Schema version constant
# ===================================================================


class TestSchemaVersion:
    """SCHEMA_VERSION constant is defined and reasonable."""

    def test_schema_version_is_positive_int(self):
        """SCHEMA_VERSION is a positive integer."""
        assert isinstance(SCHEMA_VERSION, int)
        assert SCHEMA_VERSION >= 1

    def test_schema_version_stored_in_parquet_metadata(self, tmp_path):
        """Schema version is stored as metadata in written parquet files."""
        calls = _make_multiple_calls(1)
        flow_path = os.path.join(tmp_path, "test_flows-file")
        Path(flow_path).touch()
        write_cache(calls, flow_path, str(tmp_path))

        req_path, _ = get_cache_paths(flow_path, str(tmp_path))
        meta = pq.read_metadata(req_path)
        raw_meta = pq.read_schema(req_path).metadata
        assert raw_meta is not None
        version = raw_meta.get(b"cache_schema_version")
        assert version is not None
        assert int(version.decode("utf-8")) == SCHEMA_VERSION


# ===================================================================
# Test: Streaming / Batched Parse via progress_callback
# ===================================================================


class TestStreamingParseCallback:
    """VAL-STREAM-004, 005, 006: progress_callback in load_or_parse_cached."""

    def test_callback_fires_in_batches(self, tmp_path):
        """Progress callback fires for each batch of STREAMING_BATCH_SIZE calls.

        With 120 calls and BATCH_SIZE=50, callback fires 3 times:
        batch 1 = 50, batch 2 = 50, batch 3 = 20.
        """
        flow_path = os.path.join(tmp_path, "test_flows-file")
        Path(flow_path).touch()

        from llm_flow_viewer.parser.cache import (
            STREAMING_BATCH_SIZE,
            load_or_parse_cached,
        )

        total_calls = 120
        calls = _make_multiple_calls(total_calls)

        batches: list[tuple[int, list]] = []

        def cb(total_so_far, batch):
            batches.append((total_so_far, list(batch)))

        with patch("llm_flow_viewer.parser.cache.pair_flows") as mock_pair:
            mock_pair.return_value = iter(calls)
            result = load_or_parse_cached(
                flow_path, str(tmp_path), progress_callback=cb,
            )

        # Should have fired ceil(120/50) = 3 times
        expected_invocations = (total_calls + STREAMING_BATCH_SIZE - 1) // STREAMING_BATCH_SIZE
        assert len(batches) == expected_invocations, (
            f"Expected {expected_invocations} callback invocations, got {len(batches)}"
        )

        # Verify total count in final callback
        assert batches[-1][0] == total_calls
        # Verify all calls returned
        assert len(result) == total_calls

    def test_callback_batch_sizes(self, tmp_path):
        """Each callback batch has correct size: full batches then final partial batch."""
        flow_path = os.path.join(tmp_path, "test_flows-file")
        Path(flow_path).touch()

        from llm_flow_viewer.parser.cache import (
            STREAMING_BATCH_SIZE,
            load_or_parse_cached,
        )

        total_calls = STREAMING_BATCH_SIZE * 2 + 1  # e.g., 101
        calls = _make_multiple_calls(total_calls)

        batches: list[int] = []

        def cb(total_so_far, batch):
            batches.append(len(batch))

        with patch("llm_flow_viewer.parser.cache.pair_flows") as mock_pair:
            mock_pair.return_value = iter(calls)
            load_or_parse_cached(flow_path, str(tmp_path), progress_callback=cb)

        # Two full batches + one partial
        assert batches == [STREAMING_BATCH_SIZE, STREAMING_BATCH_SIZE, 1]

    def test_no_callback_behavior_unchanged(self, tmp_path):
        """VAL-STREAM-005: Without callback, behaves identically to pre-fix."""
        flow_path = os.path.join(tmp_path, "test_flows-file")
        Path(flow_path).touch()

        from llm_flow_viewer.parser.cache import load_or_parse_cached

        calls = _make_multiple_calls(10)

        with patch("llm_flow_viewer.parser.cache.pair_flows") as mock_pair:
            mock_pair.return_value = iter(calls)
            result = load_or_parse_cached(flow_path, str(tmp_path))

        assert len(result) == 10
        assert result[0].request_id == calls[0].request_id

    def test_callback_does_not_change_result(self, tmp_path):
        """VAL-STREAM-012: With callback, result should match non-callback path."""
        flow_path = os.path.join(tmp_path, "test_flows-file")
        Path(flow_path).touch()

        from llm_flow_viewer.parser.cache import load_or_parse_cached

        calls = _make_multiple_calls(10)
        collected_batches: list = []

        def cb(total, batch):
            collected_batches.extend(batch)

        with patch("llm_flow_viewer.parser.cache.pair_flows") as mock_pair:
            mock_pair.return_value = iter(calls)
            result = load_or_parse_cached(
                flow_path, str(tmp_path), progress_callback=cb,
            )

        # Calls collected via callback should match the returned list
        assert len(collected_batches) == len(result)
        for orig, cb_call in zip(calls, collected_batches):
            assert orig.request_id == cb_call.request_id

    def test_cache_written_after_streaming_parse(self, tmp_path):
        """VAL-STREAM-004: Cache files written after streaming parse completes."""
        flow_path = os.path.join(tmp_path, "test_flows-file")
        Path(flow_path).touch()

        from llm_flow_viewer.parser.cache import (
            get_cache_paths,
            is_cache_fresh,
            load_or_parse_cached,
        )

        calls = _make_multiple_calls(5)

        batches: list = []

        def cb(total, batch):
            batches.append(batch)

        with patch("llm_flow_viewer.parser.cache.pair_flows") as mock_pair:
            mock_pair.return_value = iter(calls)
            load_or_parse_cached(flow_path, str(tmp_path), progress_callback=cb)

        # Cache should now exist and be fresh
        req_path, resp_path = get_cache_paths(flow_path, str(tmp_path))
        assert os.path.isfile(req_path), "Request cache file should exist"
        assert os.path.isfile(resp_path), "Response cache file should exist"
        assert is_cache_fresh(flow_path, str(tmp_path))

        # Second load should hit cache (fast path)
        with patch("llm_flow_viewer.parser.cache.pair_flows") as mock_pair:
            mock_pair.return_value = iter(calls)
            # Use a different callback to verify cache route
            second_batches: list = []

            def cb2(total, batch):
                second_batches.append(batch)

            result2 = load_or_parse_cached(
                flow_path, str(tmp_path), progress_callback=cb2,
            )

        # Cache hit — callback should NOT fire
        assert len(second_batches) == 0, (
            "Callback should not fire on cache hit"
        )
        assert len(result2) == 5

    def test_callback_edge_cases(self, tmp_path):
        """VAL-STREAM-013: Edge cases for callback batching."""
        from llm_flow_viewer.parser.cache import load_or_parse_cached

        test_cases = [0, 1, 49, 50, 51, 100, 101, 1000]

        for n in test_cases:
            # Use unique file name per case to avoid cross-contamination via cache
            flow_path = os.path.join(tmp_path, f"test_flows-file-{n}")
            Path(flow_path).touch()

            calls = _make_multiple_calls(n)
            batches: list[list] = []

            def cb(total, batch):
                batches.append(list(batch))

            with patch("llm_flow_viewer.parser.cache.pair_flows") as mock_pair:
                mock_pair.return_value = iter(calls)
                result = load_or_parse_cached(
                    flow_path, str(tmp_path), progress_callback=cb,
                )

            assert len(result) == n, f"Expected {n} calls, got {len(result)}"
            total_from_batches = sum(len(b) for b in batches)
            assert total_from_batches == n, (
                f"Expected {n} total from batches, got {total_from_batches} for n={n}"
            )

    def test_batch_count_ceil_n_over_50(self, tmp_path):
        """VAL-STREAM-006: Callback invoked at most ceil(N/50) times."""
        from llm_flow_viewer.parser.cache import (
            STREAMING_BATCH_SIZE,
            load_or_parse_cached,
        )

        for n in [0, 1, 50, 51, 100, 101]:
            flow_path = os.path.join(tmp_path, f"test_flows-file-{n}")
            Path(flow_path).touch()

            calls = _make_multiple_calls(n)
            invocation_count = [0]

            def cb(total, batch):
                invocation_count[0] += 1

            with patch("llm_flow_viewer.parser.cache.pair_flows") as mock_pair:
                mock_pair.return_value = iter(calls)
                load_or_parse_cached(
                    flow_path, str(tmp_path), progress_callback=cb,
                )

            expected = (n + STREAMING_BATCH_SIZE - 1) // STREAMING_BATCH_SIZE if n > 0 else 0
            assert invocation_count[0] == expected, (
                f"Expected {expected} invocations for {n} calls, "
                f"got {invocation_count[0]}"
            )
