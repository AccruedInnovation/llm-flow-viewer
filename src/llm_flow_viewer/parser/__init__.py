"""Parser module for LLM Flow Viewer.

Parses mitmproxy flow files into structured dataclasses for TUI display.
"""

from llm_flow_viewer.parser.cache import (
    SCHEMA_VERSION,
    get_cache_paths,
    is_cache_fresh,
    load_or_parse_cached,
    llmcalls_to_requests_table,
    llmcalls_to_responses_table,
    pair_flows_with_progress,
    read_cache,
    requests_table_to_list,
    write_cache,
)
from llm_flow_viewer.parser.models import (
    ConnectionTiming,
    LLMCall,
    ParsedRequest,
    ParsedResponse,
    Session,
    Timing,
    TokenUsage,
    ToolResult,
    ToolUse,
)
from llm_flow_viewer.parser.pairing import (
    generate_request_id,
    pair_flow_to_llm_call,
    pair_flows,
)
from llm_flow_viewer.parser.reader import (
    FlowFilterConfig,
    expand_flow_files,
    is_llm_call_flow,
    open_flow_file,
    parse_flows_from_file,
    read_flow_files,
)
from llm_flow_viewer.parser.request import parse_request
from llm_flow_viewer.parser.response import parse_response
from llm_flow_viewer.parser.session import flow_file_to_session

__all__ = [
    "FlowFilterConfig",
    "LLMCall",
    "ParsedRequest",
    "ParsedResponse",
    "SCHEMA_VERSION",
    "Session",
    "Timing",
    "TokenUsage",
    "ToolResult",
    "ToolUse",
    "expand_flow_files",
    "flow_file_to_session",
    "generate_request_id",
    "get_cache_paths",
    "is_cache_fresh",
    "is_llm_call_flow",
    "load_or_parse_cached",
    "llmcalls_to_requests_table",
    "llmcalls_to_responses_table",
    "open_flow_file",
    "pair_flow_to_llm_call",
    "pair_flows",
    "pair_flows_with_progress",
    "parse_flows_from_file",
    "parse_request",
    "parse_response",
    "read_cache",
    "read_flow_files",
    "requests_table_to_list",
    "write_cache",
]
