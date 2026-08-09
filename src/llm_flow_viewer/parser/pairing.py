"""Request/response pairing and timing extraction.

Takes parsed HTTP flows and pairs each :class:`ParsedRequest` with its
corresponding :class:`ParsedResponse`, generating a unique ULID per pair
and extracting timing information from the flow metadata.

Public API:
    - :func:`generate_request_id` — create a new ULID string
    - :func:`pair_flow_to_llm_call` — pair a single filtered flow
    - :func:`pair_flows` — process flow files and yield :class:`LLMCall` objects
"""

from __future__ import annotations

import logging
from typing import Generator, List, Optional

import ulid
from mitmproxy import http

from llm_flow_viewer.parser.models import (
    ConnectionTiming,
    LLMCall,
    ParsedRequest,
    ParsedResponse,
    Timing,
    TokenUsage,
)
from llm_flow_viewer.parser.reader import (
    expand_flow_files,
    is_llm_call_flow,
    open_flow_file,
)
from llm_flow_viewer.parser.request import parse_request
from llm_flow_viewer.parser.response import parse_response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ULID generation
# ---------------------------------------------------------------------------


def generate_request_id() -> str:
    """Generate a unique ULID string for identifying a request-response pair.

    Returns:
        A 26-character Crockford base32 ULID string.
    """
    return str(ulid.ULID())


# ---------------------------------------------------------------------------
# Single-flow pairing
# ---------------------------------------------------------------------------


def pair_flow_to_llm_call(
    flow: http.HTTPFlow,
) -> Optional[LLMCall]:
    """Pair a single HTTP flow into an :class:`LLMCall`.

    The flow must already pass the :func:`is_llm_call_flow` filter (i.e., it
    represents an LLM API call with both request and response content).
    This function:

    1. Generates a ULID for the pair
    2. Parses the request body into a :class:`ParsedRequest`
    3. Parses the response body into a :class:`ParsedResponse`
    4. Extracts timing from flow metadata into a :class:`Timing`
    5. Maps token usage from the response into a :class:`TokenUsage`
    6. Assembles everything into an :class:`LLMCall`

    Args:
        flow: An HTTPFlow that has passed :func:`is_llm_call_flow`.

    Returns:
        An :class:`LLMCall` if both request and response were parsed
        successfully, or ``None`` if either parsing step failed.
    """
    # --- Pre-checks -------------------------------------------------------
    if flow.response is None or not flow.response.content:
        logger.warning(
            "Skipping flow to %s%s — response is missing or empty",
            flow.request.host,
            flow.request.path,
        )
        return None

    # --- ULID -------------------------------------------------------------
    request_id = generate_request_id()

    # --- Request parsing --------------------------------------------------
    parsed_req = parse_request(flow.request.content)

    if parsed_req is None:
        logger.warning(
            "Skipping flow to %s%s — request content empty or unparseable",
            flow.request.host,
            flow.request.path,
        )
        return None

    parsed_req.request_id = request_id
    parsed_req.timestamp_start = flow.request.timestamp_start
    parsed_req.timestamp_end = flow.request.timestamp_end

    # --- Response parsing -------------------------------------------------
    parsed_resp = parse_response(
        flow.response.content,
        status_code=flow.response.status_code,
    )
    parsed_resp.request_id = request_id
    parsed_resp.timestamp_start = flow.response.timestamp_start
    parsed_resp.timestamp_end = flow.response.timestamp_end

    # --- Timing -----------------------------------------------------------
    timing = Timing(
        request_start=flow.request.timestamp_start,
        request_end=flow.request.timestamp_end,
        response_start=flow.response.timestamp_start,
        response_end=flow.response.timestamp_end,
    )

    # --- Connection timing (from server_conn) -----------------------------
    conn_timing = None
    if flow.server_conn is not None:
        server_conn = flow.server_conn
        conn_timing = ConnectionTiming(
            conn_id=server_conn.id or "",
            timestamp_start=getattr(server_conn, "timestamp_start", None),
            timestamp_tls_setup=getattr(server_conn, "timestamp_tls_setup", None),
            timestamp_end=getattr(server_conn, "timestamp_end", None),
        )

    # --- Token usage ------------------------------------------------------
    token_usage = TokenUsage(
        prompt_tokens=parsed_resp.input_tokens,
        completion_tokens=parsed_resp.output_tokens,
        total_tokens=(
            (parsed_resp.input_tokens or 0) + (parsed_resp.output_tokens or 0)
            if parsed_resp.input_tokens is not None or parsed_resp.output_tokens is not None
            else None
        ),
    )

    # --- Assemble ---------------------------------------------------------
    return LLMCall(
        request_id=request_id,
        request=parsed_req,
        response=parsed_resp,
        timing=timing,
        token_usage=token_usage,
        connection_timing=conn_timing,
    )


# ---------------------------------------------------------------------------
# Multi-file pairing
# ---------------------------------------------------------------------------


def pair_flows(
    flow_file_paths: List[str],
    error_collector: List[str] | None = None,
) -> Generator[LLMCall, None, None]:
    """Process flow files and yield paired :class:`LLMCall` objects.

    This is the top-level entry point for parsing mitmproxy flow files into
    :class:`LLMCall` objects. It:

    1. Expands file paths and glob patterns into existing file paths
    2. For each file, streams flows using :func:`FlowReader.stream`
    3. Filters for LLM API calls (POST to DeepSeek Anthropic endpoint)
    4. Pairs each request with its response via :func:`pair_flow_to_llm_call`
    5. Yields :class:`LLMCall` objects in chronological order

    Flows where request or response parsing fails are silently skipped with
    a warning log message.

    Args:
        flow_file_paths: List of file paths or glob patterns to process.
        error_collector: If provided, flow-level errors (e.g.
            ``FlowReadException``, unexpected processing errors) are
            appended to this list so the caller can display them.

    Yields:
        :class:`LLMCall` objects, one per successfully paired HTTP flow.
    """
    files = expand_flow_files(flow_file_paths)
    if not files:
        logger.warning("No flow files found matching patterns: %s", flow_file_paths)
        return

    for filepath in files:
        try:
            for flow in open_flow_file(filepath, error_collector=error_collector):
                if not is_llm_call_flow(flow):
                    continue

                call = pair_flow_to_llm_call(flow)
                if call is not None:
                    yield call
        except FileNotFoundError:
            msg = f"Flow file not found during pairing: {filepath}"
            logger.warning(msg)
            if error_collector is not None:
                error_collector.append(msg)
            raise
        except Exception as e:
            msg = f"Unexpected error processing {filepath}: {e}"
            logger.warning(msg)
            if error_collector is not None:
                error_collector.append(msg)
