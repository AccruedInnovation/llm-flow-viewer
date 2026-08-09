"""Flow file reader and endpoint filter.

Provides functions to open mitmproxy flow files, filter for LLM API calls,
and parse request bodies into ParsedRequest dataclasses.
"""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

from mitmproxy import http
from mitmproxy.exceptions import FlowReadException
from mitmproxy.io import FlowReader

from llm_flow_viewer.parser.models import ParsedRequest
from llm_flow_viewer.parser.request import parse_request

logger = logging.getLogger(__name__)


@dataclass
class FlowFilterConfig:
    """Configuration for filtering flows that represent LLM API calls.

    Attributes:
        allowed_hosts: Set of host names to match (e.g., api.deepseek.com).
        allowed_paths: Set of URL paths to match (e.g., /anthropic/v1/messages).
        allowed_methods: Set of HTTP methods to allow.
    """
    allowed_hosts: set = field(default_factory=lambda: {"api.deepseek.com"})
    allowed_paths: set = field(default_factory=lambda: {"/anthropic/v1/messages"})
    allowed_methods: set = field(default_factory=lambda: {"POST"})


def open_flow_file(
    filepath: str,
    error_collector: List[str] | None = None,
) -> Generator[http.HTTPFlow, None, None]:
    """Open a mitmproxy flow file and yield HTTPFlow objects.

    Args:
        filepath: Path to the mitmproxy binary flow dump file.
        error_collector: If provided, flow read errors (e.g.
            ``FlowReadException``) are appended to this list instead of
            silently swallowed.

    Yields:
        HTTPFlow objects from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    try:
        with open(filepath, "rb") as logfile:
            reader = FlowReader(logfile)
            for flow in reader.stream():
                yield flow
    except FileNotFoundError:
        raise
    except FlowReadException as e:
        msg = f"FlowReadException while reading {filepath}: {e}"
        logger.warning(msg)
        if error_collector is not None:
            error_collector.append(msg)
        return
    except Exception as e:
        msg = f"Unexpected error reading {filepath}: {e}"
        logger.warning(msg)
        if error_collector is not None:
            error_collector.append(msg)
        return


def is_llm_call_flow(
    flow: Any,
    config: FlowFilterConfig | None = None,
) -> bool:
    """Check whether a flow is an LLM API call matching filter criteria.

    A flow is considered an LLM call if:
    - It is an HTTPFlow instance
    - Its method is in allowed_methods (default: POST)
    - Its host is in allowed_hosts (default: api.deepseek.com)
    - Its path is in allowed_paths (default: /anthropic/v1/messages)
    - Its request content is not empty

    Args:
        flow: An object from mitmproxy flow reader (may be HTTPFlow or other).
        config: Filter configuration (uses defaults if None).

    Returns:
        True if the flow matches all LLM call criteria.
    """
    if not isinstance(flow, http.HTTPFlow):
        return False

    if config is None:
        config = FlowFilterConfig()

    request = flow.request

    if request.method not in config.allowed_methods:
        return False

    # Check host (strip potential port suffix)
    host = request.host
    if ":" in host:
        host = host.split(":")[0]
    if host not in config.allowed_hosts:
        return False

    if request.path not in config.allowed_paths:
        return False

    if not request.content:
        return False

    # Check response exists and has non-empty content
    if flow.response is None:
        return False
    if not flow.response.content:
        return False

    return True


def expand_flow_files(patterns: List[str]) -> List[str]:
    """Expand a list of file paths and glob patterns into existing file paths.

    Args:
        patterns: List of file paths or glob patterns to expand.

    Returns:
        Sorted list of unique, existing file paths matching the patterns.
    """
    files: List[str] = []
    seen: set = set()
    for pattern in patterns:
        for path in glob.glob(pattern):
            if os.path.isfile(path) and path not in seen:
                seen.add(path)
                files.append(path)
    return sorted(files)


def read_flow_files(
    file_paths: List[str],
    config: FlowFilterConfig | None = None,
    error_collector: List[str] | None = None,
) -> Generator[Tuple[str, http.HTTPFlow, ParsedRequest], None, None]:
    """Read multiple flow files, yielding flows from all matching files.

    This is the main entry point for processing one or more flow files.
    It supports both explicit file paths and glob patterns, and yields
    (filepath, flow, parsed_request) tuples from all matching files.

    Args:
        file_paths: List of file paths or glob patterns to read.
        config: Filter configuration (uses defaults if None).
        error_collector: If provided, flow read errors are appended here.

    Yields:
        Tuples of (filepath, HTTPFlow, ParsedRequest) for each matching flow.
    """
    if config is None:
        config = FlowFilterConfig()

    files = expand_flow_files(file_paths)
    if not files:
        logger.warning("No flow files found matching patterns: %s", file_paths)
        return

    for filepath in files:
        try:
            for flow, parsed in parse_flows_from_file(
                filepath, config, error_collector=error_collector,
            ):
                yield filepath, flow, parsed
        except FileNotFoundError:
            msg = f"Flow file not found during read: {filepath} — skipping"
            logger.warning(msg)
            if error_collector is not None:
                error_collector.append(msg)
        except Exception as e:
            msg = f"Unexpected error reading {filepath}: {e}"
            logger.warning(msg)
            if error_collector is not None:
                error_collector.append(msg)


def parse_flows_from_file(
    filepath: str,
    config: FlowFilterConfig | None = None,
    error_collector: List[str] | None = None,
) -> Generator[Tuple[http.HTTPFlow, ParsedRequest], None, None]:
    """Open a flow file, filter for LLM calls, and parse each request body.

    This is the main entry point for processing a single flow file. It:
    1. Opens the file and streams flows
    2. Filters for LLM API calls (POST to DeepSeek Anthropic endpoint)
    3. Parses each request body into a ParsedRequest

    Args:
        filepath: Path to the mitmproxy binary flow dump file.
        config: Filter configuration (uses defaults if None).
        error_collector: If provided, flow read errors are appended here.

    Yields:
        Tuples of (HTTPFlow, ParsedRequest) for each matching flow.
    """
    if config is None:
        config = FlowFilterConfig()

    for flow in open_flow_file(filepath, error_collector=error_collector):
        if not is_llm_call_flow(flow, config):
            continue

        parsed = parse_request(flow.request.content)
        if parsed is None:
            logger.warning(
                "Skipping flow to %s%s — request content empty or unparseable",
                flow.request.host,
                flow.request.path,
            )
            continue

        # Attach timestamps from the raw flow
        parsed.timestamp_start = flow.request.timestamp_start
        parsed.timestamp_end = flow.request.timestamp_end

        yield flow, parsed
