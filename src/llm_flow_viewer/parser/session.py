"""Session aggregation logic.

Takes a discovered flow file and produces a :class:`~llm_flow_viewer.parser.models.Session`
object containing all parsed :class:`~llm_flow_viewer.parser.models.LLMCall` objects,
ordered chronologically by request timestamp.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

from llm_flow_viewer.parser.cache import load_or_parse_cached
from llm_flow_viewer.parser.models import LLMCall, Session

logger = logging.getLogger(__name__)


def _discover_model(calls: List[LLMCall]) -> str:
    """Determine the primary model used in a list of calls.

    Returns the model from the first call that has a request with a model name,
    or an empty string if no model is found.

    Args:
        calls: List of LLMCall objects.

    Returns:
        The model name string (e.g. ``"deepseek-v4-flash"``).
    """
    for call in calls:
        if call.request and call.request.model:
            return call.request.model
    return ""


def flow_file_to_session(
    file_path: str,
    index: int,
    task_name: str,
    cache_dir: str | None = None,
    progress_callback: Callable[[int, List[LLMCall]], None] | None = None,
) -> Session:
    """Parse a flow file into a :class:`Session` object.

    Uses the Parquet cache if available (fast path), otherwise parses the raw
    mitmproxy flow file from scratch.

    When *progress_callback* is provided (streaming path), parsed calls are
    delivered in batches via the callback while parsing proceeds.  The
    non-streaming path (callback is ``None``) behaves identically to the
    original implementation.

    The returned session's calls are sorted chronologically by
    ``request.timestamp_start``.  Flow-level errors (e.g.
    ``FlowReadException`` for corrupt data) are captured in
    ``session.flow_errors``.

    Args:
        file_path: Path to the mitmproxy flow dump file.
        index: The session index number (e.g. ``1`` for session 01).
        task_name: The human-readable task name (e.g. ``"analyze_codebase"``).
        cache_dir: Optional explicit cache directory.  Defaults to the
            directory containing the flow file.
        progress_callback: Optional callback invoked for each batch of
            parsed :class:`LLMCall` objects.  Receives
            ``(total_so_far, batch)``.

    Returns:
        A :class:`Session` object containing the parsed calls and any
        flow read errors.
    """
    # Collect flow-level errors during raw parsing
    flow_errors: List[str] = []
    calls = load_or_parse_cached(
        file_path,
        cache_dir,
        error_collector=flow_errors,
        progress_callback=progress_callback,
    )

    # Sort chronologically by request timestamp
    calls.sort(key=_call_sort_key)

    model = _discover_model(calls)

    return Session(
        index=index,
        task_name=task_name,
        model=model,
        calls=calls,
        flow_errors=flow_errors,
    )


def _call_sort_key(call: LLMCall) -> float:
    """Return a sort key for chronological ordering of calls.

    Uses ``request.timestamp_start`` when available, falling back to 0.

    Args:
        call: An LLMCall object.

    Returns:
        A float timestamp suitable for sorting.
    """
    if call.request and call.request.timestamp_start is not None:
        return call.request.timestamp_start
    return 0.0
