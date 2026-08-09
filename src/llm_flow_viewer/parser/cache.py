"""Parquet cache module for LLM call data.

Provides functions to:
- Serialise lists of :class:`LLMCall` to separate Parquet files for requests
  and responses (using pyarrow)
- Deserialise Parquet files back into :class:`LLMCall` objects
- Check cache freshness based on source file modification times
- Automatically load from cache or re-parse based on freshness
- Display progress when processing many flows

Cache file naming incorporates a hash of the source file path and a schema
version to avoid stale-format caches.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from llm_flow_viewer.parser.models import (
    LLMCall,
    ParsedRequest,
    ParsedResponse,
    Timing,
    TokenUsage,
    ToolUse,
)
from llm_flow_viewer.parser.pairing import pair_flows
from llm_flow_viewer.parser.reader import expand_flow_files

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1
"""Current schema version for cache files.

Increment this when the Parquet schema changes to invalidate old cache files.
"""

STREAMING_BATCH_SIZE = 50
"""Number of LLMCalls to accumulate before firing the progress_callback.

Used by the streaming/batched parse path in :func:`load_or_parse_cached`.
"""

# ---------------------------------------------------------------------------
# Cache key / path helpers
# ---------------------------------------------------------------------------


def _generate_cache_key(flow_file_path: str) -> str:
    """Generate a deterministic cache key from a flow file path.

    Uses SHA-256 of the absolute, normalised file path.  The first 16 hex
    characters are used as the key to keep filenames manageable.

    Args:
        flow_file_path: Absolute or relative path to the flow file.

    Returns:
        A 16-character hex string uniquely derived from the file path.
    """
    abs_path = os.path.abspath(flow_file_path)
    normalized = os.path.normpath(abs_path)
    raw = normalized.lower()  # Case-insensitive on Windows
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _get_flow_basename(flow_file_path: str) -> str:
    """Get the basename of a flow file without directory or extension.

    Args:
        flow_file_path: Path to a flow file (may or may not have an extension).

    Returns:
        The basename without directory and without file extension.
    """
    base = os.path.basename(flow_file_path)
    # Strip any file extension if present (e.g., .flow, .mitm)
    root, ext = os.path.splitext(base)
    if ext:
        return root
    return base


def _resolve_cache_dir(
    flow_file_path: str,
    cache_dir: str | None = None,
) -> str:
    """Resolve the cache directory, defaulting to the flow file's directory.

    Args:
        flow_file_path: Path to the source flow file.
        cache_dir: Optional explicit cache directory.

    Returns:
        The cache directory path.
    """
    if cache_dir is not None:
        return os.path.abspath(cache_dir)
    return os.path.dirname(os.path.abspath(flow_file_path))


def get_cache_paths(
    flow_file_path: str,
    cache_dir: str | None = None,
) -> Tuple[str, str]:
    """Get the full paths for request and response Parquet cache files.

    The filenames incorporate a hash of the source file path and the
    schema version, e.g.::

        <cache_dir>/a1b2c3d4e5f6_<flow_basename>_requests.parquet
        <cache_dir>/a1b2c3d4e5f6_<flow_basename>_responses.parquet

    Args:
        flow_file_path: Path to the source flow file.
        cache_dir: Directory for cache files (defaults to flow file's dir).

    Returns:
        A tuple of (requests_parquet_path, responses_parquet_path).
    """
    resolved = _resolve_cache_dir(flow_file_path, cache_dir)
    key = _generate_cache_key(flow_file_path)
    basename = _get_flow_basename(flow_file_path)

    requests_path = os.path.join(resolved, f"{key}_{basename}_requests.parquet")
    responses_path = os.path.join(resolved, f"{key}_{basename}_responses.parquet")

    return requests_path, responses_path


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def _make_metadata() -> Dict[str, str]:
    """Create schema metadata dict to embed in Parquet files.

    Returns:
        Dict of string key-value pairs for Parquet schema metadata.
    """
    return {
        "cache_schema_version": str(SCHEMA_VERSION),
    }


def _check_metadata(schema: pa.Schema) -> bool:
    """Check that the schema metadata matches the current schema version.

    Args:
        schema: A pyarrow Schema (possibly with metadata).

    Returns:
        True if the metadata contains the correct schema version, else False.
    """
    if schema.metadata is None:
        logger.warning("Cache file missing schema metadata — assuming stale")
        return False

    raw = schema.metadata.get(b"cache_schema_version")
    if raw is None:
        logger.warning("Cache file missing cache_schema_version — assuming stale")
        return False

    try:
        version = int(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        logger.warning("Cache file has unparseable schema version — assuming stale")
        return False

    if version != SCHEMA_VERSION:
        logger.info(
            "Cache file has schema version %d, expected %d — treating as stale",
            version,
            SCHEMA_VERSION,
        )
        return False

    return True


# ---------------------------------------------------------------------------
# Serialisation: LLMCall list → pyarrow Tables
# ---------------------------------------------------------------------------


def llmcalls_to_requests_table(calls: List[LLMCall]) -> pa.Table:
    """Convert a list of LLMCall objects to a pyarrow Table for requests.

    Complex fields (messages, tools, system, output_config, thinking) are
    serialised as JSON strings.

    Args:
        calls: List of LLMCall objects.

    Returns:
        A pyarrow Table with one row per LLMCall, containing request data.
    """
    rows = []
    for call in calls:
        req = call.request
        if req is None:
            rows.append({
                "request_id": call.request_id,
                "model": "",
                "max_tokens": 0,
                "messages": "[]",
                "tools": "[]",
                "system": "[]",
                "output_config": None,
                "thinking": None,
                "stream": False,
                "timestamp_start": None,
                "timestamp_end": None,
            })
        else:
            rows.append({
                "request_id": req.request_id or call.request_id,
                "model": req.model,
                "max_tokens": req.max_tokens,
                "messages": json.dumps(req.messages, ensure_ascii=False),
                "tools": json.dumps(req.tools, ensure_ascii=False),
                "system": json.dumps(req.system, ensure_ascii=False),
                "output_config": json.dumps(req.output_config)
                if req.output_config is not None else None,
                "thinking": json.dumps(req.thinking)
                if req.thinking is not None else None,
                "stream": req.stream,
                "timestamp_start": req.timestamp_start,
                "timestamp_end": req.timestamp_end,
            })

    arrays = {
        "request_id": pa.array([r["request_id"] for r in rows], type=pa.utf8()),
        "model": pa.array([r["model"] for r in rows], type=pa.utf8()),
        "max_tokens": pa.array([r["max_tokens"] for r in rows], type=pa.int64()),
        "messages": pa.array([r["messages"] for r in rows], type=pa.utf8()),
        "tools": pa.array([r["tools"] for r in rows], type=pa.utf8()),
        "system": pa.array([r["system"] for r in rows], type=pa.utf8()),
        "output_config": pa.array(
            [r["output_config"] for r in rows], type=pa.utf8(),
        ),
        "thinking": pa.array(
            [r["thinking"] for r in rows], type=pa.utf8(),
        ),
        "stream": pa.array([r["stream"] for r in rows], type=pa.bool_()),
        "timestamp_start": pa.array(
            [r["timestamp_start"] for r in rows], type=pa.float64(),
        ),
        "timestamp_end": pa.array(
            [r["timestamp_end"] for r in rows], type=pa.float64(),
        ),
    }

    schema = pa.schema(
        [
            pa.field("request_id", pa.utf8(), nullable=False),
            pa.field("model", pa.utf8(), nullable=False),
            pa.field("max_tokens", pa.int64(), nullable=False),
            pa.field("messages", pa.utf8(), nullable=False),
            pa.field("tools", pa.utf8(), nullable=False),
            pa.field("system", pa.utf8(), nullable=False),
            pa.field("output_config", pa.utf8(), nullable=True),
            pa.field("thinking", pa.utf8(), nullable=True),
            pa.field("stream", pa.bool_(), nullable=False),
            pa.field("timestamp_start", pa.float64(), nullable=True),
            pa.field("timestamp_end", pa.float64(), nullable=True),
        ],
        metadata=_make_metadata(),
    )

    return pa.table(arrays, schema=schema)


def llmcalls_to_responses_table(calls: List[LLMCall]) -> pa.Table:
    """Convert a list of LLMCall objects to a pyarrow Table for responses.

    Complex fields (tool_uses) are serialised as JSON strings.

    Args:
        calls: List of LLMCall objects.

    Returns:
        A pyarrow Table with one row per LLMCall, containing response data.
    """
    rows = []
    for call in calls:
        resp = call.response
        if resp is None:
            rows.append({
                "request_id": call.request_id,
                "thinking": "",
                "text": "",
                "tool_uses": "[]",
                "message_id": None,
                "model": None,
                "role": None,
                "input_tokens": None,
                "output_tokens": None,
                "cache_creation_input_tokens": None,
                "cache_read_input_tokens": None,
                "service_tier": None,
                "stop_reason": None,
                "stop_sequence": None,
                "status_code": 0,
                "error_message": "",
                "timestamp_start": None,
                "timestamp_end": None,
            })
        else:
            # Serialise tool_uses to JSON
            tool_uses_json = json.dumps(
                [asdict(tu) for tu in resp.tool_uses],
                ensure_ascii=False,
            )
            rows.append({
                "request_id": resp.request_id or call.request_id,
                "thinking": resp.thinking,
                "text": resp.text,
                "tool_uses": tool_uses_json,
                "message_id": resp.message_id,
                "model": resp.model,
                "role": resp.role,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "cache_creation_input_tokens": resp.cache_creation_input_tokens,
                "cache_read_input_tokens": resp.cache_read_input_tokens,
                "service_tier": resp.service_tier,
                "stop_reason": resp.stop_reason,
                "stop_sequence": resp.stop_sequence,
                "status_code": resp.status_code,
                "error_message": resp.error_message,
                "timestamp_start": resp.timestamp_start,
                "timestamp_end": resp.timestamp_end,
            })

    schema = pa.schema(
        [
            pa.field("request_id", pa.utf8(), nullable=False),
            pa.field("thinking", pa.utf8(), nullable=False),
            pa.field("text", pa.utf8(), nullable=False),
            pa.field("tool_uses", pa.utf8(), nullable=False),
            pa.field("message_id", pa.utf8(), nullable=True),
            pa.field("model", pa.utf8(), nullable=True),
            pa.field("role", pa.utf8(), nullable=True),
            pa.field("input_tokens", pa.int64(), nullable=True),
            pa.field("output_tokens", pa.int64(), nullable=True),
            pa.field("cache_creation_input_tokens", pa.int64(), nullable=True),
            pa.field("cache_read_input_tokens", pa.int64(), nullable=True),
            pa.field("service_tier", pa.utf8(), nullable=True),
            pa.field("stop_reason", pa.utf8(), nullable=True),
            pa.field("stop_sequence", pa.utf8(), nullable=True),
            pa.field("status_code", pa.int64(), nullable=False),
            pa.field("error_message", pa.utf8(), nullable=False),
            pa.field("timestamp_start", pa.float64(), nullable=True),
            pa.field("timestamp_end", pa.float64(), nullable=True),
        ],
        metadata=_make_metadata(),
    )

    arrays = {
        col.name: pa.array([r[col.name] for r in rows], type=col.type)
        for col in schema
    }

    return pa.table(arrays, schema=schema)


# ---------------------------------------------------------------------------
# Deserialisation: pyarrow Tables → LLMCall list
# ---------------------------------------------------------------------------


def _safe_json_loads(value: Any) -> Any:
    """Safely parse a JSON string, returning a default for null/empty values.

    Args:
        value: A JSON string or None.

    Returns:
        Parsed Python object, or empty list/dict as appropriate.
    """
    if value is None or value == "":
        return [] if isinstance(value, (str, type(None))) else value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse JSON value: %s", str(value)[:200])
        return []


def requests_table_to_list(
    req_table: pa.Table,
    resp_table: pa.Table,
) -> List[LLMCall]:
    """Reconstruct a list of LLMCall objects from request and response Tables.

    The two tables are joined by ``request_id``.  Timing and TokenUsage are
    reconstructed from the timestamp and token fields in the request/response
    data.

    This implementation calls ``table.to_pydict()`` **once** per table (O(n))
    instead of calling ``table.slice(i, 1).to_pydict()`` in a loop (O(n^2)).

    Args:
        req_table: A pyarrow Table with request data.
        resp_table: A pyarrow Table with response data.

    Returns:
        A list of reconstructed LLMCall objects.
    """
    # Convert both tables to dicts in a single call (O(n) per table).
    req_dict = req_table.to_pydict()
    resp_dict = resp_table.to_pydict()

    # Build lookup dicts keyed by request_id
    reqs_by_id: Dict[str, Dict[str, Any]] = {}
    for i in range(len(req_table)):
        rid = str(req_dict["request_id"][i])
        reqs_by_id[rid] = {k: v[i] for k, v in req_dict.items()}

    resps_by_id: Dict[str, Dict[str, Any]] = {}
    for i in range(len(resp_table)):
        rid = str(resp_dict["request_id"][i])
        resps_by_id[rid] = {k: v[i] for k, v in resp_dict.items()}

    # Collect all request_ids in order (from req_table to preserve order)
    all_ids = [str(req_dict["request_id"][i]) for i in range(len(req_table))]

    calls: List[LLMCall] = []
    for rid in all_ids:
        req_row = reqs_by_id.get(rid, {})
        resp_row = resps_by_id.get(rid, {})

        # Reconstruct ParsedRequest
        req = ParsedRequest(
            request_id=rid,
            model=req_row.get("model", ""),
            max_tokens=req_row.get("max_tokens", 0) or 0,
            messages=_safe_json_loads(req_row.get("messages")),
            tools=_safe_json_loads(req_row.get("tools")),
            system=_safe_json_loads(req_row.get("system")),
            output_config=_safe_json_loads(req_row.get("output_config"))
            if req_row.get("output_config") else None,
            thinking=_safe_json_loads(req_row.get("thinking"))
            if req_row.get("thinking") else None,
            stream=bool(req_row.get("stream", False)),
            timestamp_start=req_row.get("timestamp_start"),
            timestamp_end=req_row.get("timestamp_end"),
        )

        # Reconstruct ParsedResponse
        tool_uses_data = _safe_json_loads(resp_row.get("tool_uses"))
        tool_uses = [
            ToolUse(
                name=tu.get("name", ""),
                id=tu.get("id", ""),
                input=tu.get("input", {}),
            )
            for tu in tool_uses_data
        ]

        resp = ParsedResponse(
            request_id=rid,
            thinking=resp_row.get("thinking", "") or "",
            text=resp_row.get("text", "") or "",
            tool_uses=tool_uses,
            message_id=resp_row.get("message_id"),
            model=resp_row.get("model"),
            role=resp_row.get("role"),
            input_tokens=resp_row.get("input_tokens"),
            output_tokens=resp_row.get("output_tokens"),
            cache_creation_input_tokens=resp_row.get("cache_creation_input_tokens"),
            cache_read_input_tokens=resp_row.get("cache_read_input_tokens"),
            service_tier=resp_row.get("service_tier"),
            stop_reason=resp_row.get("stop_reason"),
            stop_sequence=resp_row.get("stop_sequence"),
            status_code=resp_row.get("status_code", 200) or 200,
            error_message=resp_row.get("error_message", "") or "",
            timestamp_start=resp_row.get("timestamp_start"),
            timestamp_end=resp_row.get("timestamp_end"),
        )

        # Reconstruct Timing from timestamps
        timing = Timing(
            request_start=req.timestamp_start,
            request_end=req.timestamp_end,
            response_start=resp.timestamp_start,
            response_end=resp.timestamp_end,
        )

        # Reconstruct TokenUsage from response tokens
        token_usage = TokenUsage(
            prompt_tokens=resp.input_tokens,
            completion_tokens=resp.output_tokens,
            total_tokens=(
                (resp.input_tokens or 0) + (resp.output_tokens or 0)
                if resp.input_tokens is not None or resp.output_tokens is not None
                else None
            ),
        )

        calls.append(LLMCall(
            request_id=rid,
            request=req,
            response=resp,
            timing=timing,
            token_usage=token_usage,
        ))

    return calls


# ---------------------------------------------------------------------------
# File-level cache operations
# ---------------------------------------------------------------------------


def write_cache(
    calls: List[LLMCall],
    flow_file_path: str,
    cache_dir: str | None = None,
) -> None:
    """Write a list of LLMCall objects to Parquet cache files.

    Creates two Parquet files: one for requests and one for responses.
    The cache directory is created if it does not exist.

    Args:
        calls: List of LLMCall objects to cache.
        flow_file_path: Path to the source flow file (used for cache naming).
        cache_dir: Directory for cache files (defaults to flow file's dir).
    """
    req_path, resp_path = get_cache_paths(flow_file_path, cache_dir)

    # Ensure cache directory exists
    cache_parent = os.path.dirname(req_path)
    os.makedirs(cache_parent, exist_ok=True)

    # Convert and write
    req_table = llmcalls_to_requests_table(calls)
    resp_table = llmcalls_to_responses_table(calls)

    pq.write_table(req_table, req_path)
    pq.write_table(resp_table, resp_path)

    logger.info(
        "Wrote cache for %s: %d calls → %s, %s",
        flow_file_path,
        len(calls),
        os.path.basename(req_path),
        os.path.basename(resp_path),
    )


def read_cache(
    flow_file_path: str,
    cache_dir: str | None = None,
) -> List[LLMCall] | None:
    """Read LLMCall objects from Parquet cache files.

    Checks that both the requests and responses cache files exist and have
    matching schema versions.  Returns ``None`` if the cache is absent,
    incomplete, or has an incompatible schema version.

    Args:
        flow_file_path: Path to the source flow file (used for cache naming).
        cache_dir: Directory for cache files (defaults to flow file's dir).

    Returns:
        A list of LLMCall objects if the cache was read successfully,
        or ``None`` if the cache is unavailable or stale.
    """
    req_path, resp_path = get_cache_paths(flow_file_path, cache_dir)

    # Check both files exist
    if not os.path.isfile(req_path) or not os.path.isfile(resp_path):
        logger.debug("Cache files not found for %s", flow_file_path)
        return None

    try:
        # Check schema versions
        req_schema = pq.read_schema(req_path)
        resp_schema = pq.read_schema(resp_path)

        if not _check_metadata(req_schema) or not _check_metadata(resp_schema):
            return None

        # Read tables
        req_table = pq.read_table(req_path)
        resp_table = pq.read_table(resp_path)

        if len(req_table) != len(resp_table):
            logger.warning(
                "Cache file row count mismatch for %s: "
                "requests=%d, responses=%d — treating as stale",
                flow_file_path,
                len(req_table),
                len(resp_table),
            )
            return None

        calls = requests_table_to_list(req_table, resp_table)
        logger.info(
            "Loaded %d calls from cache for %s",
            len(calls),
            flow_file_path,
        )
        return calls

    except Exception as e:
        logger.warning("Failed to read cache for %s: %s", flow_file_path, e)
        return None


# ---------------------------------------------------------------------------
# Cache freshness
# ---------------------------------------------------------------------------


def is_cache_fresh(
    flow_file_path: str,
    cache_dir: str | None = None,
) -> bool:
    """Check whether the cache for a flow file is fresh (up-to-date).

    The cache is considered fresh when:
    - Both cache files exist
    - Both cache files have modification times **later than** the source file
    - Both cache files have matching schema versions

    Args:
        flow_file_path: Path to the source flow file.
        cache_dir: Directory for cache files (defaults to flow file's dir).

    Returns:
        True if the cache is fresh and can be used, False otherwise.
    """
    req_path, resp_path = get_cache_paths(flow_file_path, cache_dir)

    # Source must exist
    if not os.path.isfile(flow_file_path):
        logger.debug("Source file not found: %s", flow_file_path)
        return False

    # Both cache files must exist
    if not os.path.isfile(req_path) or not os.path.isfile(resp_path):
        return False

    # Check schema versions
    try:
        if not _check_metadata(pq.read_schema(req_path)):
            return False
        if not _check_metadata(pq.read_schema(resp_path)):
            return False
    except Exception:
        return False

    # Compare modification times: cache must be newer than source
    source_mtime = os.path.getmtime(flow_file_path)
    req_mtime = os.path.getmtime(req_path)
    resp_mtime = os.path.getmtime(resp_path)

    if req_mtime < source_mtime or resp_mtime < source_mtime:
        logger.info(
            "Cache stale for %s: source mtime=%.3f, "
            "requests mtime=%.3f, responses mtime=%.3f",
            flow_file_path,
            source_mtime,
            req_mtime,
            resp_mtime,
        )
        return False

    return True


# ---------------------------------------------------------------------------
# Progress-aware flow parsing
# ---------------------------------------------------------------------------

_PROGRESS_LOG_INTERVAL = 100
"""Log a progress message every N flows."""


def pair_flows_with_progress(
    flow_file_paths: List[str],
    desc: str | None = None,
    error_collector: List[str] | None = None,
) -> Generator[LLMCall, None, None]:
    """Wrapper around :func:`pair_flows` that logs periodic progress.

    Yields :class:`LLMCall` objects from the flow files, logging a progress
    message every ``_PROGRESS_LOG_INTERVAL`` flows.

    Args:
        flow_file_paths: List of file paths or glob patterns to process.
        desc: Optional description for the progress log (default: "Parsing flows").
        error_collector: If provided, flow-level errors are appended here.

    Yields:
        :class:`LLMCall` objects, one per parsed flow.
    """
    label = desc or "Parsing flows"
    count = 0
    for call in pair_flows(flow_file_paths, error_collector=error_collector):
        yield call
        count += 1
        if count % _PROGRESS_LOG_INTERVAL == 0:
            logger.info("%s: %d flows parsed", label, count)

    if count > 0:
        logger.info("%s: completed — %d total flows", label, count)


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------


def _parse_with_batch_callback(
    flow_file_paths: List[str],
    batch_size: int = STREAMING_BATCH_SIZE,
    error_collector: List[str] | None = None,
    progress_callback: Callable[[int, List[LLMCall]], None] | None = None,
) -> List[LLMCall]:
    """Parse flow files and fire *progress_callback* for each batch.

    Accumulates :class:`LLMCall` objects from :func:`pair_flows` and
    invokes ``progress_callback(total_so_far, batch)`` after every
    *batch_size* calls (and for the final partial batch).

    Args:
        flow_file_paths: List of file paths or glob patterns to process.
        batch_size: Number of calls per batch (default: ``STREAMING_BATCH_SIZE``).
        error_collector: If provided, flow-level errors are appended here.
        progress_callback: Optional callback fired for each batch.
            Receives ``(total_count, batch_list)``.

    Returns:
        The complete list of parsed :class:`LLMCall` objects.
    """
    all_calls: List[LLMCall] = []
    batch: List[LLMCall] = []

    for call in pair_flows(flow_file_paths, error_collector=error_collector):
        all_calls.append(call)
        batch.append(call)
        if len(batch) >= batch_size:
            if progress_callback is not None:
                progress_callback(len(all_calls), batch)
            batch = []

    # Final partial batch
    if batch and progress_callback is not None:
        progress_callback(len(all_calls), batch)

    if all_calls:
        logger.info(
            "Parsed %d total LLMCalls from %s",
            len(all_calls),
            flow_file_paths,
        )

    return all_calls


def load_or_parse_cached(
    flow_file_path: str,
    cache_dir: str | None = None,
    show_progress: bool = True,
    error_collector: List[str] | None = None,
    progress_callback: Callable[[int, List[LLMCall]], None] | None = None,
) -> List[LLMCall]:
    """Load LLMCall data for a flow file, using cache when available.

    If a fresh cache exists, load from cache (fast path).
    Otherwise, parse the flow file from scratch, write a new cache, and
    return the parsed data.

    When *progress_callback* is provided (streaming path), parsed calls
    are yielded in batches of ``STREAMING_BATCH_SIZE`` via the callback
    while still accumulating all calls for the final return value and
    cache write.  The non-streaming path (callback is ``None``) behaves
    identically to the original implementation.

    Args:
        flow_file_path: Path to the source mitmproxy flow dump file.
        cache_dir: Directory for cache files (defaults to flow file's dir).
        show_progress: Whether to log progress messages during parsing
            (only used when *progress_callback* is ``None``).
        error_collector: If provided, flow-level errors are appended here.
        progress_callback: Optional callback fired for each batch of
            parsed calls.  Receives ``(total_so_far, batch)``.

    Returns:
        A list of LLMCall objects parsed from the flow file.
    """
    # Fast path: load from cache
    if is_cache_fresh(flow_file_path, cache_dir):
        cached = read_cache(flow_file_path, cache_dir)
        if cached is not None:
            return cached
        logger.info(
            "Cache read returned None for %s despite freshness check — "
            "will re-parse",
            flow_file_path,
        )

    # Slow path: parse from scratch
    logger.info("Parsing flow file: %s", flow_file_path)

    if progress_callback is not None:
        # Streaming / batched path
        calls = _parse_with_batch_callback(
            [flow_file_path],
            batch_size=STREAMING_BATCH_SIZE,
            error_collector=error_collector,
            progress_callback=progress_callback,
        )
    elif show_progress:
        progress_fn = pair_flows_with_progress
        calls = list(progress_fn([flow_file_path], error_collector=error_collector))
    else:
        calls = list(pair_flows([flow_file_path], error_collector=error_collector))

    # Write cache (skip when zero calls parsed to avoid masking parse failures)
    if calls:
        write_cache(calls, flow_file_path, cache_dir)
    else:
        logger.warning(
            "No LLMCalls parsed from %s — skipping cache write",
            flow_file_path,
        )

    return calls
