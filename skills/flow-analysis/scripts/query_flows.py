"""Helper CLI for one-shot parquet queries against LLM flow data.

Usage:
    python query_flows.py --flows-dir flows/ --list-sessions
    python query_flows.py --flows-dir flows/ --session 06 --metric tokens
    python query_flows.py --flows-dir flows/ --session 06 --metric timing
    python query_flows.py --flows-dir flows/ --session 06 --metric tools
    python query_flows.py --flows-dir flows/ --session 06 --metric cache
    python query_flows.py --flows-dir flows/ --session 06 --metric models
    python query_flows.py --flows-dir flows/ --session 06 --metric call-detail --request-id <ULID>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Optional


def find_parquet_files(flows_dir: str) -> list[tuple[str, str, str]]:
    """Discover (session_label, req_path, resp_path) tuples from flows dir."""
    result = []
    seen = set()
    for fname in sorted(os.listdir(flows_dir)):
        if not fname.endswith(".parquet"):
            continue
        # Parse out the session label from: <hash>_<label>_requests.parquet
        base = fname
        for suffix in ("_requests.parquet", "_responses.parquet"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        if base in seen:
            continue
        seen.add(base)

        req_path = os.path.join(flows_dir, f"{base}_requests.parquet")
        resp_path = os.path.join(flows_dir, f"{base}_responses.parquet")
        if os.path.isfile(req_path) and os.path.isfile(resp_path):
            # Find matching label in flow file names
            label = base.split("_", 1)[1] if "_" in base else base
            result.append((label, req_path, resp_path))
    return result


def _find_by_index(results: list[tuple[str, str, str]], index: str) -> Optional[tuple[str, str, str]]:
    """Match session by numeric index prefix (e.g. '06')."""
    idx_padded = index.zfill(2)
    for label, req_path, resp_path in results:
        if label.startswith(idx_padded) or label.startswith(index):
            return (label, req_path, resp_path)
    return None


def cmd_list_sessions(results: list[tuple[str, str, str]]) -> None:
    """Print all discovered sessions."""
    for label, req_path, resp_path in results:
        import pyarrow.parquet as pq
        try:
            table = pq.read_table(resp_path, columns=["request_id"])
            count = table.num_rows
        except Exception:
            count = "?"
        print(f"  {label}: {count} calls")
        print(f"    requests:  {req_path}")
        print(f"    responses: {resp_path}")


def cmd_tokens(label: str, req_path: str, resp_path: str) -> None:
    """Print token usage summary."""
    import pyarrow.parquet as pq

    resp = pq.read_table(resp_path, columns=["input_tokens", "output_tokens", "status_code"])
    total_in = 0
    total_out = 0
    errors = 0
    for i in range(len(resp)):
        sc = resp["status_code"][i].as_py()
        if sc != 200:
            errors += 1
            continue
        inp = resp["input_tokens"][i].as_py()
        out = resp["output_tokens"][i].as_py()
        total_in += inp or 0
        total_out += out or 0

    call_count = len(resp) - errors
    print(f"Session: {label}")
    print(f"  Calls:           {call_count}")
    if errors:
        print(f"  Errors:          {errors}")
    print(f"  Input tokens:    {total_in:,}")
    print(f"  Output tokens:   {total_out:,}")
    print(f"  Total tokens:    {total_in + total_out:,}")
    if call_count > 0:
        print(f"  Avg tokens/call: {(total_in + total_out) / call_count:,.0f}")


def cmd_timing(label: str, req_path: str, resp_path: str) -> None:
    """Print timing analysis (RTT per call)."""
    import pyarrow.parquet as pq

    req = pq.read_table(req_path, columns=["request_id", "timestamp_start"])
    resp = pq.read_table(resp_path, columns=["request_id", "timestamp_end"])

    req_map = {}
    for i in range(len(req)):
        rid = str(req["request_id"][i].as_py())
        ts = req["timestamp_start"][i].as_py()
        req_map[rid] = ts

    resp_map = {}
    for i in range(len(resp)):
        rid = str(resp["request_id"][i].as_py())
        ts = resp["timestamp_end"][i].as_py()
        resp_map[rid] = ts

    rtts = []
    for rid, t_start in req_map.items():
        t_end = resp_map.get(rid)
        if t_start and t_end and t_end >= t_start:
            rtts.append((t_end - t_start, rid))

    if not rtts:
        print("No timing data available")
        return

    rtts.sort()
    values = [r[0] for r in rtts]

    def fmt(sec: float) -> str:
        if sec < 0.001:
            return f"{sec*1_000_000:.0f}us"
        elif sec < 1:
            return f"{sec*1000:.0f}ms"
        else:
            return f"{sec:.2f}s"

    print(f"Session: {label}")
    print(f"  Calls with timing: {len(rtts)}")
    print(f"  Min RTT:  {fmt(values[0])}")
    print(f"  Avg RTT:  {fmt(sum(values) / len(values))}")
    print(f"  Max RTT:  {fmt(values[-1])}")
    print(f"  Median:   {fmt(values[len(values) // 2])}")
    print(f"\n  Top 5 slowest calls:")
    for rtt, rid in rtts[-5:]:
        print(f"    {rid}: {fmt(rtt)}")
    print(f"\n  Top 5 fastest calls:")
    for rtt, rid in rtts[:5]:
        print(f"    {rid}: {fmt(rtt)}")


def cmd_tools(label: str, req_path: str, resp_path: str) -> None:
    """Print tool usage aggregation."""
    import pyarrow.parquet as pq

    resp = pq.read_table(resp_path, columns=["tool_uses"])
    tool_counts: Counter = Counter()
    call_with_tools = 0
    call_without_tools = 0

    for row in resp.column("tool_uses").to_pylist():
        if row:
            try:
                tools = json.loads(row)
            except json.JSONDecodeError:
                call_without_tools += 1
                continue
            if tools:
                call_with_tools += 1
                for tu in tools:
                    tool_counts[tu.get("name", "unknown")] += 1
            else:
                call_without_tools += 1
        else:
            call_without_tools += 1

    print(f"Session: {label}")
    print(f"  Calls with tools:    {call_with_tools}")
    print(f"  Calls without tools: {call_without_tools}")
    print(f"  Unique tools:        {len(tool_counts)}")
    if tool_counts:
        print(f"\n  Tool usage (top 20):")
        for name, count in tool_counts.most_common(20):
            print(f"    {name}: {count}")


def cmd_cache(label: str, req_path: str, resp_path: str) -> None:
    """Print cache efficiency metrics."""
    import pyarrow.parquet as pq

    resp = pq.read_table(resp_path, columns=[
        "input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens",
    ])
    total_input = 0
    total_cache_read = 0
    total_cache_write = 0
    calls_with_cache_read = 0
    calls_with_cache_write = 0

    for i in range(len(resp)):
        inp = resp["input_tokens"][i].as_py() or 0
        cr = resp["cache_read_input_tokens"][i].as_py() or 0
        cw = resp["cache_creation_input_tokens"][i].as_py() or 0
        total_input += inp
        total_cache_read += cr
        total_cache_write += cw
        if cr > 0:
            calls_with_cache_read += 1
        if cw > 0:
            calls_with_cache_write += 1

    print(f"Session: {label}")
    print(f"  Total input tokens:        {total_input:,}")
    print(f"  Cache read tokens:         {total_cache_read:,}")
    print(f"  Cache write tokens:        {total_cache_write:,}")
    if total_input > 0:
        print(f"  Cache hit rate (tokens):   {total_cache_read / total_input:.1%}")
    print(f"  Calls with cache reads:    {calls_with_cache_read}")
    print(f"  Calls with cache writes:   {calls_with_cache_write}")


def cmd_models(label: str, req_path: str, resp_path: str) -> None:
    """Print model distribution."""
    import pyarrow.parquet as pq

    resp = pq.read_table(resp_path, columns=["model"])
    models = Counter(resp.column("model").to_pylist())

    print(f"Session: {label}")
    print(f"  Total calls: {len(resp)}")
    if models:
        print(f"\n  Model distribution:")
        for model, count in models.most_common():
            pct = count / len(resp) * 100
            print(f"    {model}: {count} calls ({pct:.1f}%)")


def cmd_call_detail(label: str, req_path: str, resp_path: str, request_id: str) -> None:
    """Print full detail for a single call by request_id."""
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    resp = pq.read_table(resp_path)
    req = pq.read_table(req_path)

    resp_rows = resp.filter(pc.equal(resp.column("request_id"), request_id))
    req_rows = req.filter(pc.equal(req.column("request_id"), request_id))

    if len(resp_rows) == 0 and len(req_rows) == 0:
        print(f"No call found with request_id: {request_id}")
        return

    print(f"Call Detail: {request_id}")
    print(f"Session: {label}")

    if len(req_rows) > 0:
        rd = req_rows.to_pydict()
        print(f"\n--- Request ---")
        print(f"  Model:      {rd['model'][0]}")
        print(f"  Max tokens: {rd['max_tokens'][0]}")
        print(f"  Stream:     {rd['stream'][0]}")
        if rd["timestamp_start"][0]:
            print(f"  Timestamp:  {rd['timestamp_start'][0]} -> {rd['timestamp_end'][0]}")

        if rd["messages"][0]:
            msgs = json.loads(rd["messages"][0])
            print(f"  Messages:   {len(msgs)} total")
            for i, msg in enumerate(msgs[:5]):
                role = msg.get("role", "?")
                content = str(msg.get("content", ""))[:100]
                print(f"    [{i}] {role}: {content}...")
            if len(msgs) > 5:
                print(f"    ... and {len(msgs) - 5} more")

        if rd["tools"][0]:
            tools = json.loads(rd["tools"][0])
            print(f"  Tools:      {len(tools)} defined")
            for t in tools[:5]:
                print(f"    - {t.get('name', '?')}")

    if len(resp_rows) > 0:
        rd = resp_rows.to_pydict()
        print(f"\n--- Response ---")
        print(f"  Status:     {rd['status_code'][0]}")
        if rd["model"][0]:
            print(f"  Model:      {rd['model'][0]}")
        print(f"  Input tokens:    {rd['input_tokens'][0]}")
        print(f"  Output tokens:   {rd['output_tokens'][0]}")
        if rd["cache_read_input_tokens"][0]:
            print(f"  Cache read:      {rd['cache_read_input_tokens'][0]}")
        if rd["stop_reason"][0]:
            print(f"  Stop reason:     {rd['stop_reason'][0]}")
        if rd["error_message"][0]:
            print(f"  Error:           {rd['error_message'][0]}")
        if rd["thinking"][0]:
            print(f"  Thinking:        {len(rd['thinking'][0])} chars")
        if rd["text"][0]:
            print(f"  Text:            {len(rd['text'][0])} chars")
            text_preview = rd['text'][0][:500]
            print(f"    {text_preview}...")
        if rd["tool_uses"][0]:
            tools = json.loads(rd["tool_uses"][0])
            print(f"  Tool uses:       {len(tools)}")
            for tu in tools[:10]:
                name = tu.get("name", "?")
                inp = json.dumps(tu.get("input", {}))[:100]
                print(f"    - {name}: {inp}")
            if len(tools) > 10:
                print(f"    ... and {len(tools) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description="Query LLM flow parquet caches for quick analysis.",
    )
    parser.add_argument("--flows-dir", default="./flows", help="Path to flows directory")
    parser.add_argument("--list-sessions", action="store_true", help="List all discovered sessions")
    parser.add_argument("--session", help="Session index or label to query (e.g. '06' or '06_flows-mission')")
    parser.add_argument("--metric", choices=["tokens", "timing", "tools", "cache", "models", "call-detail"],
                        help="Metric to compute")
    parser.add_argument("--request-id", help="ULID for call-detail metric")
    args = parser.parse_args()

    flows_dir = os.path.abspath(args.flows_dir)
    if not os.path.isdir(flows_dir):
        print(f"Error: flows directory not found: {flows_dir}", file=sys.stderr)
        sys.exit(1)

    results = find_parquet_files(flows_dir)

    if args.list_sessions:
        if not results:
            print("No parquet cache files found.")
        else:
            cmd_list_sessions(results)
        return

    if not args.session:
        print("Error: --session is required (use --list-sessions to see options)", file=sys.stderr)
        sys.exit(1)

    if not args.metric:
        print("Error: --metric is required", file=sys.stderr)
        sys.exit(1)

    match = _find_by_index(results, args.session)
    if match is None:
        print(f"Error: session '{args.session}' not found in {flows_dir}", file=sys.stderr)
        print("Available sessions:")
        for label, _, _ in results:
            print(f"  {label}")
        sys.exit(1)

    label, req_path, resp_path = match

    if args.metric == "tokens":
        cmd_tokens(label, req_path, resp_path)
    elif args.metric == "timing":
        cmd_timing(label, req_path, resp_path)
    elif args.metric == "tools":
        cmd_tools(label, req_path, resp_path)
    elif args.metric == "cache":
        cmd_cache(label, req_path, resp_path)
    elif args.metric == "models":
        cmd_models(label, req_path, resp_path)
    elif args.metric == "call-detail":
        if not args.request_id:
            print("Error: --request-id is required for call-detail metric", file=sys.stderr)
            sys.exit(1)
        cmd_call_detail(label, req_path, resp_path, args.request_id)


if __name__ == "__main__":
    main()
