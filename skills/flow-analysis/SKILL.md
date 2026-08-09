---
name: flow-analysis
version: 1.0.0
description: |
  Analyze LLM API call flows captured by mitmproxy. Use when the user wants to:
  - Query token usage, timing, tool calls, cache efficiency, or model distribution across sessions
  - Compare metric trends between sessions
  - Investigate specific LLM API requests/responses by ID
  - Extract patterns from parsed flow data (parquet caches or raw flow files)
  - Answer analytical questions about agent behavior from captured API traffic
  This skill teaches direct parquet cache reading and use of the llm_flow_viewer.parser layer.
---

# Flow Analysis

This skill teaches you to read and analyze LLM API call flow data captured by this project. The data originates from mitmproxy captures of LLM API traffic and is stored in two layers: raw flow files and parquet caches.

## Quick start

When the user asks an analytical question, first determine whether the data is already cached as parquet. If it is, query the parquet files directly with `pyarrow.parquet` -- this is the fastest path and avoids re-parsing. If not, use the parser layer to parse from raw flow files.

### Fast path: query parquet directly

```python
import pyarrow.parquet as pq

# Read request parquet -- columns: request_id, model, max_tokens, messages (JSON), tools (JSON),
#   system (JSON), output_config (JSON), thinking (JSON), stream, timestamp_start, timestamp_end
req = pq.read_table("flows/<hash>_<name>_requests.parquet")

# Read response parquet -- columns: request_id, thinking, text, tool_uses (JSON),
#   message_id, model, role, input_tokens, output_tokens, cache_creation_input_tokens,
#   cache_read_input_tokens, service_tier, stop_reason, stop_sequence,
#   status_code, error_message, timestamp_start, timestamp_end
resp = pq.read_table("flows/<hash>_<name>_responses.parquet")

# Convert to Python dicts for analysis
req_dict = req.to_pydict()
resp_dict = resp.to_pydict()
```

Always use `.to_pydict()` once per table rather than slicing in a loop -- it's O(n) vs O(n^2).

### Slow path: parse raw flow files

```python
# If no parquet cache exists, use the parser layer
from llm_flow_viewer.parser import flow_file_to_session

session = flow_file_to_session("flows/06_flows-mission", index=6, task_name="mission")
for call in session.calls:
    print(call.request_id, call.response.input_tokens, call.response.output_tokens)
```

### Helper script

For one-shot queries, use the helper script:

```bash
.venv\Scripts\python.exe .factory/skills/flow-analysis/scripts/query_flows.py --help
```

## Data models

The full model hierarchy is in `src/llm_flow_viewer/parser/models.py`. Key structures:

### Response parquet columns

| Column | Type | Meaning |
|--------|------|---------|
| `request_id` | `utf8` | ULID joining request+response |
| `thinking` | `utf8` | DeepSeek reasoning/thinking content |
| `text` | `utf8` | Generated text output |
| `tool_uses` | `utf8` (JSON) | Array of `{name, id, input}` objects |
| `model` | `utf8` | Model name (e.g. `deepseek-v4-flash`) |
| `input_tokens` | `int64` | Prompt tokens consumed |
| `output_tokens` | `int64` | Completion tokens generated |
| `cache_creation_input_tokens` | `int64` | Tokens written to prompt cache |
| `cache_read_input_tokens` | `int64` | Tokens served from prompt cache |
| `service_tier` | `utf8` | API service tier |
| `stop_reason` | `utf8` | `end_turn`, `tool_use`, `max_tokens`, `stop_sequence` |
| `stop_sequence` | `utf8` | The sequence that stopped generation |
| `status_code` | `int64` | HTTP status (200 = ok) |
| `error_message` | `utf8` | Error text if request failed |
| `timestamp_start` | `float64` | Unix seconds when response started |
| `timestamp_end` | `float64` | Unix seconds when response ended |

### Request parquet columns

| Column | Type | Meaning |
|--------|------|---------|
| `request_id` | `utf8` | ULID joining request+response |
| `model` | `utf8` | Requested model name |
| `max_tokens` | `int64` | Max generation tokens |
| `messages` | `utf8` (JSON) | Full message array |
| `tools` | `utf8` (JSON) | Tool definitions |
| `system` | `utf8` (JSON) | System prompt messages |
| `output_config` | `utf8` (JSON) | Output format config |
| `thinking` | `utf8` (JSON) | Thinking config (enabled, budget_tokens) |
| `stream` | `bool` | Whether streaming was requested |
| `timestamp_start` | `float64` | Unix seconds when request started |
| `timestamp_end` | `float64` | Unix seconds when request ended |

### Python data model (ParsedResponse)

When using the parser layer directly, fields map as follows:

- `response.input_tokens` / `response.output_tokens` -- token counts
- `response.cache_read_input_tokens` / `response.cache_creation_input_tokens` -- cache tokens
- `response.tool_uses` -- `List[ToolUse]` with `.name`, `.id`, `.input` (dict)
- `response.thinking` / `response.text` -- content strings
- `response.stop_reason` -- why generation stopped
- `response.status_code` / `response.error_message` -- HTTP status
- `call.timing.request_start` / `call.timing.response_end` -- for RTT computation

## Session discovery

Sessions are stored as files in the `flows/` directory. Each session may have:

- **Raw flow file**: `flows/<NN>_flows-<task_name>` -- the mitmproxy binary dump
- **Parquet caches**: `<hash>_<NN>_flows-<task_name>_requests.parquet` / `_responses.parquet` -- generated on first parse
- **Dashboard cache**: `flows/.dashboard_metrics.json` -- pre-computed cross-session metrics

To discover sessions:

```python
from llm_flow_viewer.tui.widgets.session_list import discover_sessions

sessions = discover_sessions("flows/")
for s in sessions:
    print(f"{s.index:02d} | {s.task_name} -> {s.file_path}")
```

## Common analysis patterns

### Token usage by session

```python
import pyarrow.parquet as pq
import os

for fname in sorted(os.listdir("flows/")):
    if not fname.endswith("_responses.parquet"):
        continue
    path = os.path.join("flows", fname)
    table = pq.read_table(path, columns=["input_tokens", "output_tokens"])
    inp = sum(v for v in table.column("input_tokens").to_pylist() if v)
    out = sum(v for v in table.column("output_tokens").to_pylist() if v)
    print(f"{fname}: {table.num_rows} calls, {inp:,} in, {out:,} out")
```

### Timing analysis (round-trip time)

```python
# Join requests and responses on request_id, compute RTT
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
        rtts.append(t_end - t_start)

if rtts:
    print(f"RTT: min={min(rtts):.2f}s, avg={sum(rtts)/len(rtts):.2f}s, max={max(rtts):.2f}s")
```

### Tool usage aggregation

```python
import json

resp = pq.read_table(resp_path, columns=["tool_uses"])
tool_counts = {}
for row in resp.column("tool_uses").to_pylist():
    if row:
        for tu in json.loads(row):
            name = tu.get("name", "")
            tool_counts[name] = tool_counts.get(name, 0) + 1

for name, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
    print(f"  {name}: {count}")
```

### Cache efficiency

```python
resp = pq.read_table(resp_path, columns=["cache_read_input_tokens", "input_tokens"])
total_cache_read = sum(v for v in resp.column("cache_read_input_tokens").to_pylist() if v)
total_input = sum(v for v in resp.column("input_tokens").to_pylist() if v)
if total_input > 0:
    print(f"Cache hit rate: {total_cache_read/total_input:.1%}")
```

### Model distribution

```python
from collections import Counter

resp = pq.read_table(resp_path, columns=["model"])
models = Counter(resp.column("model").to_pylist())
for model, count in models.most_common():
    print(f"  {model}: {count}")
```

### Inspect a single call by request_id

```python
import pyarrow.compute as pc

target = "01JN..."  # the ULID request_id
resp = pq.read_table(resp_path)
req = pq.read_table(req_path)

resp_rows = resp.filter(pc.equal(resp.column("request_id"), target)).to_pydict()
req_rows = req.filter(pc.equal(req.column("request_id"), target)).to_pydict()
```

### Message count per call

```python
import json

req = pq.read_table(req_path, columns=["messages"])
msg_counts = []
for row in req.column("messages").to_pylist():
    if row:
        msgs = json.loads(row)
        msg_counts.append(len(msgs))
    else:
        msg_counts.append(0)
print(f"Messages per call: min={min(msg_counts)}, avg={sum(msg_counts)/len(msg_counts):.0f}, max={max(msg_counts)}")
```

## Edge cases

### No parquet cache exists
Fall back to the parser layer: `flow_file_to_session(file_path, index, task_name)`. This parses the raw flow file and auto-writes parquet caches for future fast-path access.

### Stale parquet cache
Use `is_cache_fresh(flow_file_path)` from `llm_flow_viewer.parser.cache` to check. If stale, call `load_or_parse_cached(file_path)` which handles invalidation and re-parsing automatically.

### Zero-call sessions
Some sessions may have 0 LLM API calls (flow file contained no matching POST requests to the API endpoint). Check `table.num_rows == 0` before processing. Don't divide by zero.

### Error responses
Check `status_code != 200` or non-empty `error_message` in the response parquet. These calls still appear in the data but may have `None` for token/timing fields. Filter them out or handle separately.

### Missing optional fields
`thinking`, `cache_read_input_tokens`, `cache_creation_input_tokens`, `stop_reason`, `service_tier`, `output_config`, `thinking_config`, `message_id`, `role` may all be `None`. Always guard with `if v is not None` or use `or 0` / `or ""` defaults.

### JSON fields in parquet
`tool_uses`, `messages`, `tools`, `system`, `output_config`, `thinking` are stored as JSON strings in the parquet files. Parse with `json.loads()`. Handle `json.JSONDecodeError` for malformed data -- use the `_safe_json_loads()` utility from `llm_flow_viewer.parser.cache` if needed.

### Large sessions
Use `pq.read_table(path, columns=[...])` to read only the columns you need. The `06_flows-mission` session produces large parquet files. Always specify column projection to avoid reading unnecessary data.

### Non-numeric session names
Session files without numeric prefixes (e.g. `my_custom_flows`) use `discover_sessions()` which assigns synthetic indices by file modification time. Don't assume `NN_flows-*` naming.

## Verify it worked

Run these checks after your analysis:

```python
import pyarrow.parquet as pq
import os

# 1. Find at least one parquet file
parquet_files = [f for f in os.listdir("flows/") if f.endswith(".parquet")]
assert parquet_files, "No parquet cache files found"
print(f"Found {len(parquet_files)} parquet files")

# 2. Read one file and confirm row count
table = pq.read_table(os.path.join("flows", parquet_files[0]))
print(f"File: {parquet_files[0]}, rows: {table.num_rows}, columns: {len(table.schema)}")

# 3. Cross-check with parser output
from llm_flow_viewer.parser import flow_file_to_session
# session = flow_file_to_session("flows/01_flows-analyze_codebase", 1, "analyze")
# assert len(session.calls) == table.num_rows
```
