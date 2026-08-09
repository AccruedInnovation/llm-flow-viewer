"""Custom widgets for the Dashboard screen: metrics panel, bar charts, avg tokens.

Uses Rich renderables (Table, Bar) inside Textual Static widgets for
a clean terminal-based chart display.
"""

from __future__ import annotations

from textual.binding import Binding
from textual.reactive import reactive
from textual.widgets import Static
from rich.table import Table
from rich.bar import Bar
from rich.style import Style
from rich.text import Text
from rich import box


# ---------------------------------------------------------------------------
# Color palette for chart bars
# ---------------------------------------------------------------------------

_BAR_COLORS = [
    "#00aa00",  # green
    "#0088ff",  # blue
    "#ff8800",  # orange
    "#cc44cc",  # purple
    "#ffcc00",  # yellow
    "#ff4444",  # red
    "#44cccc",  # cyan
]

# Use Rich-compatible style constants (no Textual CSS variables)
_STYLE_BOLD = Style(bold=True)
_STYLE_BOLD_ACCENT = Style(bold=True, color="#00aa00")
_STYLE_ITALIC_MUTED = Style(italic=True, color="#888888")
_STYLE_BOLD_ITALIC = Style(bold=True, italic=True)


def _format_number(n: int | float) -> str:
    """Format a number with abbreviation for readability.

    - 0-999: plain number
    - 1,000-999,999: X.XK
    - 1,000,000+: X.XM
    """
    if n < 0:
        return "0"
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n/1000:.1f}K"
    return f"{n/1_000_000:.1f}M"


def _format_duration(seconds: float, always_show_ms: bool = False) -> str:
    """Format a duration in seconds as a human-readable string.

    Durations < 1 second are shown in milliseconds (e.g. ``"847ms"``).
    Durations >= 1 second are shown with one decimal place (e.g. ``"2.1s"``).

    Args:
        seconds: Duration in seconds.
        always_show_ms: If True, always show as whole ms even for >=1s.

    Returns:
        A human-readable duration string.
    """
    if seconds < 1.0:
        ms = int(round(seconds * 1000))
        if ms < 1:
            return "<1ms"
        return f"{ms}ms"
    if always_show_ms:
        ms = int(round(seconds * 1000))
        return f"{ms}ms"
    return f"{seconds:.1f}s"


def _format_thousands(n: int) -> str:
    """Format a number with thousands separators (e.g., 12800 → '12,800')."""
    return f"{n:,}"


def _bar_color(index: int) -> str:
    """Return a color for the bar at the given index."""
    return _BAR_COLORS[index % len(_BAR_COLORS)]


# ---------------------------------------------------------------------------
# Label width computation and truncation helpers
# ---------------------------------------------------------------------------


def _compute_max_label_width(widget_width: int, column_count: int) -> int:
    """Compute the maximum available width for labels in a chart table.

    Estimates the character width available for labels based on the
    widget's content region width and the number of table columns.
    Accounts for borders, per-column padding, and typical ratio-based
    column distribution where the label column receives roughly 30%
    of usable space.

    Args:
        widget_width: The content region width of the widget in characters.
        column_count: The number of columns in the Rich table.

    Returns:
        Maximum label width in characters, minimum 4.
    """
    if widget_width <= 0:
        return 40
    # Reserve space for: 2 border chars + 2 padding chars per column + slack
    reserved = 2 + (column_count * 2) + 2
    available = max(widget_width - reserved, 0)
    # Label typically gets ~40% of remaining width in ratio-based tables
    label_width = int(available * 0.40)
    return max(label_width, 4)


def _truncate_label(label: str | Text, max_width: int) -> Text:
    """Truncate a label to *max_width* characters, appending ``...``.

    Preserves any Rich styling (bold, colors) applied to the label.
    If *max_width* is less than 4, returns just ``"..."``.

    Args:
        label: The full label string or styled Text object.
        max_width: Maximum allowed length before truncation.

    Returns:
        A Text object with the label (possibly truncated with ``...``).
    """
    if isinstance(label, Text):
        plain = label.plain
        style = label.style
    else:
        plain = str(label)
        style = None

    if max_width < 4:
        if style is not None:
            return Text("...", style=style)
        return Text("...")

    if len(plain) <= max_width:
        if isinstance(label, Text):
            return label
        return Text(label)

    truncated = plain[:max_width - 3] + "..."
    if style is not None:
        return Text(truncated, style=style)
    return Text(truncated)


# ---------------------------------------------------------------------------
# Metrics Summary Panel
# ---------------------------------------------------------------------------


class MetricsPanel(Static):
    """Displays aggregate metrics summary (total sessions, calls, tokens).

    Renders a multi-line metrics table with visible border.
    """

    DEFAULT_CSS = """
    MetricsPanel {
        width: 100%;
        height: auto;
        margin: 0 0 1 0;
        border: solid $primary;
    }
    """

    # Reactive data: (sessions, calls, input_tokens, output_tokens)
    metrics_data: tuple[int, int, int, int] = reactive((0, 0, 0, 0))

    def __init__(
        self,
        total_sessions: int = 0,
        total_calls: int = 0,
        total_input_tokens: int = 0,
        total_output_tokens: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.metrics_data = (
            total_sessions,
            total_calls,
            total_input_tokens,
            total_output_tokens,
        )
        self.border_title = "Metrics Summary"

    def watch_metrics_data(
        self, old: tuple[int, int, int, int], new: tuple[int, int, int, int]
    ) -> None:
        """Re-render when metrics data changes."""
        self.refresh()

    def render(self) -> Table:
        """Render the metrics panel as a plain Rich Table."""
        sessions, calls, input_tok, output_tok = self.metrics_data
        total_tok = input_tok + output_tok

        table = Table(
            show_header=False,
            box=box.SIMPLE,
            padding=(0, 2),
            expand=True,
        )
        table.add_column("Metric", justify="left")
        table.add_column("Value", justify="right")

        table.add_row(
            Text("Total Sessions", style=_STYLE_BOLD),
            Text(str(sessions), style=_STYLE_BOLD_ACCENT),
        )
        table.add_row(
            Text("Total Calls", style=_STYLE_BOLD),
            Text(_format_thousands(calls), style=_STYLE_BOLD_ACCENT),
        )
        table.add_row(
            Text("Total Input Tokens", style=_STYLE_BOLD),
            Text(_format_thousands(input_tok), style=_STYLE_BOLD_ACCENT),
        )
        table.add_row(
            Text("Total Output Tokens", style=_STYLE_BOLD),
            Text(_format_thousands(output_tok), style=_STYLE_BOLD_ACCENT),
        )
        table.add_row(
            Text("Total Tokens", style=_STYLE_BOLD),
            Text(_format_thousands(total_tok), style=_STYLE_BOLD_ACCENT),
        )

        return table


# ---------------------------------------------------------------------------
# Bar Chart Widget
# ---------------------------------------------------------------------------


class SessionBarChart(Static):
    """Horizontal bar chart showing a metric per session.

    Uses Rich Table renderable with each row containing a session label,
    numeric value, and a Rich Bar visual element.
    """

    DEFAULT_CSS = """
    SessionBarChart {
        width: 100%;
        height: auto;
        margin: 0 0 1 0;
        border: solid $primary;
    }
    """

    # Reactive data: list of (label, value) tuples
    chart_data: list[tuple[str, float | int]] = reactive(list, always_update=True)
    _chart_title: str = ""
    _value_label: str = "Value"

    def __init__(
        self,
        title: str = "Chart",
        value_label: str = "Value",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._chart_title = title
        self._value_label = value_label
        self.border_title = title

    def update_data(self, data: list[tuple[str, float | int]]) -> None:
        """Update the chart data and re-render."""
        self.chart_data = list(data)

    def watch_chart_data(
        self, old: list, new: list[tuple[str, float | int]]
    ) -> None:
        """Re-render when data changes."""
        self.refresh()

    def render(self) -> Table | Text:
        """Render the bar chart as a Rich Table."""
        if not self.chart_data:
            return Text("No data", style=_STYLE_ITALIC_MUTED)

        max_val = max(v for _, v in self.chart_data) if self.chart_data else 1
        if max_val == 0:
            max_val = 1

        table = Table(
            show_header=True,
            header_style=_STYLE_BOLD,
            box=box.SIMPLE,
            padding=(0, 1),
            expand=True,
        )
        table.add_column("Session", no_wrap=True, ratio=4)
        table.add_column(self._value_label, justify="right", no_wrap=True, ratio=1)
        table.add_column("", ratio=4)

        bar_width = 30
        for i, (label, val) in enumerate(self.chart_data):
            ratio = val / max_val if max_val > 0 else 0
            color = _bar_color(i)

            if isinstance(val, float) and val != int(val):
                val_str = f"{val:.1f}"
            else:
                val_str = _format_number(int(val))

            bar = Bar(
                size=1.0,
                begin=0,
                end=min(ratio, 1.0),
                width=bar_width,
                color=color,
            )

            max_label_width = _compute_max_label_width(
                self.content_region.width, 3
            )
            label_text = _truncate_label(
                Text(label, no_wrap=True), max_label_width
            )

            table.add_row(label_text, val_str, bar)

        return table


# ---------------------------------------------------------------------------
# Token Bar Chart (split input/output tokens)
# ---------------------------------------------------------------------------


class TokenBarChart(Static):
    """Horizontal bar chart showing input and output tokens per session.

    Each session renders two side-by-side bars in a single row — one for
    *input_tokens* (blue) and one for *output_tokens* (orange).  Both bars
    are scaled proportionally to the maximum value across all input and
    output token counts, so the visual ratio between input and output is
    preserved.

    **VAL-DASH-003**: This widget replaces the single-bar-per-session
    ``SessionBarChart`` for the *Token Usage per Session* chart.  It fulfils
    the requirement that *"both input tokens and output tokens are shown
    (stacked bars or side-by-side bars)"*.

    Data format: ``list[tuple[str, int, int]]`` where each tuple is
    ``(label, input_tokens, output_tokens)``.

    Edge cases handled:
    - Empty data → shows "No data"
    - Zero input tokens, zero output tokens, or both → empty bars drawn
    - Single session → works identically (one row, two bars)
    - Very large token counts → abbreviated (1.2K, 4.5M, etc.)
    """

    DEFAULT_CSS = """
    TokenBarChart {
        width: 100%;
        height: auto;
        margin: 0 0 1 0;
        border: solid $primary;
    }
    """

    # Reactive data: list of (label, input_tokens, output_tokens)
    chart_data: list[tuple[str, int, int]] = reactive(list, always_update=True)
    _chart_title: str = "Chart"

    _INPUT_COLOR = "#0088ff"   # blue
    _OUTPUT_COLOR = "#ff8800"  # orange

    def __init__(
        self,
        title: str = "Chart",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._chart_title = title
        self.border_title = title

    def update_data(self, data: list[tuple[str, int, int]]) -> None:
        """Update the chart data and re-render.

        Args:
            data: List of ``(label, input_tokens, output_tokens)`` tuples.
        """
        self.chart_data = list(data)

    def watch_chart_data(
        self,
        old: list,
        new: list[tuple[str, int, int]],
    ) -> None:
        """Re-render when data changes."""
        self.refresh()

    def render(self) -> Table | Text:
        """Render the token bar chart as a Rich Table with two bars per row."""
        if not self.chart_data:
            return Text("No data", style=_STYLE_ITALIC_MUTED)

        # Determine common scale: max of all input and output values
        max_val = max(
            (max(inp, out) for _, inp, out in self.chart_data),
            default=1,
        )
        if max_val == 0:
            max_val = 1

        table = Table(
            show_header=True,
            header_style=_STYLE_BOLD,
            box=box.SIMPLE,
            padding=(0, 1),
            expand=True,
        )
        table.add_column("Session", no_wrap=True, ratio=3)
        table.add_column("Input", justify="right", no_wrap=True, ratio=2)
        table.add_column("", ratio=3)
        table.add_column("Output", justify="right", no_wrap=True, ratio=2)

        bar_width = 16
        for i, (label, inp, out_val) in enumerate(self.chart_data):
            inp_ratio = inp / max_val if max_val > 0 else 0
            out_ratio = out_val / max_val if max_val > 0 else 0

            inp_bar = Bar(
                size=1.0,
                begin=0,
                end=min(inp_ratio, 1.0),
                width=bar_width,
                color=self._INPUT_COLOR,
            )
            out_bar = Bar(
                size=1.0,
                begin=0,
                end=min(out_ratio, 1.0),
                width=bar_width,
                color=self._OUTPUT_COLOR,
            )

            max_label_width = _compute_max_label_width(
                self.content_region.width, 5
            )
            label_text = _truncate_label(
                Text(label, no_wrap=True), max_label_width
            )

            inp_str = _format_number(inp)
            out_str = _format_number(out_val)

            table.add_row(label_text, inp_str, inp_bar, out_str, out_bar)

        return table


# ---------------------------------------------------------------------------
# Avg Tokens Per Call Chart
# ---------------------------------------------------------------------------


class AvgTokensChart(Static):
    """Displays average tokens per call per session.

    Similar to SessionBarChart but specifically for avg tokens.
    Includes a computed average across all sessions as a header.
    """

    DEFAULT_CSS = """
    AvgTokensChart {
        width: 100%;
        height: auto;
        margin: 0 0 1 0;
        border: solid $primary;
    }
    """

    avg_data: list[tuple[str, float]] = reactive(list, always_update=True)
    _overall_avg: float = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.avg_data = []
        self.border_title = "Avg Tokens / Call"

    def update_data(
        self, data: list[tuple[str, float]], overall_avg: float = 0
    ) -> None:
        """Update chart data and overall average."""
        self.avg_data = list(data)
        self._overall_avg = overall_avg

    def watch_avg_data(
        self, old: list, new: list[tuple[str, float]]
    ) -> None:
        """Re-render when data changes."""
        self.refresh()

    def render(self) -> Table | Text:
        """Render the avg tokens chart."""
        if not self.avg_data:
            return Text("No data", style=_STYLE_ITALIC_MUTED)

        max_val = max(v for _, v in self.avg_data) if self.avg_data else 1
        if max_val == 0:
            max_val = 1

        table = Table(
            show_header=True,
            header_style=_STYLE_BOLD,
            box=box.SIMPLE,
            padding=(0, 1),
            expand=True,
        )
        table.add_column("Session", no_wrap=True)
        table.add_column("Avg Tokens", justify="right", no_wrap=True)
        table.add_column("")

        bar_width = 30
        for i, (label, val) in enumerate(self.avg_data):
            ratio = val / max_val if max_val > 0 else 0
            color = _bar_color(i)

            max_label_width = _compute_max_label_width(
                self.content_region.width, 3
            )
            label_text = _truncate_label(
                Text(label, no_wrap=True), max_label_width
            )

            bar = Bar(
                size=1.0,
                begin=0,
                end=min(ratio, 1.0),
                width=bar_width,
                color=color,
            )

            table.add_row(label_text, f"{val:,.1f}", bar)

        if self._overall_avg > 0:
            table.add_row(
                Text("Overall Avg", style=_STYLE_BOLD_ITALIC),
                Text(f"{self._overall_avg:,.1f}", style=_STYLE_BOLD_ACCENT),
                "",
            )

        return table


# ---------------------------------------------------------------------------
# Timing Chart (min / avg / max RTT per session)
# ---------------------------------------------------------------------------


class TimingChart(Static):
    """Displays min/avg/max round-trip time per session with bars.

    Each row shows a session label, three timing values (min, avg, max),
    and a proportional bar for each metric.  The session with the lowest
    average RTT (fastest) is highlighted in green.  The session with the
    highest average RTT (slowest) is highlighted in red.

    Time values are formatted in human-readable units:
      - < 1 ms → ``"<1ms"``
      - >= 1 ms, < 1 s → ``"Xms"``
      - >= 1 s → ``"X.Xs"``
    """

    DEFAULT_CSS = """
    TimingChart {
        width: 100%;
        height: auto;
        margin: 0 0 1 0;
        border: solid $primary;
    }
    """

    # Reactive data: list of (label, min_rtt, avg_rtt, max_rtt) tuples.
    # Each RTT value is in seconds or None if unavailable.
    timing_data: list[tuple[str, float | None, float | None, float | None]] = (
        reactive(list, always_update=True)
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.timing_data = []
        self.border_title = "Round-Trip Time per Session"

    def update_data(
        self,
        data: list[tuple[str, float | None, float | None, float | None]],
    ) -> None:
        """Update the chart data and re-render.

        Args:
            data: List of ``(label, min_rtt_sec, avg_rtt_sec, max_rtt_sec)``
                  tuples.  Any value may be ``None`` if timing is unavailable.
        """
        self.timing_data = list(data)

    def watch_timing_data(
        self,
        old: list,
        new: list[tuple[str, float | None, float | None, float | None]],
    ) -> None:
        """Re-render when data changes."""
        self.refresh()

    def render(self) -> Table | Text:
        """Render the timing chart as a Rich Table with bars."""
        if not self.timing_data:
            return Text("No timing data", style=_STYLE_ITALIC_MUTED)

        # Use all entries — sessions with all-None values get N/A text
        entries = list(self.timing_data)

        # Determine fastest / slowest by avg RTT (only among sessions with avg data)
        entries_with_avg = [
            (label, mn or 0, avg, mx or 0)
            for (label, mn, avg, mx) in entries
            if avg is not None
        ]
        fastest_idx: int | None = None
        slowest_idx: int | None = None
        if entries_with_avg:
            sorted_by_avg = sorted(entries_with_avg, key=lambda x: x[2])
            fastest_label = sorted_by_avg[0][0]
            slowest_label = sorted_by_avg[-1][0]
            for i, (label, _, _, _) in enumerate(entries):
                if label == fastest_label:
                    fastest_idx = i
                if label == slowest_label:
                    slowest_idx = i

        # Find global max for bar scaling (among all non-None values)
        all_vals: list[float] = []
        for _, mn, avg, mx in entries:
            if mn is not None:
                all_vals.append(mn)
            if avg is not None:
                all_vals.append(avg)
            if mx is not None:
                all_vals.append(mx)
        max_val = max(all_vals) if all_vals else 1.0
        if max_val == 0:
            max_val = 1.0

        table = Table(
            show_header=True,
            header_style=_STYLE_BOLD,
            box=box.SIMPLE,
            padding=(0, 1),
            expand=True,
        )
        table.add_column("Session", no_wrap=True)
        table.add_column("Min", justify="right", no_wrap=True)
        table.add_column("")
        table.add_column("Avg", justify="right", no_wrap=True)
        table.add_column("")
        table.add_column("Max", justify="right", no_wrap=True)
        table.add_column("")

        bar_width = 12
        for i, (label, mn, avg, mx) in enumerate(entries):
            # Determine row style
            is_fastest = fastest_idx is not None and i == fastest_idx
            is_slowest = slowest_idx is not None and i == slowest_idx

            label_style = Style(bold=True)
            if is_fastest:
                label_style = Style(bold=True, color="#00cc00")
            elif is_slowest:
                label_style = Style(bold=True, color="#ff4444")

            max_label_width = _compute_max_label_width(
                self.content_region.width, 7
            )
            label_text = _truncate_label(
                Text(label, style=label_style, no_wrap=True), max_label_width
            )

            # Format each value
            min_str = _format_duration(mn) if mn is not None else "N/A"
            avg_str = _format_duration(avg) if avg is not None else "N/A"
            max_str = _format_duration(mx) if mx is not None else "N/A"

            # Build bars proportional to global max
            min_ratio = (mn / max_val) if mn is not None and max_val > 0 else 0
            avg_ratio = (avg / max_val) if avg is not None and max_val > 0 else 0
            max_ratio = (mx / max_val) if mx is not None and max_val > 0 else 0

            min_bar = Bar(
                size=1.0, begin=0, end=min(min_ratio, 1.0),
                width=bar_width, color="#00cc00",
            ) if mn is not None else Text("")

            avg_bar = Bar(
                size=1.0, begin=0, end=min(avg_ratio, 1.0),
                width=bar_width, color="#0088ff",
            ) if avg is not None else Text("")

            max_bar = Bar(
                size=1.0, begin=0, end=min(max_ratio, 1.0),
                width=bar_width, color="#ff4444",
            ) if mx is not None else Text("")

            table.add_row(
                label_text,
                Text(min_str, style=label_style) if is_fastest or is_slowest else min_str,
                min_bar,
                Text(avg_str, style=label_style) if is_fastest or is_slowest else avg_str,
                avg_bar,
                Text(max_str, style=label_style) if is_fastest or is_slowest else max_str,
                max_bar,
            )

        return table


# ---------------------------------------------------------------------------
# Tool Usage Chart (sorted by frequency, top-N with "Other" aggregation)
# ---------------------------------------------------------------------------


class ToolUsageChart(Static):
    """Displays tool invocation counts per session or across all sessions.

    Tools are sorted by frequency descending.  A configurable top-N limit
    (default 10) is applied; tools beyond the limit are aggregated into
    an "Other" row.  Each tool shows its count and percentage of total
    tool invocations.

    Supports both per-session and all-sessions modes.  When showing data
    for a single session, the border title reflects the session label.
    When showing all sessions combined, the border title indicates
    "All Sessions Combined".
    """

    DEFAULT_CSS = """
    ToolUsageChart {
        width: 100%;
        height: auto;
        margin: 0 0 1 0;
        border: solid $primary;
    }
    """

    # Reactive data: list of (tool_name, count) tuples
    tool_data: list[tuple[str, int]] = reactive(list, always_update=True)
    _top_n: int = 10
    _chart_title: str = "Tool Usage"

    def __init__(
        self,
        title: str = "Tool Usage",
        top_n: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._chart_title = title
        self._top_n = top_n
        self.border_title = title

    def update_data(
        self,
        data: list[tuple[str, int]],
        top_n: int | None = None,
    ) -> None:
        """Update the chart data and re-render.

        Args:
            data: List of ``(tool_name, count)`` tuples.
            top_n: Optional override for the top-N limit.
        """
        if top_n is not None:
            self._top_n = top_n
        self.tool_data = list(data)

    def watch_tool_data(
        self, old: list, new: list[tuple[str, int]]
    ) -> None:
        """Re-render when data changes."""
        self.refresh()

    def render(self) -> Table | Text:
        """Render the tool usage chart as a Rich Table with bars."""
        if not self.tool_data:
            return Text("No tool usage data", style=_STYLE_ITALIC_MUTED)

        # Sort by count descending, then alphabetically for ties
        sorted_data = sorted(self.tool_data, key=lambda x: (-x[1], x[0]))

        total_count = sum(count for _, count in sorted_data)

        # Apply top-N limit with "Other" aggregation
        if len(sorted_data) > self._top_n:
            top_data = sorted_data[:self._top_n]
            other_count = sum(count for _, count in sorted_data[self._top_n:])
            display_data = list(top_data)
            if other_count > 0:
                display_data.append(("Other", other_count))
        else:
            display_data = list(sorted_data)

        # Find max count for bar scaling
        max_count = max(c for _, c in display_data) if display_data else 1
        if max_count == 0:
            max_count = 1

        table = Table(
            show_header=True,
            header_style=_STYLE_BOLD,
            box=box.SIMPLE,
            padding=(0, 1),
            expand=True,
        )
        table.add_column("Tool", no_wrap=True)
        table.add_column("Count", justify="right", no_wrap=True)
        table.add_column("%", justify="right", no_wrap=True)
        table.add_column("")

        bar_width = 30
        for i, (name, count) in enumerate(display_data):
            ratio = count / max_count if max_count > 0 else 0
            pct = (count / total_count * 100.0) if total_count > 0 else 0.0
            color = _bar_color(i)

            bar = Bar(
                size=1.0, begin=0, end=min(ratio, 1.0),
                width=bar_width, color=color,
            )

            max_label_width = _compute_max_label_width(
                self.content_region.width, 4
            )
            name_text = _truncate_label(
                Text(name, no_wrap=True), max_label_width
            )

            count_str = _format_thousands(count)
            pct_str = f"{pct:.1f}%" if pct < 100.0 else "100%"

            table.add_row(name_text, count_str, pct_str, bar)

        # Total row
        table.add_row(
            Text("Total", style=_STYLE_BOLD_ITALIC),
            Text(_format_thousands(total_count), style=_STYLE_BOLD_ACCENT),
            "100%",
            "",
        )

        return table


# ---------------------------------------------------------------------------
# Cache Efficiency Chart (cache hit/miss ratio per session)
# ---------------------------------------------------------------------------


class CacheEfficiencyChart(Static):
    """Displays cache efficiency (hit rate) per session as a percentage bar.

    Each row shows a session label, cache read tokens, input tokens, and
    a percentage bar representing the cache hit rate.  The hit rate is
    computed as::

        cache_hit_ratio = cache_read / (cache_read + input_tokens)

    displayed as a percentage.

    Color coding:
    - >90%  : green — high cache efficiency
    - 50-90%: yellow — moderate cache efficiency
    - <50%  : red — low cache efficiency
    - N/A   : sessions with no cache data at all (cache_read is None)

    The widget receives raw cache data and computes percentages internally.
    """

    DEFAULT_CSS = """
    CacheEfficiencyChart {
        width: 100%;
        height: auto;
        margin: 0 0 1 0;
        border: solid $primary;
    }
    """

    # Reactive data: list of (label, cache_read, input_tokens) tuples.
    # cache_read may be None if no cache data exists for that session.
    cache_data: list[tuple[str, int | None, int | None]] = (
        reactive(list, always_update=True)
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cache_data = []
        self.border_title = "Cache Efficiency"

    def update_data(
        self,
        data: list[tuple[str, int | None, int | None]],
    ) -> None:
        """Update the chart data and re-render.

        Args:
            data: List of ``(label, cache_read, input_tokens)`` tuples.
                  cache_read and input_tokens are ``None`` when data is
                  unavailable.
        """
        self.cache_data = list(data)

    def watch_cache_data(
        self,
        old: list,
        new: list[tuple[str, int | None, int | None]],
    ) -> None:
        """Re-render when data changes."""
        self.refresh()

    def _compute_hit_rate(
        self, cache_read: int | None, input_tokens: int | None
    ) -> float | None:
        """Compute cache hit rate as a fraction (0.0-1.0).

        Returns ``None`` if data is unavailable.
        """
        if cache_read is None:
            return None
        cr = int(cache_read)
        inp = int(input_tokens) if input_tokens is not None else 0
        denominator = cr + inp
        if denominator == 0:
            return 0.0
        return cr / denominator

    def _hit_rate_color(self, hit_rate: float | None) -> str:
        """Return a CSS-compatible color string for the given hit rate.

        Args:
            hit_rate: Cache hit rate as a fraction (0.0-1.0), or ``None``.

        Returns:
            A hex colour string.
        """
        if hit_rate is None:
            return "#888888"  # grey for N/A
        if hit_rate >= 0.9:
            return "#00cc00"  # green
        if hit_rate >= 0.5:
            return "#cccc00"  # yellow
        return "#ff4444"  # red

    def render(self) -> Table | Text:
        """Render the cache efficiency chart as a Rich Table with bars."""
        if not self.cache_data:
            return Text("No cache data", style=_STYLE_ITALIC_MUTED)

        # Find max cache_read for bar scaling (among non-None values)
        all_vals: list[int] = []
        for _, cr, _ in self.cache_data:
            if cr is not None:
                all_vals.append(int(cr))
        max_val = max(all_vals) if all_vals else 1
        if max_val == 0:
            max_val = 1

        table = Table(
            show_header=True,
            header_style=_STYLE_BOLD,
            box=box.SIMPLE,
            padding=(0, 1),
            expand=True,
        )
        table.add_column("Session", no_wrap=True)
        table.add_column("Cache Read", justify="right", no_wrap=True)
        table.add_column("Hit Rate", justify="right", no_wrap=True)
        table.add_column("")

        bar_width = 30
        for i, (label, cache_read, input_tokens) in enumerate(self.cache_data):
            hit_rate = self._compute_hit_rate(cache_read, input_tokens)
            color = self._hit_rate_color(hit_rate)

            # Format cache_read with thousands separator or "N/A"
            if cache_read is not None:
                cache_read_str = _format_thousands(int(cache_read))
            else:
                cache_read_str = "N/A"

            # Format hit rate as percentage or "N/A"
            if hit_rate is not None:
                pct = hit_rate * 100.0
                pct_str = f"{pct:.1f}%"
                bar = Bar(
                    size=1.0, begin=0, end=min(hit_rate, 1.0),
                    width=bar_width, color=color,
                )
            else:
                pct_str = "N/A"
                bar = Text("N/A", style=_STYLE_ITALIC_MUTED)

            max_label_width = _compute_max_label_width(
                self.content_region.width, 4
            )
            label_text = _truncate_label(
                Text(label, no_wrap=True, style=Style(bold=True)),
                max_label_width,
            )

            table.add_row(label_text, cache_read_str, pct_str, bar)

        return table


# ---------------------------------------------------------------------------
# Model Usage Chart (model breakdown per session)
# ---------------------------------------------------------------------------


class ModelUsageChart(Static):
    """Displays model usage breakdown per session.

    Shows which LLM models were invoked and their call counts.  Models
    are sorted by count descending.  Each model shows its name, call
    count, and percentage of total calls.

    When viewing a single session, only that session's models are shown.
    When viewing all sessions, the data is aggregated across all sessions.
    """

    DEFAULT_CSS = """
    ModelUsageChart {
        width: 100%;
        height: auto;
        margin: 0 0 1 0;
        border: solid $primary;
    }
    """

    # Reactive data: list of (model_name, count) tuples
    model_data: list[tuple[str, int]] = reactive(list, always_update=True)
    _chart_title: str = "Model Usage"

    def __init__(
        self,
        title: str = "Model Usage",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._chart_title = title
        self.border_title = title

    def update_data(self, data: list[tuple[str, int]]) -> None:
        """Update the chart data and re-render.

        Args:
            data: List of ``(model_name, count)`` tuples.
        """
        self.model_data = list(data)

    def watch_model_data(
        self, old: list, new: list[tuple[str, int]]
    ) -> None:
        """Re-render when data changes."""
        self.refresh()

    def render(self) -> Table | Text:
        """Render the model usage chart as a Rich Table with bars."""
        if not self.model_data:
            return Text("No model usage data", style=_STYLE_ITALIC_MUTED)

        # Sort by count descending, then alphabetically for ties
        sorted_data = sorted(self.model_data, key=lambda x: (-x[1], x[0]))
        total_count = sum(count for _, count in sorted_data)

        max_count = max(c for _, c in sorted_data) if sorted_data else 1
        if max_count == 0:
            max_count = 1

        table = Table(
            show_header=True,
            header_style=_STYLE_BOLD,
            box=box.SIMPLE,
            padding=(0, 1),
            expand=True,
        )
        table.add_column("Model", no_wrap=True)
        table.add_column("Calls", justify="right", no_wrap=True)
        table.add_column("%", justify="right", no_wrap=True)
        table.add_column("")

        bar_width = 30
        for i, (name, count) in enumerate(sorted_data):
            ratio = count / max_count if max_count > 0 else 0
            pct = (count / total_count * 100.0) if total_count > 0 else 0.0
            color = _bar_color(i)

            bar = Bar(
                size=1.0, begin=0, end=min(ratio, 1.0),
                width=bar_width, color=color,
            )

            max_label_width = _compute_max_label_width(
                self.content_region.width, 4
            )
            name_text = _truncate_label(
                Text(name, no_wrap=True), max_label_width
            )

            count_str = _format_thousands(count)
            pct_str = f"{pct:.1f}%" if pct < 100.0 else "100%"

            table.add_row(name_text, count_str, pct_str, bar)

        # Total row
        table.add_row(
            Text("Total", style=_STYLE_BOLD_ITALIC),
            Text(_format_thousands(total_count), style=_STYLE_BOLD_ACCENT),
            "100%",
            "",
        )

        return table


# ---------------------------------------------------------------------------
# Comparison Panel (side-by-side session comparison)
# ---------------------------------------------------------------------------


class ComparisonPanel(Static):
    """Displays metrics for a single session in comparison mode.

    Shows a summary of call count, input/output tokens, average RTT,
    and top tool usage for one session.  Multiple ComparisonPanel
    widgets are shown side-by-side when the user enters comparison mode
    and selects 2+ sessions.

    The widget is focusable and supports keyboard interaction:
    - Enter triggers drill-down into the session's detailed view
    - Focus indicator visible via CSS ``:focus-within``
    - Tab/Arrow keys navigate between panels
    """

    DEFAULT_CSS = """
    ComparisonPanel {
        width: 1fr;
        height: auto;
        min-width: 20;
        min-height: 8;
        border: solid $primary;
        margin: 0 1 1 0;
        padding: 0 1;
    }

    ComparisonPanel:focus {
        border: solid $accent;
        background: $surface;
    }

    ComparisonPanel:focus-within {
        border: solid $accent;
    }

    ComparisonPanel.drill-down {
        width: 100%;
        height: 1fr;
        border: solid $warning;
    }
    """

    BINDINGS = [
        Binding("enter", "drill_down", "Drill Down", show=True),
    ]

    can_focus = True

    def __init__(
        self,
        session_index: int,
        session_label: str = "",
        call_count: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        avg_rtt: float | None = None,
        tool_usage: dict[str, int] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.session_index = session_index
        self.session_label = session_label
        self.call_count = call_count
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.avg_rtt = avg_rtt
        self.tool_usage = tool_usage or {}
        self._is_drill_down = False

    def action_drill_down(self) -> None:
        """Handle Enter key - trigger drill-down on the parent screen.

        When the user presses Enter on a focused ComparisonPanel,
        this method finds the parent screen and calls its
        ``_enter_drill_down`` method with this panel's session index.
        """
        if not self._is_drill_down:
            screen = self.screen
            if hasattr(screen, '_enter_drill_down'):
                screen._enter_drill_down(self.session_index)

    def set_drill_down(self, enabled: bool) -> None:
        """Toggle drill-down mode for this panel.

        When enabled, the panel adds the ``drill-down`` CSS class
        to signal it should fill the full width.

        Args:
            enabled: ``True`` to enter drill-down mode.
        """
        self._is_drill_down = enabled
        if enabled:
            self.add_class("drill-down")
        else:
            self.remove_class("drill-down")

    def render(self) -> Table | Text:
        """Render the comparison panel as a Rich Table with session metrics.

        Shows:
        - Session label and index
        - Call count
        - Input and output tokens (formatted with thousands separators)
        - Average round-trip time (or "N/A" if unavailable)
        - Top tools with their counts (up to 5)
        """
        table = Table(
            show_header=False,
            box=box.SIMPLE,
            padding=(0, 1),
            expand=True,
        )
        table.add_column("Metric", justify="left")
        table.add_column("Value", justify="right")

        # Session header row
        table.add_row(
            Text(self.session_label, style=_STYLE_BOLD),
            "",
            end_section=True,
        )

        # Call count
        table.add_row(
            Text("Calls", style=_STYLE_BOLD),
            Text(str(self.call_count), style=_STYLE_BOLD_ACCENT),
        )

        # Input tokens
        table.add_row(
            "Input Tokens",
            _format_thousands(self.input_tokens),
        )

        # Output tokens
        table.add_row(
            "Output Tokens",
            _format_thousands(self.output_tokens),
        )

        # Total tokens
        total_tokens = self.input_tokens + self.output_tokens
        table.add_row(
            Text("Total Tokens", style=_STYLE_BOLD),
            _format_thousands(total_tokens),
        )

        # Average RTT
        if self.avg_rtt is not None:
            rtt_str = _format_duration(self.avg_rtt)
        else:
            rtt_str = "N/A"
        table.add_row(
            Text("Avg RTT", style=_STYLE_BOLD),
            rtt_str,
        )

        # Top tools (up to 5)
        if self.tool_usage:
            sorted_tools = sorted(
                self.tool_usage.items(), key=lambda x: -x[1]
            )[:5]
            # Add a spacer section
            table.add_row(
                Text("Top Tools", style=_STYLE_BOLD_ITALIC),
                "",
                end_section=True,
            )
            for tool_name, count in sorted_tools:
                table.add_row(
                    f"  {tool_name}",
                    str(count),
                )

        # Drill-down hint
        table.add_row(
            "",
            Text("[Enter] Drill Down", style=_STYLE_ITALIC_MUTED),
        )

        return table
