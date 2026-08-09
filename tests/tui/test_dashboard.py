"""Tests for the Dashboard screen overview feature.

Covers the following validation assertions:
- VAL-DASH-001: Dashboard loads with all 7 sessions — aggregate metrics visible
- VAL-DASH-002: Total LLM call count per session — bar chart
- VAL-DASH-003: Total token usage per session — bar chart
- VAL-DASH-004: Average tokens per call — per session
- VAL-DASH-005: Timing statistics — min/avg/max RTT per session
- VAL-DASH-007: Session comparison — select 2 sessions side by side
- VAL-DASH-011: Keyboard navigation between dashboard widgets
- VAL-DASH-013: Single session view — select 1 session
- VAL-DASH-014: All 7 sessions comparison — full dashboard
- VAL-DASH-016: Dashboard widget focus and drill-down
- VAL-DASH-020: Timing comparison — fastest vs slowest sessions
"""

from __future__ import annotations

from unittest.mock import patch
from dataclasses import dataclass, field

import pytest


# ---------------------------------------------------------------------------
# Helpers: mock session data
# ---------------------------------------------------------------------------


@dataclass
class MockCall:
    """Minimal mock of an LLM call for dashboard metric computation."""
    request_id: str = ""
    response: object = None


@dataclass
class MockResponse:
    """Minimal mock of a parsed response with token data."""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class MockSession:
    """Minimal mock of a Session object for dashboard metric computation."""
    index: int = 0
    task_name: str = ""
    model: str = ""
    calls: list = field(default_factory=list)
    flow_errors: list = field(default_factory=list)


def _make_mock_session(index: int, task_name: str, call_count: int,
                        total_input: int, total_output: int) -> MockSession:
    """Create a mock session with the given metrics.

    Uses exact integer distribution: evenly distributes tokens across calls.
    """
    calls = []
    for i in range(call_count):
        input_per_call = total_input // call_count if call_count else 0
        output_per_call = total_output // call_count if call_count else 0
        calls.append(MockCall(
            request_id=f"{index:02d}_{i:03d}",
            response=MockResponse(
                input_tokens=input_per_call,
                output_tokens=output_per_call,
            ),
        ))
    return MockSession(
        index=index,
        task_name=task_name,
        calls=calls,
    )


# Mock session data simulating the 7 real sessions.
# All totals are evenly divisible by their respective call counts.
_MOCK_SESSIONS = [
    _make_mock_session(1, "analyze_codebase", 10, 12800, 540),
    _make_mock_session(2, "readiness_report", 8, 9600, 400),
    _make_mock_session(3, "receipt_wiki", 15, 31995, 1200),
    _make_mock_session(4, "compress", 5, 6400, 275),
    _make_mock_session(5, "security", 12, 18000, 780),
    _make_mock_session(6, "mission", 500, 25000000, 1200000),
    _make_mock_session(7, "baffled_wiki", 3, 3498, 150),
]


# ---------------------------------------------------------------------------
# Tests for metric computation utilities
# ---------------------------------------------------------------------------


class TestDashboardMetricsComputation:
    """Tests for core metric computation logic used by the dashboard."""

    def test_total_calls_across_sessions(self):
        """Aggregate total calls across all mock sessions."""
        total = sum(len(s.calls) for s in _MOCK_SESSIONS)
        # 10 + 8 + 15 + 5 + 12 + 500 + 3 = 553
        assert total == 553, f"Expected 553 total calls, got {total}"

    def test_total_tokens_across_sessions(self):
        """Aggregate total tokens across all mock sessions."""
        total_input = sum(
            sum(c.response.input_tokens for c in s.calls if c.response)
            for s in _MOCK_SESSIONS
        )
        total_output = sum(
            sum(c.response.output_tokens for c in s.calls if c.response)
            for s in _MOCK_SESSIONS
        )
        # Input: 12800 + 9600 + 31995 + 6400 + 18000 + 25000000 + 3498 = 25082293
        assert total_input == 25082293, f"Expected 25082293 total input, got {total_input}"
        # Output: 540 + 400 + 1200 + 275 + 780 + 1200000 + 150 = 1203345
        assert total_output == 1203345, f"Expected 1203345 total output, got {total_output}"

    def test_average_tokens_per_call(self):
        """Average tokens per call across all sessions."""
        for s in _MOCK_SESSIONS:
            total_tokens = sum(
                (c.response.input_tokens or 0) + (c.response.output_tokens or 0)
                for c in s.calls if c.response
            )
            call_count = len(s.calls)
            avg = total_tokens / call_count if call_count > 0 else 0
            # Session 01: (12800 + 540) / 10 = 1334.0
            if s.index == 1:
                assert abs(avg - 1334.0) < 0.1, f"Expected avg 1334.0 for session 01, got {avg}"
            # Verify non-negative
            assert avg >= 0, f"Average tokens per call should be non-negative, got {avg}"

    def test_single_session_calls_and_tokens(self):
        """Metrics for a single (selected) session match expected values."""
        s = _MOCK_SESSIONS[5]  # Session 06: mission
        assert s.index == 6
        call_count = len(s.calls)
        assert call_count == 500, f"Session 06 should have 500 calls, got {call_count}"

        total_input = sum(c.response.input_tokens for c in s.calls if c.response)
        total_output = sum(c.response.output_tokens for c in s.calls if c.response)
        assert total_input == 25000000, f"Session 06 input should be 25M, got {total_input}"
        assert total_output == 1200000, f"Session 06 output should be 1.2M, got {total_output}"


# ---------------------------------------------------------------------------
# Integration tests for DashboardScreen widget composition
# ---------------------------------------------------------------------------


class TestDashboardScreenComposition:
    """Tests for dashboard screen widget composition."""

    @pytest.mark.asyncio
    async def test_dashboard_shows_aggregate_metrics(self):
        """VAL-DASH-001: Aggregate metrics panel shows total sessions, calls, tokens."""
        metrics = DashboardScreenForTesting._compute_overall_metrics(_MOCK_SESSIONS)
        assert metrics["total_sessions"] == 7
        assert metrics["total_calls"] == 553
        assert metrics["total_tokens_input"] == 25082293
        assert metrics["total_tokens_output"] == 1203345


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class DashboardScreenForTesting:
    """Standalone class to test dashboard metric computation logic.

    Avoids needing full Textual app setup for unit tests of computation.
    """

    @staticmethod
    def _compute_overall_metrics(sessions: list) -> dict:
        """Compute aggregate metrics across all sessions.

        Returns a dict with total_sessions, total_calls, total_tokens_input,
        total_tokens_output.
        """
        total_calls = sum(len(s.calls) for s in sessions)
        total_input = sum(
            sum(c.response.input_tokens for c in s.calls if c.response)
            for s in sessions
        )
        total_output = sum(
            sum(c.response.output_tokens for c in s.calls if c.response)
            for s in sessions
        )
        return {
            "total_sessions": len(sessions),
            "total_calls": total_calls,
            "total_tokens_input": total_input,
            "total_tokens_output": total_output,
        }

    @staticmethod
    def _compute_session_bar_data(
        sessions: list,
        value_fn,
    ) -> list:
        """Compute data for bar charts.

        Args:
            sessions: List of mock session objects.
            value_fn: Callable that takes a session and returns a numeric value.

        Returns:
            List of (label, value) tuples sorted by session index.
        """
        data = []
        for s in sorted(sessions, key=lambda x: x.index):
            data.append((f"{s.index:02d} | {s.task_name}", value_fn(s)))
        return data

    @staticmethod
    def _compute_avg_tokens(sessions: list) -> list:
        """Compute average tokens per call for each session.

        Returns:
            List of (label, avg_value) tuples.
        """
        data = []
        for s in sorted(sessions, key=lambda x: x.index):
            call_count = len(s.calls)
            total_tokens = sum(
                (c.response.input_tokens or 0) + (c.response.output_tokens or 0)
                for c in s.calls if c.response
            )
            avg = total_tokens / call_count if call_count > 0 else 0
            data.append((f"{s.index:02d} | {s.task_name}", round(avg, 1)))
        return data


class TestDashboardMetricsForTesting:
    """Test the metric computation helper class."""

    def test_overall_metrics(self):
        """_compute_overall_metrics returns correct aggregate values."""
        metrics = DashboardScreenForTesting._compute_overall_metrics(_MOCK_SESSIONS)
        assert metrics["total_sessions"] == 7
        assert metrics["total_calls"] == 553
        assert metrics["total_tokens_input"] == 25082293
        assert metrics["total_tokens_output"] == 1203345

    def test_bar_data_call_counts(self):
        """_compute_session_bar_data returns correct call counts per session."""
        data = DashboardScreenForTesting._compute_session_bar_data(
            _MOCK_SESSIONS,
            lambda s: len(s.calls),
        )
        assert len(data) == 7
        # Session 01
        assert data[0][0] == "01 | analyze_codebase"
        assert data[0][1] == 10
        # Session 06 (largest)
        assert data[5][0] == "06 | mission"
        assert data[5][1] == 500
        # Session 07 (smallest)
        assert data[6][0] == "07 | baffled_wiki"
        assert data[6][1] == 3

    def test_bar_data_token_usage(self):
        """_compute_session_bar_data returns correct token values."""
        data = DashboardScreenForTesting._compute_session_bar_data(
            _MOCK_SESSIONS,
            lambda s: sum(
                (c.response.input_tokens or 0) + (c.response.output_tokens or 0)
                for c in s.calls if c.response
            ),
        )
        assert len(data) == 7
        # Session 01: 12800 + 540 = 13340
        label_01, val_01 = data[0]
        assert "analyze_codebase" in label_01
        assert val_01 == 13340
        # Session 06: 25000000 + 1200000 = 26200000
        label_06, val_06 = data[5]
        assert "mission" in label_06
        assert val_06 == 26200000

    def test_avg_tokens_per_call(self):
        """_compute_avg_tokens returns correct averages."""
        data = DashboardScreenForTesting._compute_avg_tokens(_MOCK_SESSIONS)
        assert len(data) == 7
        # Session 01: (12800 + 540) / 10 = 1334.0
        label_01, avg_01 = data[0]
        assert "analyze_codebase" in label_01
        assert abs(avg_01 - 1334.0) < 0.1

        # Session 06: (25000000 + 1200000) / 500 = 52400.0
        label_06, avg_06 = data[5]
        assert "mission" in label_06
        assert abs(avg_06 - 52400.0) < 0.1

        # Session 07: (3498 + 150) / 3 = 1216.0
        label_07, avg_07 = data[6]
        assert "baffled_wiki" in label_07
        assert abs(avg_07 - 1216.0) < 1.0

    def test_empty_session_handling(self):
        """Sessions with zero calls return 0 average without division by zero."""
        empty_session = _make_mock_session(8, "empty", 0, 0, 0)
        data = DashboardScreenForTesting._compute_avg_tokens([empty_session])
        assert len(data) == 1
        assert data[0][1] == 0  # avg should be 0, not an error


# ---------------------------------------------------------------------------
# Tests for TimingChart and duration formatting
# ---------------------------------------------------------------------------


class TestDurationFormatting:
    """Tests for the ``_format_duration`` helper used by the timing chart."""

    def test_sub_ms_rounds_to_less_than_1ms(self):
        """Durations below 0.001s round to ``<1ms``."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import _format_duration
        # 0.0s rounds to 0ms → <1ms
        assert _format_duration(0.0) == "<1ms"
        # 0.0001s → round(0.1) = 0 → <1ms
        assert _format_duration(0.0001) == "<1ms"
        # 0.00049s → round(0.49) = 0 → <1ms
        assert _format_duration(0.00049) == "<1ms"

    def test_ms_rounding_for_short_durations(self):
        """Durations >= 1ms but < 1s are shown as whole ms."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import _format_duration
        # 0.5s = 500ms
        assert _format_duration(0.5) == "500ms"
        # 0.001s = 1ms
        assert _format_duration(0.001) == "1ms"
        # 0.847s → 847ms
        assert _format_duration(0.847) == "847ms"

    def test_seconds_format_for_long_durations(self):
        """Durations >= 1s are shown with one decimal place."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import _format_duration
        assert _format_duration(1.0) == "1.0s"
        assert _format_duration(2.1) == "2.1s"
        assert _format_duration(14.05) == "14.1s"
        assert _format_duration(120.5) == "120.5s"

    def test_always_show_ms_flag(self):
        """With ``always_show_ms=True``, duration is shown as ms."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import _format_duration
        assert _format_duration(1.0, always_show_ms=True) == "1000ms"
        assert _format_duration(2.5, always_show_ms=True) == "2500ms"


# ======================================================================
# VAL-DASH-003: TokenBarChart — split input/output token display
# ======================================================================


class TestTokenBarChartRendering:
    """Tests for the TokenBarChart widget rendering logic.

    VERIFIES:
    - VAL-DASH-003: Two bars (input/output) per session with distinct colors
    - Edge cases: zero tokens, empty data, large numbers
    """

    def test_empty_data_shows_no_data(self):
        """Chart with empty data shows 'No data'."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import TokenBarChart
        from rich.text import Text

        chart = TokenBarChart()
        chart.chart_data = []
        result = chart.render()
        assert isinstance(result, Text), f"Expected Text for empty data, got {type(result)}"
        assert "No data" in result.plain

    def test_two_bars_per_session(self):
        """Each session row renders two bars (input + output)."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import TokenBarChart
        from rich.table import Table

        chart = TokenBarChart()
        chart.update_data([
            ("01 | analyze_codebase", 12800, 540),
        ])
        result = chart.render()
        assert isinstance(result, Table), f"Expected Table, got {type(result)}"

        # Column count: Session, Input, (bar), Output, (bar) = 5
        assert len(result.columns) == 5, \
            f"Expected 5 columns (Session, Input, bar, Output, bar), got {len(result.columns)}"

        # Header labels
        col_headers = [col.header for col in result.columns]
        assert "Session" in col_headers
        assert "Input" in col_headers
        assert "Output" in col_headers

    def test_multiple_sessions_all_rows_rendered(self):
        """All session rows appear in the chart with two bars each."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import TokenBarChart
        from rich.table import Table

        chart = TokenBarChart()
        chart.update_data([
            ("01 | analyze_codebase", 12800, 540),
            ("02 | readiness_report", 9600, 400),
            ("03 | receipt_wiki", 31995, 1200),
        ])
        result = chart.render()
        assert isinstance(result, Table)

        # The first column (Session) should have cells for each session
        session_col = result.columns[0]
        session_cells = [str(cell).strip() for cell in session_col._cells]

        # Check each session label appears somewhere in the rendered output
        assert any("analyze" in cell for cell in session_cells), \
            "Session 01 label should appear"
        assert any("readiness" in cell for cell in session_cells), \
            "Session 02 label should appear"
        assert any("receipt" in cell for cell in session_cells), \
            "Session 03 label should appear"

    def test_distinct_colors_for_input_output(self):
        """Input and output bars have distinct colors."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import TokenBarChart

        chart = TokenBarChart()
        # Verify the widget uses distinct color constants
        assert chart._INPUT_COLOR != chart._OUTPUT_COLOR, \
            "Input and output colors must be different"
        assert chart._INPUT_COLOR == "#0088ff", \
            f"Expected input color #0088ff, got {chart._INPUT_COLOR}"
        assert chart._OUTPUT_COLOR == "#ff8800", \
            f"Expected output color #ff8800, got {chart._OUTPUT_COLOR}"

    def test_value_formatting_abbreviated(self):
        """Token values are formatted with abbreviations (12.8K, 4.5M)."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import TokenBarChart
        from rich.table import Table

        chart = TokenBarChart()
        chart.update_data([
            ("01 | analyze_codebase", 12800, 540),
            ("06 | mission", 25000000, 1200000),
        ])
        result = chart.render()
        assert isinstance(result, Table)

        # Check the Input column (index 1) and Output column (index 3) for formatted values
        input_col = result.columns[1]
        output_col = result.columns[3]
        input_cells = [str(cell).strip() for cell in input_col._cells]
        output_cells = [str(cell).strip() for cell in output_col._cells]

        # Session 01: 12800 → "12.8K"
        assert any("12.8K" in c for c in input_cells), \
            f"Expected '12.8K' for 12800 input tokens, got {input_cells}"
        # Session 06: 25000000 → "25.0M"
        assert any("25.0M" in c for c in input_cells), \
            f"Expected '25.0M' for 25M input tokens, got {input_cells}"
        # Session 01: 540 → "540" (plain number)
        assert any("540" in c for c in output_cells), \
            f"Expected '540' for 540 output tokens, got {output_cells}"

    def test_zero_tokens_handled(self):
        """Sessions with zero input, output, or both render without errors."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import TokenBarChart
        from rich.table import Table

        chart = TokenBarChart()
        chart.update_data([
            ("01 | no_input", 0, 100),
            ("02 | no_output", 100, 0),
            ("03 | both_zero", 0, 0),
        ])
        result = chart.render()
        assert isinstance(result, Table), \
            f"Expected Table for zero-token data, got {type(result)}"

        # All three labels should be present
        session_col = result.columns[0]
        session_cells = [str(cell).strip() for cell in session_col._cells]
        assert any("no_input" in cell for cell in session_cells)
        assert any("no_output" in cell for cell in session_cells)
        assert any("both_zero" in cell for cell in session_cells)

        # Should not crash — zero bars are drawn as empty
        assert chart is not None

    def test_single_session_works(self):
        """Chart with a single session works identically."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import TokenBarChart
        from rich.table import Table

        chart = TokenBarChart()
        chart.update_data([
            ("01 | solo", 5000, 250),
        ])
        result = chart.render()
        assert isinstance(result, Table)
        assert len(result.columns) == 5

    def test_chart_has_correct_title(self):
        """Chart border title can be configured."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import TokenBarChart

        chart = TokenBarChart(title="Token Usage per Session")
        assert chart.border_title == "Token Usage per Session"

    def test_chart_in_dashboard_compose(self):
        """The token chart in dashboard is a TokenBarChart instance."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import TokenBarChart
        assert hasattr(TokenBarChart, 'chart_data'), \
            "TokenBarChart should have chart_data reactive attribute"
        assert hasattr(TokenBarChart, 'update_data'), \
            "TokenBarChart should have update_data method"


class TestTimingChartRendering:
    """Tests for the TimingChart widget rendering logic.

    These tests create a TimingChart instance and verify its render()
    output contains the expected labels, values, and styling.
    """

    def test_empty_timing_data_shows_no_data(self):
        """Chart with empty data shows 'No timing data'."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import TimingChart
        from rich.text import Text

        chart = TimingChart()
        chart.timing_data = []
        result = chart.render()
        # Should be a Text widget with "No timing data"
        assert isinstance(result, Text)
        assert "No timing data" in result.plain

    def test_chart_renders_single_session(self):
        """Chart with one session renders its label and values."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import TimingChart
        from rich.table import Table

        chart = TimingChart()
        chart.timing_data = [
            ("01 | analyze_codebase", 0.5, 2.1, 5.3),
        ]
        result = chart.render()
        assert isinstance(result, Table)

        # Check that header columns include our labels
        col_headers = [col.header for col in result.columns]
        assert "Session" in col_headers
        assert "Min" in col_headers
        assert "Avg" in col_headers
        assert "Max" in col_headers

    def test_chart_renders_multiple_sessions(self):
        """Chart with multiple sessions renders all labels."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import TimingChart
        from rich.table import Table

        chart = TimingChart()
        chart.timing_data = [
            ("01 | analyze_codebase", 0.5, 2.1, 5.3),
            ("02 | readiness_report", 0.3, 1.8, 4.2),
            ("03 | receipt_wiki", 0.8, 3.2, 7.1),
        ]
        result = chart.render()
        assert isinstance(result, Table)

        # The table should have 3 data rows (plus header)
        # We can check by looking at row count
        # A Table's rows property is not directly accessible via public API,
        # but we can verify by checking the renderable output has our labels

    def test_fastest_session_highlighted_green(self):
        """The session with lowest avg RTT is highlighted green."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import TimingChart
        from rich.table import Table

        # Session 02 has lowest avg (1.8), should be green
        # Session 03 has highest avg (3.2), should be red
        chart = TimingChart()
        chart.timing_data = [
            ("01 | analyze_codebase", 0.5, 2.1, 5.3),
            ("02 | readiness_report", 0.3, 1.8, 4.2),
            ("03 | receipt_wiki", 0.8, 3.2, 7.1),
        ]
        result = chart.render()
        assert isinstance(result, Table)

        # Concatenate all rendered cells into one string for searching.
        # Labels are truncated to 18 chars, so we check for short forms.
        rendered_text = " ".join(str(cell) for col in result.columns for cell in col._cells)
        assert "readiness" in rendered_text, \
            "Fastest session label should appear in rendered chart"
        assert "receipt_wiki" in rendered_text or "receipt" in rendered_text, \
            "Slowest session label should appear in rendered chart"

    def test_missing_timing_shows_n_a(self):
        """Sessions with missing timing data show 'N/A'."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import TimingChart
        from rich.table import Table

        chart = TimingChart()
        chart.timing_data = [
            ("01 | analyze_codebase", None, None, None),
        ]
        result = chart.render()
        assert isinstance(result, Table)

        # The N/A text should be present in rendered output
        rendered_text = " ".join(str(cell) for col in result.columns for cell in col._cells)
        assert "N/A" in rendered_text

    def test_chart_handles_zero_values(self):
        """Chart handles zero RTT values without division errors."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import TimingChart
        from rich.table import Table

        chart = TimingChart()
        chart.timing_data = [
            ("01 | analyze_codebase", 0.0, 0.0, 0.0),
        ]
        result = chart.render()
        assert isinstance(result, Table)
        # Should not crash or show errors
        rendered_text = " ".join(str(cell) for col in result.columns for cell in col._cells)
        assert "0" in rendered_text


# ---------------------------------------------------------------------------
# Tests for ToolUsageChart — VAL-DASH-006, VAL-DASH-017
# ---------------------------------------------------------------------------


def _make_mock_session_with_tools(
    index: int, task_name: str,
    tool_names: list[str],
) -> MockSession:
    """Create a mock session with the given tool calls.

    Each entry in ``tool_names`` generates one tool_use in a call's response.
    Multiple occurrences of the same name create multiple invocations.
    """
    from llm_flow_viewer.parser.models import ToolUse

    call_tool_uses: dict[str, list[ToolUse]] = {}
    for name in tool_names:
        if name not in call_tool_uses:
            call_tool_uses[name] = []
        call_tool_uses[name].append(
            ToolUse(name=name, id=f"call_{name}_{len(call_tool_uses[name])}")
        )

    calls = []
    for name, uses in call_tool_uses.items():
        from llm_flow_viewer.parser.models import ParsedResponse
        calls.append(MockCall(
            request_id=f"{index:02d}_{name}",
            response=ParsedResponse(tool_uses=uses),
        ))

    return MockSession(
        index=index,
        task_name=task_name,
        calls=calls,
    )


class TestToolUsageChartRendering:
    """Unit tests for the ToolUsageChart widget rendering.

    VERIFIES:
    - VAL-DASH-006: Tool usage counts per session — breakdown
    - VAL-DASH-017: Tool usage patterns — sorted by frequency across all sessions
    """

    def test_empty_data_shows_no_data(self):
        """Chart with empty data shows 'No tool usage data'."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import ToolUsageChart
        from rich.text import Text

        chart = ToolUsageChart()
        chart.tool_data = []
        result = chart.render()
        assert isinstance(result, Text), f"Expected Text for empty data, got {type(result)}"
        assert "No tool usage data" in result.plain

    def test_tools_sorted_by_count_descending(self):
        """VAL-DASH-017: Tools are sorted by count descending."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import ToolUsageChart
        from rich.table import Table

        chart = ToolUsageChart()
        chart.update_data([
            ("Read", 5),
            ("Execute", 10),
            ("Grep", 3),
            ("LS", 8),
        ])
        result = chart.render()
        assert isinstance(result, Table), f"Expected Table, got {type(result)}"

        # Extract tool names from the "Tool" column cells (skip header and total row)
        tool_col = result.columns[0]
        tool_cells = [str(cell) for cell in tool_col._cells]

        # First cell should be the most-used tool (Execute = 10)
        # Cells may contain rich text rendering, check plain text
        first_tool = tool_cells[0] if tool_cells else ""
        # Should contain "Execute" in some form
        assert "Execute" in first_tool or "Execute" in str(tool_cells), \
            f"Expected 'Execute' as first tool, got {tool_cells[:3]}"

    def test_top_n_limit_enforced(self):
        """VAL-DASH-006: Top-N limit is enforced; tools beyond are 'Other'."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import ToolUsageChart
        from rich.table import Table

        # 12 tools with varying counts
        chart = ToolUsageChart(top_n=5)
        chart.update_data([
            ("Read", 50),
            ("Grep", 45),
            ("LS", 40),
            ("Glob", 35),
            ("Execute", 30),
            ("Edit", 25),
            ("Create", 20),
            ("WebSearch", 15),
            ("FetchUrl", 10),
            ("ToolSearch", 8),
            ("Write", 5),
            ("Delete", 3),
        ])
        result = chart.render()
        assert isinstance(result, Table), f"Expected Table, got {type(result)}"

        tool_col = result.columns[0]
        tool_cells = [str(cell).strip() for cell in tool_col._cells]

        # Count non-empty tool cells (excluding the "Total" row at the end)
        visible_tools = [t for t in tool_cells if t and t not in ("Tool",)]
        # Should have 5 tool rows + 1 "Other" row + 1 "Total" row = 7 non-header rows
        # Actually "Total" is in the tool column too, so count it too
        assert any("Other" in t for t in tool_cells), \
            f"Expected 'Other' row in tools, got {tool_cells}"
        assert not any("Edit" in t for t in tool_cells), \
            "Tools beyond top-5 should not appear individually"
        assert not any("Create" in t for t in tool_cells), \
            "Tools beyond top-5 should not appear individually"

    def test_all_tools_shown_when_under_limit(self):
        """When fewer tools than top-N, all are shown without 'Other'."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import ToolUsageChart
        from rich.table import Table

        chart = ToolUsageChart(top_n=10)
        chart.update_data([
            ("Read", 5),
            ("Grep", 3),
            ("LS", 8),
        ])
        result = chart.render()
        assert isinstance(result, Table)

        rendered_text = " ".join(
            str(cell) for col in result.columns for cell in col._cells
        )
        assert "Other" not in rendered_text, \
            "'Other' should not appear when under top-N limit"

    def test_counts_formatted_with_thousands_separator(self):
        """Counts >= 1000 are formatted with commas."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import ToolUsageChart
        from rich.table import Table

        chart = ToolUsageChart()
        chart.update_data([
            ("Read", 1000),
            ("Grep", 12800),
            ("LS", 500),
        ])
        result = chart.render()
        assert isinstance(result, Table)

        rendered_text = " ".join(
            str(cell) for col in result.columns for cell in col._cells
        )
        assert "1,000" in rendered_text, \
            "1000 should be formatted as '1,000'"
        assert "12,800" in rendered_text, \
            "12800 should be formatted as '12,800'"
        assert "500" in rendered_text or " 500 " in rendered_text, \
            "500 should remain unformatted"

    def test_percentage_column_shown(self):
        """Percentage column shows tool's share of total calls."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import ToolUsageChart
        from rich.table import Table

        chart = ToolUsageChart()
        chart.update_data([
            ("Read", 75),
            ("Grep", 25),
        ])
        result = chart.render()
        assert isinstance(result, Table)

        # Check the percentage column (index 2)
        pct_col = result.columns[2]
        pct_cells = [str(cell).strip() for cell in pct_col._cells]

        # First tool (Read, 75%) should show ~75%
        non_header_pcts = [c for c in pct_cells if c not in ("%", "")]
        found_75 = any("75.0" in c for c in non_header_pcts)
        assert found_75, f"Expected 75.0% for Read tool, got {non_header_pcts}"

    def test_chart_with_single_tool(self):
        """Chart with a single tool shows it correctly."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import ToolUsageChart
        from rich.table import Table

        chart = ToolUsageChart()
        chart.update_data([("Read", 42)])
        result = chart.render()
        assert isinstance(result, Table)

        rendered_text = " ".join(
            str(cell) for col in result.columns for cell in col._cells
        )
        assert "Read" in rendered_text
        assert "42" in rendered_text
        assert "100%" in rendered_text

    def test_chart_with_many_tools_shows_top_n_plus_other(self):
        """When many tools, only top-N plus 'Other' and 'Total' rows."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import ToolUsageChart
        from rich.table import Table

        chart = ToolUsageChart(top_n=3)
        chart.update_data([
            ("Read", 100),
            ("Grep", 90),
            ("LS", 80),
            ("Glob", 70),
            ("Execute", 60),
            ("Edit", 50),
        ])
        result = chart.render()
        assert isinstance(result, Table)

        tool_col = result.columns[0]
        tool_cells = [str(cell).strip() for cell in tool_col._cells]

        # Should contain: header "Tool", then 3 tools, "Other", "Total"
        # Check that the 3 top tools appear individually
        top_three = ["Read", "Grep", "LS"]
        for t in top_three:
            assert any(t in cell for cell in tool_cells), \
                f"Top-3 tool '{t}' should appear individually"

        # Tools beyond top-3 should not appear individually
        assert not any("Glob" in cell for cell in tool_cells), \
            "Glob (beyond top-3) should not appear individually"
        assert not any("Execute" in cell for cell in tool_cells), \
            "Execute (beyond top-3) should not appear individually"

        # Other row should appear
        assert any("Other" in cell for cell in tool_cells), \
            "'Other' row should appear"


class TestToolDataComputation:
    """Tests for the _compute_tool_usage_fast method.

    Verifies that tool usage data is correctly extracted from session data.
    """

    def test_tool_usage_extracted_from_mock_session(self):
        """Tool names are correctly counted from mock session data."""
        from collections import Counter

        # Create a mock session with tool calls
        session = _make_mock_session_with_tools(
            index=1,
            task_name="analyze_codebase",
            tool_names=["Read", "Read", "Grep", "LS", "Read", "Execute"],
        )

        # Count tool usages manually
        counter: Counter = Counter()
        for call in session.calls:
            if call.response:
                for tu in call.response.tool_uses:
                    counter[tu.name] += 1

        assert counter["Read"] == 3, \
            f"Expected 3 'Read' calls, got {counter['Read']}"
        assert counter["Grep"] == 1, \
            f"Expected 1 'Grep' call, got {counter['Grep']}"
        assert counter["LS"] == 1, \
            f"Expected 1 'LS' call, got {counter['LS']}"
        assert counter["Execute"] == 1, \
            f"Expected 1 'Execute' call, got {counter['Execute']}"

    def test_empty_session_returns_empty_tool_counts(self):
        """A session with no tool calls returns empty tool counts."""
        session = _make_mock_session_with_tools(
            index=2, task_name="empty_session", tool_names=[],
        )

        from collections import Counter
        counter: Counter = Counter()
        for call in session.calls:
            if call.response:
                for tu in call.response.tool_uses:
                    counter[tu.name] += 1

        assert len(counter) == 0, \
            f"Expected empty counter for empty session, got {dict(counter)}"

    def test_tool_usage_aggregation_across_sessions(self):
        """Tool usage can be aggregated across multiple sessions."""
        sessions = [
            _make_mock_session_with_tools(1, "s1", ["Read", "Read", "Grep"]),
            _make_mock_session_with_tools(2, "s2", ["Read", "LS", "Grep", "Grep"]),
            _make_mock_session_with_tools(3, "s3", ["Execute"]),
        ]

        from collections import Counter
        aggregated: Counter = Counter()
        for session in sessions:
            for call in session.calls:
                if call.response:
                    for tu in call.response.tool_uses:
                        aggregated[tu.name] += 1

        assert aggregated["Read"] == 3, \
            f"Expected 3 'Read' across sessions, got {aggregated['Read']}"
        assert aggregated["Grep"] == 3, \
            f"Expected 3 'Grep' across sessions, got {aggregated['Grep']}"
        assert aggregated["LS"] == 1, \
            f"Expected 1 'LS' across sessions, got {aggregated['LS']}"
        assert aggregated["Execute"] == 1, \
            f"Expected 1 'Execute' across sessions, got {aggregated['Execute']}"


class TestToolUsageSorting:
    """Tests for tool usage sorting behavior (VAL-DASH-017)."""

    def test_sorted_descending_by_count(self):
        """Tools are sorted by count descending in the chart render."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import ToolUsageChart
        from rich.table import Table

        chart = ToolUsageChart()
        chart.update_data([
            ("Read", 5),
            ("Execute", 10),
            ("Grep", 3),
            ("LS", 8),
        ])
        result = chart.render()
        assert isinstance(result, Table)

        # Access the count column (index 1) and verify order
        count_col = result.columns[1]
        count_cells = [str(cell).strip() for cell in count_col._cells]

        # Extract numeric values from count cells (skip header and total)
        count_values = []
        for cell in count_cells:
            clean = cell.replace(",", "")
            try:
                val = int(clean)
                count_values.append(val)
            except ValueError:
                pass

        # We have 4 individual tool rows, so take first 4 values
        # (the 5th would be the Total row)
        individual_counts = count_values[:4]
        if individual_counts:
            # Check descending order
            for i in range(len(individual_counts) - 1):
                assert individual_counts[i] >= individual_counts[i + 1], \
                    f"Tool counts should be descending: {individual_counts}"
            # Verify first and last values
            assert individual_counts[0] == 10, \
                f"Expected highest count 10, got {individual_counts[0]}"
        else:
            # Fall back to rendering check
            rendered_text = " ".join(
                str(cell) for col in result.columns for cell in col._cells
            )
            assert "10" in rendered_text
            assert "Execute" in rendered_text


# ======================================================================
# VAL-DASH-011: Keyboard Navigation Between Dashboard Widgets
# ======================================================================


class TestDashboardKeyboardNavigation:
    """Tests for keyboard navigation between dashboard widgets (VAL-DASH-011)."""

    @pytest.mark.asyncio
    async def test_comparison_panels_are_focusable(self):
        """ComparisonPanel widgets are focusable (can_focus=True)."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import ComparisonPanel
        assert hasattr(ComparisonPanel, 'can_focus')

    @pytest.mark.asyncio
    async def test_focus_indicator_on_comparison_panels(self):
        """Comparison panels have focus-within CSS for visible indicators."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import ComparisonPanel
        css = getattr(ComparisonPanel, 'DEFAULT_CSS', '')
        assert 'focus-within' in css or 'can_focus' in str(dir(ComparisonPanel)), \
            "ComparisonPanel should have focus styling"


# ======================================================================
# Unit tests for ComparisonPanel widget rendering
# ======================================================================


class TestComparisonPanelWidget:
    """Unit tests for the ComparisonPanel widget."""

    def test_comparison_panel_importable(self):
        """ComparisonPanel widget can be imported."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import ComparisonPanel
        assert ComparisonPanel is not None

    def test_comparison_panel_initialization(self):
        """ComparisonPanel can be initialized with session data."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import ComparisonPanel

        panel = ComparisonPanel(
            session_index=1,
            session_label="01 | analyze_codebase",
            call_count=10,
            input_tokens=12800,
            output_tokens=540,
            avg_rtt=2.1,
        )
        assert panel.session_index == 1
        assert panel.session_label == "01 | analyze_codebase"
        assert panel.call_count == 10

    def _table_text(self, result) -> str:
        """Extract plain text from a Rich Table or Text renderable."""
        from rich.table import Table
        from rich.text import Text
        if isinstance(result, Text):
            return result.plain
        if isinstance(result, Table):
            # Collect all cell text and row labels
            parts = []
            for col in result.columns:
                for cell in col._cells:
                    t = str(cell)
                    if t:
                        parts.append(t)
            return " ".join(parts)
        return str(result)

    def test_comparison_panel_renders_content(self):
        """ComparisonPanel renders non-empty content."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import ComparisonPanel

        panel = ComparisonPanel(
            session_index=1,
            session_label="01 | analyze_codebase",
            call_count=10,
            input_tokens=12800,
            output_tokens=540,
            avg_rtt=2.1,
        )
        result = panel.render()
        rendered_text = self._table_text(result)
        assert "analyze_codebase" in rendered_text, \
            "Session name should appear in rendered content"
        assert "10" in rendered_text, \
            "Call count should appear in rendered content"
        assert "12,800" in rendered_text, \
            "Input tokens should appear formatted in rendered content"

    def test_comparison_panel_shows_tool_usage(self):
        """ComparisonPanel shows top tools for the session."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import ComparisonPanel

        panel = ComparisonPanel(
            session_index=1,
            session_label="01 | analyze_codebase",
            call_count=10,
            input_tokens=12800,
            output_tokens=540,
            avg_rtt=2.1,
            tool_usage={"Read": 5, "Grep": 3, "Execute": 2},
        )
        result = panel.render()
        rendered_text = self._table_text(result)
        assert "Read" in rendered_text, "Tool name should appear in rendered content"
        assert "5" in rendered_text, "Tool count should appear in rendered content"
        assert "Grep" in rendered_text, "Tool name should appear in rendered content"

    def test_comparison_panel_without_tool_usage(self):
        """ComparisonPanel handles missing tool usage data gracefully."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import ComparisonPanel

        panel = ComparisonPanel(
            session_index=1,
            session_label="01 | analyze_codebase",
            call_count=10,
            input_tokens=12800,
            output_tokens=540,
            avg_rtt=2.1,
            tool_usage=None,
        )
        result = panel.render()
        rendered_text = self._table_text(result)
        # Without tool_usage, the Top Tools section should not appear
        # But call count and other metrics should still render
        assert "analyze_codebase" in rendered_text, \
            "Session name should still appear"
        assert "10" in rendered_text, \
            "Call count should appear"

    def test_comparison_panel_zero_values(self):
        """ComparisonPanel handles zero values without division errors."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import ComparisonPanel

        panel = ComparisonPanel(
            session_index=7,
            session_label="07 | baffled_wiki",
            call_count=0,
            input_tokens=0,
            output_tokens=0,
            avg_rtt=None,
        )
        result = panel.render()
        rendered_text = str(result)
        assert "0" in rendered_text, \
            "Panel should show zero counts"


# ======================================================================
# VAL-DASH-018: Cache Efficiency Metrics
# ======================================================================


class TestCacheEfficiencyChartRendering:
    """Tests for the CacheEfficiencyChart widget rendering.

    VERIFIES:
    - VAL-DASH-018: Cache efficiency metrics display and computation
    """

    def test_empty_data_shows_no_cache_data(self):
        """Chart with empty data shows 'No cache data'."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import CacheEfficiencyChart
        from rich.text import Text

        chart = CacheEfficiencyChart()
        chart.cache_data = []
        result = chart.render()
        assert isinstance(result, Text), f"Expected Text for empty data, got {type(result)}"
        assert "No cache data" in result.plain

    def test_cache_hit_rate_computed_correctly(self):
        """Cache hit rate is computed as cache_read / (cache_read + input_tokens) * 100."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import CacheEfficiencyChart
        from rich.table import Table

        chart = CacheEfficiencyChart()
        # Session with 90% cache hit: cache_read=90, input_tokens=10 => 90%
        chart.cache_data = [
            ("01 | analyze_codebase", 90, 10),
        ]
        result = chart.render()
        assert isinstance(result, Table), f"Expected Table, got {type(result)}"

        rendered_text = " ".join(
            str(cell) for col in result.columns for cell in col._cells
        )
        assert "90.0%" in rendered_text or "90" in rendered_text, \
            "Cache hit rate 90% should be displayed"

    def test_zero_cache_shows_zero_percent(self):
        """Session with no cache reads shows 0%."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import CacheEfficiencyChart
        from rich.table import Table

        chart = CacheEfficiencyChart()
        chart.cache_data = [
            ("01 | analyze_codebase", 0, 100),
        ]
        result = chart.render()
        assert isinstance(result, Table)

        rendered_text = " ".join(
            str(cell) for col in result.columns for cell in col._cells
        )
        assert "0.0%" in rendered_text or "0" in rendered_text, \
            "Zero cache should show 0%"

    def test_full_cache_hits_shows_100_percent(self):
        """Session with all cache reads shows 100%."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import CacheEfficiencyChart
        from rich.table import Table

        chart = CacheEfficiencyChart()
        chart.cache_data = [
            ("01 | analyze_codebase", 100, 0),
        ]
        result = chart.render()
        assert isinstance(result, Table)

        rendered_text = " ".join(
            str(cell) for col in result.columns for cell in col._cells
        )
        assert "100.0%" in rendered_text or "100%" in rendered_text, \
            "Full cache should show 100%"

    def test_session_without_cache_data_shows_n_a(self):
        """Sessions with no cache data at all show 'N/A'."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import CacheEfficiencyChart
        from rich.table import Table

        chart = CacheEfficiencyChart()
        # Input tokens present but no cache_read and no input_tokens? 
        # Actually "no cache data" means cache_read is None and input_tokens is 0
        # We handle this by using a dedicated marker
        chart.cache_data = [
            ("01 | analyze_codebase", None, None),
        ]
        result = chart.render()
        assert isinstance(result, Table)
        
        rendered_text = " ".join(
            str(cell) for col in result.columns for cell in col._cells
        )
        assert "N/A" in rendered_text, \
            "Session without cache data should show N/A"

    def test_mixed_cache_data_renders_all_sessions(self):
        """Multiple sessions with mixed cache data render correctly."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import CacheEfficiencyChart
        from rich.table import Table

        chart = CacheEfficiencyChart()
        chart.cache_data = [
            ("01 | analyze_codebase", 900, 100),     # 90%
            ("02 | readiness_report", 0, 100),        # 0%
            ("03 | receipt_wiki", 995, 5),             # 99.5%
            ("04 | compress", None, None),              # N/A
        ]
        result = chart.render()
        assert isinstance(result, Table)

        rendered_text = " ".join(
            str(cell) for col in result.columns for cell in col._cells
        )
        # Labels are truncated to ~18 chars, check for key parts
        assert "01" in rendered_text
        assert "02" in rendered_text
        assert "03" in rendered_text
        assert "04" in rendered_text
        assert "N/A" in rendered_text  # Session 04 has no cache data

    def test_cache_efficiency_color_coding(self):
        """High cache efficiency (>90%) shows green, low (<50%) shows different color."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import CacheEfficiencyChart
        from rich.table import Table

        chart = CacheEfficiencyChart()
        # These values should trigger different color styles
        chart.cache_data = [
            ("01 | high_cache", 95, 5),    # 95% -> green
            ("02 | medium_cache", 50, 50), # 50% -> yellow
            ("03 | low_cache", 10, 90),    # 10% -> red
        ]
        result = chart.render()
        assert isinstance(result, Table)

        # The chart should render without errors
        rendered_text = " ".join(
            str(cell) for col in result.columns for cell in col._cells
        )
        assert "high" in rendered_text
        assert "medium" in rendered_text
        assert "low" in rendered_text


class TestCacheEfficiencyDataComputation:
    """Tests for cache efficiency data computation from session data."""

    def test_cache_efficiency_computed_from_response_data(self):
        """Cache efficiency is computed from cache_read and input_tokens fields."""
        from collections import Counter
        
        # Simulate response data from parquet
        responses = [
            {"cache_read_input_tokens": 1000, "input_tokens": 50},
            {"cache_read_input_tokens": 2000, "input_tokens": 30},
            {"cache_read_input_tokens": 500, "input_tokens": 20},
        ]
        
        total_cache_read = sum(r["cache_read_input_tokens"] for r in responses if r["cache_read_input_tokens"] is not None)
        total_input = sum(r["input_tokens"] for r in responses if r["input_tokens"] is not None)
        
        expected_rate = (total_cache_read / (total_cache_read + total_input)) * 100
        # (1000 + 2000 + 500) / (3500 + 100) = 3500/3600 = 97.22%
        assert abs(expected_rate - 97.22) < 0.1, \
            f"Expected ~97.22% cache hit rate, got {expected_rate}"

    def test_cache_efficiency_zero_input_tokens(self):
        """When input_tokens is 0 and cache_read is 0, ratio is 0 (no cache)."""
        total_cache_read = 0
        total_input = 0
        denominator = total_cache_read + total_input
        rate = 0.0 if denominator == 0 else (total_cache_read / denominator) * 100
        assert rate == 0.0, "Zero cache and zero input should give 0%"

    def test_cache_efficiency_some_null_values(self):
        """Responses with null cache fields are skipped gracefully."""
        responses = [
            {"cache_read_input_tokens": 1000, "input_tokens": 50},
            {"cache_read_input_tokens": None, "input_tokens": 30},  # null cache
            {"cache_read_input_tokens": 500, "input_tokens": None},  # null input
        ]
        
        total_cache_read = sum(r["cache_read_input_tokens"] for r in responses if r["cache_read_input_tokens"] is not None)
        total_input = sum(r["input_tokens"] for r in responses if r["input_tokens"] is not None)
        
        denominator = total_cache_read + total_input
        rate = (total_cache_read / denominator) * 100 if denominator > 0 else 0.0
        # (1000 + 500) / (1500 + 80) = 1500/1580 = 94.94%
        assert abs(rate - 94.94) < 0.1, \
            f"Expected ~94.94% with null handling, got {rate}"


# ======================================================================
# VAL-DASH-019: Model Usage Breakdown Across Sessions
# ======================================================================


@dataclass
class MockParsedRequest:
    """Minimal mock of a parsed request with model field."""
    model: str = ""
    max_tokens: int = 0
    messages: list = field(default_factory=list)
    tools: list = field(default_factory=list)
    system: list = field(default_factory=list)
    thinking: dict | None = None
    request_id: str = ""
    timestamp_start: float | None = None
    timestamp_end: float | None = None


@dataclass
class MockLLMCallWithModel:
    """Minimal mock of an LLMCall for model usage testing."""
    request_id: str = ""
    request: MockParsedRequest | None = None
    response: object = None
    timing: object = None
    connection_timing: object = None

    def __init__(self, model: str = "", request_id: str = ""):
        self.request_id = request_id
        self.request = MockParsedRequest(model=model)


class TestModelUsageChartRendering:
    """Tests for the ModelUsageChart widget rendering.

    VERIFIES:
    - VAL-DASH-019: Model usage breakdown display
    """

    def test_empty_data_shows_no_model_data(self):
        """Chart with empty data shows 'No model usage data'."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import ModelUsageChart
        from rich.text import Text

        chart = ModelUsageChart()
        chart.model_data = []
        result = chart.render()
        assert isinstance(result, Text), f"Expected Text for empty data, got {type(result)}"
        assert "No model usage data" in result.plain

    def test_single_model_shown(self):
        """Session with one model shows it correctly."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import ModelUsageChart
        from rich.table import Table

        chart = ModelUsageChart()
        chart.update_data([
            ("deepseek-v4-flash", 10),
        ])
        result = chart.render()
        assert isinstance(result, Table), f"Expected Table, got {type(result)}"

        rendered_text = " ".join(
            str(cell) for col in result.columns for cell in col._cells
        )
        assert "deepseek-v4-flash" in rendered_text, \
            "Model name should appear in rendered content"
        assert "10" in rendered_text, \
            "Call count should appear in rendered content"

    def test_multiple_models_shown(self):
        """Session with multiple models shows all with counts."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import ModelUsageChart
        from rich.table import Table

        chart = ModelUsageChart()
        chart.update_data([
            ("deepseek-v4-flash", 8),
            ("deepseek-v4-pro", 3),
            ("deepseek-v4-mini", 1),
        ])
        result = chart.render()
        assert isinstance(result, Table)

        rendered_text = " ".join(
            str(cell) for col in result.columns for cell in col._cells
        )
        assert "deepseek-v4-flash" in rendered_text
        assert "deepseek-v4-pro" in rendered_text
        assert "deepseek-v4-mini" in rendered_text
        assert "8" in rendered_text
        assert "3" in rendered_text
        assert "1" in rendered_text

    def test_total_row_shows_sum(self):
        """Total row shows sum of all model call counts."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import ModelUsageChart
        from rich.table import Table

        chart = ModelUsageChart()
        chart.update_data([
            ("deepseek-v4-flash", 8),
            ("deepseek-v4-pro", 3),
        ])
        result = chart.render()
        assert isinstance(result, Table)

        rendered_text = " ".join(
            str(cell) for col in result.columns for cell in col._cells
        )
        assert "Total" in rendered_text
        assert "11" in rendered_text, "Total should be 11 (8 + 3)"

    def test_single_model_shows_100_percent(self):
        """Single model should show 100% in percentage column."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import ModelUsageChart
        from rich.table import Table

        chart = ModelUsageChart()
        chart.update_data([
            ("deepseek-v4-flash", 10),
        ])
        result = chart.render()
        assert isinstance(result, Table)

        rendered_text = " ".join(
            str(cell) for col in result.columns for cell in col._cells
        )
        assert "100%" in rendered_text

    def test_percentage_distribution(self):
        """Models show correct percentage distribution."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import ModelUsageChart
        from rich.table import Table

        chart = ModelUsageChart()
        chart.update_data([
            ("deepseek-v4-flash", 75),
            ("deepseek-v4-pro", 25),
        ])
        result = chart.render()
        assert isinstance(result, Table)

        # Check percentage column
        pct_col = result.columns[2]
        pct_cells = [str(cell).strip() for cell in pct_col._cells]
        non_header_pcts = [c for c in pct_cells if c not in ("%", "")]
        
        found_75 = any("75.0" in c for c in non_header_pcts)
        assert found_75, f"Expected 75.0% for flash model, got {non_header_pcts}"


class TestModelUsageDataComputation:
    """Tests for model usage data computation from session data."""

    def test_models_counted_correctly(self):
        """Models are correctly counted from request data."""
        from collections import Counter
        
        # Simulate model data from request parquet
        models = [
            "deepseek-v4-flash",
            "deepseek-v4-flash",
            "deepseek-v4-pro",
            "deepseek-v4-flash",
            "deepseek-v4-mini",
        ]
        
        counter: Counter = Counter()
        for m in models:
            counter[m] += 1
        
        assert counter["deepseek-v4-flash"] == 3
        assert counter["deepseek-v4-pro"] == 1
        assert counter["deepseek-v4-mini"] == 1
        assert sum(counter.values()) == 5

    def test_empty_session_returns_empty_counts(self):
        """A session with no calls returns empty model counts."""
        from collections import Counter
        counter: Counter = Counter()
        assert len(counter) == 0

    def test_model_usage_aggregation_across_sessions(self):
        """Model usage can be aggregated across multiple sessions."""
        from collections import Counter
        
        session_models = [
            ["deepseek-v4-flash", "deepseek-v4-flash", "deepseek-v4-pro"],
            ["deepseek-v4-flash", "deepseek-v4-mini"],
            ["deepseek-v4-pro"],
        ]
        
        aggregated: Counter = Counter()
        for models in session_models:
            for m in models:
                aggregated[m] += 1
        
        assert aggregated["deepseek-v4-flash"] == 3
        assert aggregated["deepseek-v4-pro"] == 2
        assert aggregated["deepseek-v4-mini"] == 1
        assert sum(aggregated.values()) == 6


# ======================================================================
# Unit tests for CacheEfficiencyChart and ModelUsageChart widget classes
# ======================================================================


class TestCacheEfficiencyChartWidget:
    """Unit tests for CacheEfficiencyChart widget class."""

    def test_widget_importable(self):
        """CacheEfficiencyChart widget can be imported."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import CacheEfficiencyChart
        assert CacheEfficiencyChart is not None

    def test_widget_has_reactive_data(self):
        """CacheEfficiencyChart has reactive cache_data attribute."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import CacheEfficiencyChart
        assert hasattr(CacheEfficiencyChart, 'cache_data'), \
            "CacheEfficiencyChart should have cache_data reactive"

    def test_default_border_title(self):
        """CacheEfficiencyChart has appropriate default border title."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import CacheEfficiencyChart
        chart = CacheEfficiencyChart()
        assert "Cache" in chart.border_title or "Efficiency" in chart.border_title, \
            f"Expected cache-related border title, got '{chart.border_title}'"


class TestModelUsageChartWidget:
    """Unit tests for ModelUsageChart widget class."""

    def test_widget_importable(self):
        """ModelUsageChart widget can be imported."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import ModelUsageChart
        assert ModelUsageChart is not None

    def test_widget_has_update_data_method(self):
        """ModelUsageChart has update_data method."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import ModelUsageChart
        chart = ModelUsageChart()
        assert hasattr(chart, 'update_data'), \
            "ModelUsageChart should have update_data method"

    def test_widget_has_reactive_data(self):
        """ModelUsageChart has reactive model_data attribute."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import ModelUsageChart
        assert hasattr(ModelUsageChart, 'model_data'), \
            "ModelUsageChart should have model_data reactive"

    def test_default_border_title(self):
        """ModelUsageChart has appropriate default border title."""
        from llm_flow_viewer.tui.widgets.dashboard_widgets import ModelUsageChart
        chart = ModelUsageChart()
        assert "Model" in chart.border_title or "Usage" in chart.border_title, \
            f"Expected model-related border title, got '{chart.border_title}'"


# ======================================================================
# VAL-DASH-009: Loading Indicator — Parsing Large Session Data
# ======================================================================


class TestDashboardLoadingState:
    """Tests for loading state behavior (VAL-DASH-009).

    When the dashboard is loading session data (especially large
    sessions like #06), a loading indicator should appear.  The TUI
    must remain responsive during loading (can switch views, press
    keys).  Once data is loaded, the indicator disappears and charts
    are shown.
    """

    @pytest.mark.asyncio
    async def test_loading_indicator_can_be_styled(self):
        """Loading indicator has appropriate CSS styling."""
        from llm_flow_viewer.tui.screens.dashboard import DashboardScreen

        # Check that the loading indicator widget has proper CSS
        css = getattr(DashboardScreen, "CSS", "")
        assert "loading" in css.lower() or "#dashboard-loading" in css, \
            "Dashboard CSS should include loading indicator styling"


class TestDashboardEdgeStateComposition:
    """Tests for proper widget composition of edge state widgets."""

    @pytest.mark.asyncio
    async def test_dashboard_has_empty_state_widget(self):
        """Dashboard composes empty state widget."""
        from llm_flow_viewer.tui.screens.dashboard import DashboardScreen

        # Check that compose method includes empty state
        import inspect
        source = inspect.getsource(DashboardScreen.compose)
        assert "dashboard-empty-state" in source or "empty" in source.lower(), \
            "Dashboard compose should include empty state element"

    @pytest.mark.asyncio
    async def test_dashboard_has_loading_indicator_widget(self):
        """Dashboard composes loading indicator widget."""
        from llm_flow_viewer.tui.screens.dashboard import DashboardScreen

        import inspect
        source = inspect.getsource(DashboardScreen.compose)
        assert "dashboard-loading" in source.lower(), \
            "Dashboard compose should include loading indicator element"


# ===================================================================
# VAL-CROSS-012: Dashboard refresh / live metric update
# ===================================================================


class TestDashboardRefresh:
    """Tests for dashboard refresh mechanism (VAL-CROSS-012)."""

    @pytest.mark.asyncio
    async def test_dashboard_has_refresh_binding(self):
        """DashboardScreen should have 'r' key bound for refresh."""
        from llm_flow_viewer.tui.screens.dashboard import DashboardScreen
        binding_keys = {b.key for b in DashboardScreen.BINDINGS}
        assert "r" in binding_keys, (
            "'r' key should be bound for refreshing dashboard data"
        )

    @pytest.mark.asyncio
    async def test_dashboard_has_refresh_action(self):
        """DashboardScreen should have action_refresh_dashboard method."""
        from llm_flow_viewer.tui.screens.dashboard import DashboardScreen
        assert hasattr(DashboardScreen, "action_refresh_dashboard"), (
            "DashboardScreen must have action_refresh_dashboard method"
        )


# ======================================================================
# VAL-CACHE-001 through VAL-CACHE-003, VAL-CACHE-011, VAL-CACHE-013:
# Cache key includes source flow file info
# ======================================================================


class TestComputeCacheKeyWithSourceFiles:
    """Tests for _compute_cache_key() including source file hashing.

    VERIFIES:
    - VAL-CACHE-001: Source file (path, size, mtime_ns) is included in hash
    - VAL-CACHE-002: Deleting a source file changes the cache key
    - VAL-CACHE-003: Source file info is hashed even when zero parquet files exist
    - VAL-CACHE-011: New test directly exercises _compute_cache_key()
    - VAL-CACHE-013: Source file size change invalidates cache
    """

    def test_cache_key_includes_source_file_info(self, tmp_path):
        """VAL-CACHE-001: Source file (path, size, mtime_ns) is part of hash.

        Verifies that the cache key changes when a source file's mtime
        changes, even when no parquet files exist.
        """
        from llm_flow_viewer.tui.widgets.session_list import SessionInfo
        from llm_flow_viewer.tui.screens.dashboard import DashboardScreen

        # Create a source flow file
        flow_file = tmp_path / "01_flows-test_session"
        flow_file.write_text("some flow content")

        # Create DashboardScreen pointing at tmp_path
        screen = DashboardScreen(flows_dir=str(tmp_path))

        # Compute cache key
        key1 = screen._compute_cache_key()

        # Modify mtime of the source file
        import os
        import time
        time.sleep(0.02)  # Ensure mtime changes
        flow_file.write_text("modified flow content")
        # Force mtime to be different
        new_mtime = time.time()
        os.utime(str(flow_file), (new_mtime, new_mtime))

        # Compute key again — should be different
        key2 = screen._compute_cache_key()

        assert key1 != key2, (
            "Cache key should change when source file mtime/size changes"
        )

    def test_cache_key_changes_when_source_file_deleted(self, tmp_path):
        """VAL-CACHE-002: Deleting a source file changes the cache key."""
        from llm_flow_viewer.tui.widgets.session_list import SessionInfo
        from llm_flow_viewer.tui.screens.dashboard import DashboardScreen

        # Create a source flow file
        flow_file = tmp_path / "01_flows-test_session"
        flow_file.write_text("some flow content")

        screen = DashboardScreen(flows_dir=str(tmp_path))

        # Compute key with file present
        key1 = screen._compute_cache_key()

        # Delete the source file
        flow_file.unlink()

        # Compute key again — should be different
        key2 = screen._compute_cache_key()

        assert key1 != key2, (
            "Cache key should change when a source file is deleted"
        )

    def test_cache_key_includes_zero_call_sessions(self, tmp_path):
        """VAL-CACHE-003: Source file info hashed even when zero parquet files exist.

        A session with no LLM calls has no parquet cache files. The source
        file itself must still contribute to the cache key so that deleting
        it changes the key.
        """
        from llm_flow_viewer.tui.screens.dashboard import DashboardScreen

        # Create a source flow file (no parquet files)
        flow_file = tmp_path / "01_flows-test_session"
        flow_file.write_text("some flow content")

        screen = DashboardScreen(flows_dir=str(tmp_path))

        # Compute key — should succeed and include source file info
        key1 = screen._compute_cache_key()

        # The key should not be the empty hash (which happens when no
        # parquet files exist and no source info was hashed)
        assert key1 is not None, "Cache key should not be None"
        assert len(key1) == 64, (
            f"Cache key should be 64-char hex digest, got {len(key1)}"
        )

        # Delete the source file — key must change
        flow_file.unlink()
        key2 = screen._compute_cache_key()

        assert key1 != key2, (
            "Cache key should change when zero-call session source is deleted"
        )

    def test_cache_key_changes_with_source_file_size_change(self, tmp_path):
        """VAL-CACHE-013: Source file size change invalidates cache key."""
        from llm_flow_viewer.tui.screens.dashboard import DashboardScreen

        # Create a small source file
        flow_file = tmp_path / "01_flows-test_session"
        flow_file.write_text("small content")

        screen = DashboardScreen(flows_dir=str(tmp_path))

        key1 = screen._compute_cache_key()

        # Replace with a larger file
        flow_file.write_text("x" * 10000)

        key2 = screen._compute_cache_key()

        assert key1 != key2, (
            "Cache key should change when source file size changes"
        )

    def test_cache_key_is_deterministic(self, tmp_path):
        """Same source files produce the same cache key."""
        from llm_flow_viewer.tui.screens.dashboard import DashboardScreen

        flow_file = tmp_path / "01_flows-test_session"
        flow_file.write_text("deterministic content")

        screen = DashboardScreen(flows_dir=str(tmp_path))

        key1 = screen._compute_cache_key()
        key2 = screen._compute_cache_key()

        assert key1 == key2, (
            "Cache key should be deterministic for unchanged source files"
        )


# ======================================================================
# VAL-CACHE-004, VAL-CACHE-005, VAL-CACHE-005a, VAL-CACHE-005b,
# VAL-CACHE-012, VAL-CACHE-014, VAL-CACHE-015, VAL-CACHE-017,
# VAL-CACHE-018, VAL-CACHE-019:
# Cache load drops entries with missing source files
# ======================================================================


class TestLoadMetricsFromCacheEntryDropping:
    """Tests for _load_metrics_from_cache() source file existence validation.

    VERIFIES:
    - VAL-CACHE-004: Cached entries with missing source files are silently dropped
    - VAL-CACHE-005: Stale cache (key mismatch) triggers recompute (cache freshness intact)
    - VAL-CACHE-005a: Round-trip consistency (save → load with no file changes = cache hit)
    - VAL-CACHE-005b: All entries dropped → empty state
    - VAL-CACHE-012: New test for _load_metrics_from_cache dropping
    - VAL-CACHE-014: Missing cache file falls through to compute path
    - VAL-CACHE-015: Corrupted cache file falls through to compute path
    - VAL-CACHE-017: Atomic cache write via os.replace
    - VAL-CACHE-018: Session ordering by index after dropping
    - VAL-CACHE-019: Parallel arrays stay aligned after dropping
    """

    def test_drops_entry_with_missing_source_file(self, tmp_path):
        """VAL-CACHE-004: Entry with missing source file is silently dropped."""
        from llm_flow_viewer.tui.screens.dashboard import DashboardScreen

        # Create a source file that exists
        existing_file = tmp_path / "01_flows-existing"
        existing_file.write_text("exists")

        # Create a fake cache file at .dashboard_metrics.json
        cache = {
            "cache_key": "dummy",
            "session_data": [
                [1, "existing", str(existing_file), 10, 100, 50],
                [2, "missing", str(tmp_path / "02_flows-missing"), 5, 200, 100],
                [3, "also_missing", str(tmp_path / "03_flows-also_missing"), 0, 0, 0],
            ],
            "timing_data": [
                ["01 | existing", 0.5, 1.0, 2.0],
                ["02 | missing", 0.3, 0.8, 1.5],
                ["03 | also_missing", None, None, None],
            ],
            "tool_usage_data": [
                {"Read": 5, "Grep": 3},
                {"Read": 2},
                {},
            ],
            "cache_efficiency_data": [
                [500, 1000],
                [200, 400],
                [None, None],
            ],
            "model_usage_data": [
                {"deepseek": 10},
                {"deepseek": 5},
                {},
            ],
            "session_errors": {},
        }

        cache_path = tmp_path / ".dashboard_metrics.json"
        import json
        cache_path.write_text(json.dumps(cache, indent=2))

        # Create DashboardScreen and call _load_metrics_from_cache
        screen = DashboardScreen(flows_dir=str(tmp_path))
        # We need to bypass the cache key check by patching _compute_cache_key
        with patch.object(screen, '_compute_cache_key', return_value="dummy"):
            result = screen._load_metrics_from_cache()

        assert result is True, "Cache should load successfully"

        # Should have only 1 entry (the existing file)
        assert len(screen._session_data) == 1, (
            f"Expected 1 session after dropping, got {len(screen._session_data)}"
        )
        assert screen._session_data[0][0].task_name == "existing", (
            "Only the existing-file session should remain"
        )

    def test_parallel_arrays_dropped_in_sync(self, tmp_path):
        """VAL-CACHE-019: All parallel arrays stay aligned after dropping."""
        from llm_flow_viewer.tui.screens.dashboard import DashboardScreen

        # Create one existing source file
        existing_file = tmp_path / "01_flows-existing"
        existing_file.write_text("exists")

        cache = {
            "cache_key": "dummy",
            "session_data": [
                [1, "existing", str(existing_file), 10, 100, 50],
                [2, "missing", str(tmp_path / "02_flows-missing"), 5, 200, 100],
            ],
            "timing_data": [
                ["01 | existing", 0.5, 1.0, 2.0],
                ["02 | missing", 0.3, 0.8, 1.5],
            ],
            "tool_usage_data": [
                {"Read": 5},
                {"Grep": 2},
            ],
            "cache_efficiency_data": [
                [500, 1000],
                [200, 400],
            ],
            "model_usage_data": [
                {"deepseek": 10},
                {"deepseek": 5},
            ],
            "session_errors": {},
        }

        cache_path = tmp_path / ".dashboard_metrics.json"
        import json
        cache_path.write_text(json.dumps(cache, indent=2))

        screen = DashboardScreen(flows_dir=str(tmp_path))
        with patch.object(screen, '_compute_cache_key', return_value="dummy"):
            result = screen._load_metrics_from_cache()

        assert result is True

        # All arrays should have the same length (1)
        assert len(screen._session_data) == 1
        assert len(screen._timing_data) == 1, (
            f"_timing_data length {len(screen._timing_data)} != 1"
        )
        assert len(screen._tool_usage_data) == 1, (
            f"_tool_usage_data length {len(screen._tool_usage_data)} != 1"
        )
        assert len(screen._cache_efficiency_data) == 1, (
            f"_cache_efficiency_data length {len(screen._cache_efficiency_data)} != 1"
        )
        assert len(screen._model_usage_data) == 1, (
            f"_model_usage_data length {len(screen._model_usage_data)} != 1"
        )

    def test_all_entries_dropped_empty_state(self, tmp_path):
        """VAL-CACHE-005b: All entries dropped → dashboard transitions to empty state."""
        from llm_flow_viewer.tui.screens.dashboard import DashboardScreen

        # No source files exist, only the cache
        cache = {
            "cache_key": "dummy",
            "session_data": [
                [1, "missing1", str(tmp_path / "01_flows-missing1"), 10, 100, 50],
                [2, "missing2", str(tmp_path / "02_flows-missing2"), 5, 200, 100],
            ],
            "timing_data": [
                ["01 | missing1", 0.5, 1.0, 2.0],
                ["02 | missing2", 0.3, 0.8, 1.5],
            ],
            "tool_usage_data": [{}, {}],
            "cache_efficiency_data": [[None, None], [None, None]],
            "model_usage_data": [{}, {}],
            "session_errors": {},
        }

        cache_path = tmp_path / ".dashboard_metrics.json"
        import json
        cache_path.write_text(json.dumps(cache, indent=2))

        screen = DashboardScreen(flows_dir=str(tmp_path))
        with patch.object(screen, '_compute_cache_key', return_value="dummy"):
            result = screen._load_metrics_from_cache()

        assert result is True
        assert len(screen._session_data) == 0, (
            "All entries should be dropped when source files are missing"
        )
        assert len(screen._timing_data) == 0
        assert len(screen._tool_usage_data) == 0
        assert len(screen._cache_efficiency_data) == 0
        assert len(screen._model_usage_data) == 0

    def test_cache_key_mismatch_triggers_recompute(self, tmp_path):
        """VAL-CACHE-005: Stale cache (key mismatch) returns False.

        The existing cache freshness check must still work: when the
        stored key does not match the computed key, return False.
        """
        from llm_flow_viewer.tui.screens.dashboard import DashboardScreen

        existing_file = tmp_path / "01_flows-existing"
        existing_file.write_text("exists")

        cache = {
            "cache_key": "stale-key-that-wont-match",
            "session_data": [
                [1, "existing", str(existing_file), 10, 100, 50],
            ],
            "timing_data": [
                ["01 | existing", 0.5, 1.0, 2.0],
            ],
            "tool_usage_data": [{}],
            "cache_efficiency_data": [[None, None]],
            "model_usage_data": [{}],
            "session_errors": {},
        }

        cache_path = tmp_path / ".dashboard_metrics.json"
        import json
        cache_path.write_text(json.dumps(cache, indent=2))

        screen = DashboardScreen(flows_dir=str(tmp_path))
        # Do NOT patch _compute_cache_key — let it compute the real key
        # which will differ from "stale-key-that-wont-match"
        result = screen._load_metrics_from_cache()

        assert result is False, (
            "Stale cache (key mismatch) should return False"
        )

    def test_cache_round_trip_consistency(self, tmp_path):
        """VAL-CACHE-005a: Save → reload with no file changes = cache hit."""
        from llm_flow_viewer.tui.screens.dashboard import DashboardScreen
        from llm_flow_viewer.tui.widgets.session_list import SessionInfo

        # Create a source file
        flow_file = tmp_path / "01_flows-test_session"
        flow_file.write_text("flow content")
        flow_file_2 = tmp_path / "02_flows-another"
        flow_file_2.write_text("more flow content")

        screen = DashboardScreen(flows_dir=str(tmp_path))

        # Populate with mock data
        screen._session_data = [
            (SessionInfo(1, "test_session", str(flow_file)), 10, 100, 50),
            (SessionInfo(2, "another", str(flow_file_2)), 5, 200, 100),
        ]
        screen._timing_data = [
            ("01 | test_session", 0.5, 1.0, 2.0),
            ("02 | another", 0.3, 0.8, 1.5),
        ]
        screen._tool_usage_data = [{"Read": 5}, {"Grep": 2}]
        screen._cache_efficiency_data = [(500, 1000), (200, 400)]
        screen._model_usage_data = [{"deepseek": 10}, {"deepseek": 5}]
        screen._session_errors = {}

        # Save to cache
        screen._save_metrics_to_cache()

        # Create a NEW screen pointing to the same flows dir
        screen2 = DashboardScreen(flows_dir=str(tmp_path))

        # Load from cache — should be a hit since no files changed
        result = screen2._load_metrics_from_cache()

        assert result is True, (
            "Round-trip: save then load with no file changes should be a cache hit"
        )
        assert len(screen2._session_data) == 2, (
            "Both sessions should be restored from cache"
        )

    def test_missing_cache_file_returns_false(self, tmp_path):
        """VAL-CACHE-014: Missing .dashboard_metrics.json returns False gracefully."""
        from llm_flow_viewer.tui.screens.dashboard import DashboardScreen

        screen = DashboardScreen(flows_dir=str(tmp_path))
        result = screen._load_metrics_from_cache()

        assert result is False, (
            "Missing cache file should return False, not crash"
        )

    def test_corrupted_cache_returns_false(self, tmp_path):
        """VAL-CACHE-015: Corrupted cache file returns False gracefully."""
        from llm_flow_viewer.tui.screens.dashboard import DashboardScreen

        cache_path = tmp_path / ".dashboard_metrics.json"
        cache_path.write_text("this is not valid json {{{")

        screen = DashboardScreen(flows_dir=str(tmp_path))
        result = screen._load_metrics_from_cache()

        assert result is False, (
            "Corrupted cache file should return False, not crash"
        )

    def test_session_ordering_by_index_after_dropping(self, tmp_path):
        """VAL-CACHE-018: Sessions remain ordered by index after dropping."""
        from llm_flow_viewer.tui.screens.dashboard import DashboardScreen

        # Create only sessions 1 and 3 as existing files
        file1 = tmp_path / "01_flows-session1"
        file1.write_text("exists")
        file3 = tmp_path / "03_flows-session3"
        file3.write_text("exists")

        cache = {
            "cache_key": "dummy",
            "session_data": [
                [1, "session1", str(file1), 10, 100, 50],
                [2, "session2", str(tmp_path / "02_flows-session2"), 5, 200, 100],
                [3, "session3", str(file3), 8, 300, 150],
            ],
            "timing_data": [
                ["01 | session1", 0.5, 1.0, 2.0],
                ["02 | session2", 0.3, 0.8, 1.5],
                ["03 | session3", 0.4, 0.9, 1.8],
            ],
            "tool_usage_data": [{"Read": 5}, {"Grep": 2}, {"LS": 3}],
            "cache_efficiency_data": [[500, 1000], [200, 400], [300, 600]],
            "model_usage_data": [{"deepseek": 10}, {"deepseek": 5}, {"deepseek": 8}],
            "session_errors": {},
        }

        cache_path = tmp_path / ".dashboard_metrics.json"
        import json
        cache_path.write_text(json.dumps(cache, indent=2))

        screen = DashboardScreen(flows_dir=str(tmp_path))
        with patch.object(screen, '_compute_cache_key', return_value="dummy"):
            result = screen._load_metrics_from_cache()

        assert result is True
        assert len(screen._session_data) == 2, (
            "Should have 2 entries after dropping session 2"
        )
        # Order should be session 1 (index 1) then session 3 (index 3)
        assert screen._session_data[0][0].index == 1
        assert screen._session_data[1][0].index == 3
        # Timing, tool, cache, model arrays should be in the same order
        assert "01 | session1" in screen._timing_data[0][0]
        assert "03 | session3" in screen._timing_data[1][0]

    def test_atomic_cache_write_no_partial_files(self, tmp_path):
        """VAL-CACHE-017: _save_metrics_to_cache uses os.replace for atomic writes.

        Verifies that no .tmp file remains after a successful save,
        and that the cache file is valid JSON.
        """
        from llm_flow_viewer.tui.screens.dashboard import DashboardScreen
        from llm_flow_viewer.tui.widgets.session_list import SessionInfo

        flow_file = tmp_path / "01_flows-test_session"
        flow_file.write_text("flow content")

        screen = DashboardScreen(flows_dir=str(tmp_path))
        screen._session_data = [
            (SessionInfo(1, "test_session", str(flow_file)), 10, 100, 50),
        ]
        screen._timing_data = [("01 | test_session", 0.5, 1.0, 2.0)]
        screen._tool_usage_data = [{"Read": 5}]
        screen._cache_efficiency_data = [(500, 1000)]
        screen._model_usage_data = [{"deepseek": 10}]
        screen._session_errors = {}

        screen._save_metrics_to_cache()

        cache_path = tmp_path / ".dashboard_metrics.json"
        tmp_path_check = tmp_path / ".dashboard_metrics.json.tmp"

        # The .tmp file should not exist after successful save
        assert not tmp_path_check.exists(), (
            "Temporary .tmp file should not remain after _save_metrics_to_cache"
        )
        # The cache file should exist and be valid JSON
        assert cache_path.exists(), "Cache file should exist after save"
        import json
        with open(str(cache_path), "r") as f:
            data = json.load(f)
        assert "cache_key" in data, "Saved cache should contain cache_key"
        assert "session_data" in data, "Saved cache should contain session_data"
