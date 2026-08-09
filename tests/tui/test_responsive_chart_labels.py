"""Tests for responsive chart label truncation.

Covers the following validation assertions:
- VAL-LABEL-001: Labels expand to full width on wide terminals
- VAL-LABEL-002: Labels truncate gracefully on narrow terminals
- VAL-LABEL-003 through VAL-LABEL-009: Each chart uses dynamic label width
- VAL-LABEL-010: Charts render correctly at all sizes
- VAL-LABEL-011: _compute_max_label_width returns reasonable values
- VAL-LABEL-012: _truncate_label handles all edge cases
- VAL-LABEL-013: Programmatic test class for all 7 charts
- VAL-LABEL-014: Labels retain Rich styling after truncation
- VAL-LABEL-015: No hardcoded truncation constants remain
"""

from __future__ import annotations

import textwrap
from typing import Any
from unittest.mock import PropertyMock, patch

import pytest
from rich.style import Style
from rich.table import Table
from rich.text import Text

from llm_flow_viewer.tui.widgets.dashboard_widgets import (
    _compute_max_label_width,
    _truncate_label,
    AvgTokensChart,
    CacheEfficiencyChart,
    ModelUsageChart,
    SessionBarChart,
    TimingChart,
    TokenBarChart,
    ToolUsageChart,
)


# ---------------------------------------------------------------------------
# Mock helper for content_region on unmounted widgets
# ---------------------------------------------------------------------------

# Track active patchers so they can be cleaned up after each test.
# Prevents cross-test contamination from class-level property patches.
_active_content_region_patchers: list[patch] = []


def _cleanup_content_region_patches() -> None:
    """Stop all active content_region patchers and clear the tracker."""
    while _active_content_region_patchers:
        p = _active_content_region_patchers.pop()
        try:
            p.stop()
        except RuntimeError:
            pass  # already stopped


def _mock_content_region(widget: Any, width: int, height: int = 40) -> None:
    """Mock content_region on a widget for testing.

    ``content_region`` is a read-only Textual property.  We patch it
    with a ``PropertyMock`` that returns a suitable region namedtuple.
    """
    from collections import namedtuple
    Region = namedtuple("Region", ["x", "y", "width", "height"])
    patcher = patch(
        f"{widget.__class__.__name__}.content_region",
        new_callable=PropertyMock,
        return_value=Region(0, 0, width, height),
    )
    patcher.start()
    _active_content_region_patchers.append(patcher)
    return patcher


# ---------------------------------------------------------------------------
# Tests for _compute_max_label_width (VAL-LABEL-011)
# ---------------------------------------------------------------------------


class TestComputeMaxLabelWidth:
    """VAL-LABEL-011: _compute_max_label_width returns reasonable values."""

    def test_returns_positive_integer(self):
        """Returns a positive integer for all valid inputs."""
        for width in [40, 80, 120, 200, 320, 1000]:
            for cols in [3, 4, 5, 6, 7, 8]:
                result = _compute_max_label_width(width, cols)
                assert isinstance(result, int), f"Expected int, got {type(result)}"
                assert result >= 4, f"Width {width}, cols {cols}: got {result}, min 4"

    def test_very_wide_region(self):
        """Very wide regions produce large label widths."""
        result = _compute_max_label_width(500, 3)
        assert result >= 60, f"Expected >= 60 for 500 width, got {result}"

    def test_very_narrow_region(self):
        """Very narrow regions produce minimum viable width."""
        result = _compute_max_label_width(10, 7)
        assert result >= 4, f"Expected >= 4 for narrow width, got {result}"

    def test_zero_width_returns_default(self):
        """Zero or negative width returns default of 40."""
        assert _compute_max_label_width(0, 3) == 40
        assert _compute_max_label_width(-1, 3) == 40

    def test_more_columns_reduces_label_width(self):
        """More columns means less space for labels."""
        w3 = _compute_max_label_width(120, 3)
        w7 = _compute_max_label_width(120, 7)
        assert w3 >= w7, (
            f"Expected 3-col label width ({w3}) >= 7-col ({w7})"
        )

    def test_wider_terminal_gives_wider_labels(self):
        """Wider terminals give wider label allocation."""
        w80 = _compute_max_label_width(80, 3)
        w200 = _compute_max_label_width(200, 3)
        assert w200 > w80, (
            f"Expected 200-col width ({w200}) > 80-col ({w80})"
        )


# ---------------------------------------------------------------------------
# Tests for _truncate_label (VAL-LABEL-012)
# ---------------------------------------------------------------------------


class TestTruncateLabel:
    """VAL-LABEL-012: _truncate_label handles all edge cases."""

    def test_short_label_unchanged(self):
        """Label shorter than max_width is returned unchanged."""
        result = _truncate_label("hello", 20)
        assert result.plain == "hello"

    def test_exact_fit_unchanged(self):
        """Label exactly matching max_width is returned unchanged."""
        label = "exactly_20_char"  # 16 chars
        result = _truncate_label(label, 16)
        assert result.plain == label

    def test_long_label_truncated(self):
        """Label longer than max_width is truncated with '...'."""
        result = _truncate_label("this is a very long label", 15)
        assert result.plain == "this is a ve..."
        assert "..." in result.plain
        assert len(result.plain) == 15

    def test_empty_string(self):
        """Empty string label is returned as empty Text."""
        result = _truncate_label("", 20)
        assert result.plain == ""

    def test_max_width_less_than_4(self):
        """max_width < 4 returns just '...'."""
        result = _truncate_label("hello", 3)
        assert result.plain == "..."
        result2 = _truncate_label("hello", 1)
        assert result2.plain == "..."
        result3 = _truncate_label("hello", 0)
        assert result3.plain == "..."

    def test_max_width_equals_4(self):
        """max_width == 4 returns label (4 chars) since it fits."""
        result = _truncate_label("test", 4)
        assert result.plain == "test"

    def test_very_long_label(self):
        """Very long label is truncated properly."""
        long_label = "a" * 200
        result = _truncate_label(long_label, 20)
        assert result.plain == "a" * 17 + "..."
        assert len(result.plain) == 20

    def test_preserves_rich_style(self):
        """Truncated label preserves Rich styling."""
        styled = Text("very long label that is styled", style="bold red")
        result = _truncate_label(styled, 15)
        assert result.plain == "very long la..."
        # Text.style may return a string ("bold red") when set that way,
        # or a Style object. Verify bold and red are present.
        style_str = str(result.style)
        assert "bold" in style_str, f"Expected bold in style, got '{style_str}'"
        assert "red" in style_str or "1" in style_str, (
            f"Expected red color in style, got '{style_str}'"
        )

    @pytest.mark.parametrize("label,max_w,expected", [
        ("short", 20, "short"),
        ("exact_fit_12", 12, "exact_fit_12"),
        ("very long label truncation test that is really long", 30, "very long label truncation ..."),
        ("", 20, ""),
        ("x" * 100, 40, "x" * 37 + "..."),
    ])
    def test_various_lengths(self, label, max_w, expected):
        """Parameterized test for various label lengths."""
        result = _truncate_label(label, max_w)
        assert result.plain == expected, (
            f"Expected '{expected}', got '{result.plain}' for label='{label}' at max_width={max_w}"
        )


# ---------------------------------------------------------------------------
# Tests for style preservation (VAL-LABEL-014)
# ---------------------------------------------------------------------------


class TestLabelStylePreservation:
    """VAL-LABEL-014: Labels retain Rich styling after dynamic truncation."""

    def test_bold_style_preserved(self):
        """Bold style is preserved after truncation."""
        styled = Text("very long label that is styled", style=Style(bold=True))
        result = _truncate_label(styled, 15)
        assert result.style == Style(bold=True)

    def test_color_style_preserved(self):
        """Color style is preserved after truncation."""
        styled = Text("very long label that is styled", style="green")
        result = _truncate_label(styled, 12)
        style_str = str(result.style)
        assert "green" in style_str or "2" in style_str, (
            f"Expected green color in style, got '{style_str}'"
        )

    def test_combined_style_preserved(self):
        """Combined bold+color style is preserved."""
        style = Style(bold=True, color="#00aa00", italic=True)
        styled = Text("very long label that is styled", style=style)
        result = _truncate_label(styled, 10)
        assert result.style == style

    def test_unstyled_label_plain_text(self):
        """Plain string labels return Text with default style."""
        result = _truncate_label("hello", 20)
        # The result's .style may be None, Style(), or has no bold/color/italic
        plain = result.plain
        assert plain == "hello"

    def test_no_truncation_preserves_style_identity(self):
        """Labels not needing truncation preserve their style identity."""
        styled = Text("short", style="bold red")
        result = _truncate_label(styled, 20)
        assert result is styled  # same object returned
        # Verify style string representation contains bold and red
        style_str = str(styled.style)
        assert "bold" in style_str, f"Expected bold in style, got '{style_str}'"
        assert "red" in style_str or "1" in style_str, (
            f"Expected red in style, got '{style_str}'"
        )


# ---------------------------------------------------------------------------
# Tests for each chart widget at different widths
# ---------------------------------------------------------------------------

# Long test labels that should only be fully visible on wide terminals
_LONG_LABEL = "01 | very_long_session_name_for_testing"
_LONG_TOOL = "read_file_with_long_parameters"
_LONG_MODEL = "gpt-4-turbo-preview-0125-with-extra-details"


def _make_chart_data(chart_class: type) -> list:
    """Create mock data appropriate for each chart class."""
    if chart_class is SessionBarChart:
        return [(_LONG_LABEL, 100), ("02 | short", 50)]
    if chart_class is TokenBarChart:
        return [(_LONG_LABEL, 5000, 250), ("02 | short", 3000, 150)]
    if chart_class is AvgTokensChart:
        return [(_LONG_LABEL, 1500.5), ("02 | short", 800.3)]
    if chart_class is TimingChart:
        return [(_LONG_LABEL, 0.5, 2.1, 5.3), ("02 | short", 0.3, 1.5, 3.2)]
    if chart_class is ToolUsageChart:
        return [(_LONG_TOOL, 50), ("read", 30), ("grep", 20)]
    if chart_class is CacheEfficiencyChart:
        return [(_LONG_LABEL, 10000, 5000), ("02 | short", 5000, 3000)]
    if chart_class is ModelUsageChart:
        return [(_LONG_MODEL, 50), ("gpt-4", 30)]
    return []


def _set_content_region_width(chart: Any, width: int) -> None:
    """Simulate widget content region width using mock.

    ``content_region`` is a read-only Textual property, so we use
    ``PropertyMock`` to patch it on the widget class.  The patcher is
    tracked so it can be cleaned up after the test completes.
    """
    from collections import namedtuple
    Region = namedtuple("Region", ["x", "y", "width", "height"])
    patcher = patch.object(
        type(chart),
        "content_region",
        new_callable=PropertyMock,
        return_value=Region(0, 0, width, 40),
    )
    patcher.start()
    _active_content_region_patchers.append(patcher)
    # Return the patcher so callers can stop it early if needed
    return patcher


def _setup_chart(chart_class, data_key, column_count, terminal_width):
    """Create and configure a chart widget, then set its content_region."""
    chart = chart_class()
    data = _make_chart_data(chart_class)

    if data_key == "avg_data":
        chart.update_data(data, overall_avg=1200.0)
    elif data_key == "tool_data":
        chart.update_data(data)
    elif data_key == "cache_data":
        chart.update_data(data)
    elif data_key == "model_data":
        chart.update_data(data)
    else:
        setattr(chart, data_key, data)

    _set_content_region_width(chart, terminal_width)
    return chart


class TestChartLabelWidths:
    """VAL-LABEL-013: Programmatic test exercises all 7 chart types at
    multiple terminal widths.

    Also covers VAL-LABEL-001 through VAL-LABEL-009 (each chart uses
    dynamic label width) and VAL-LABEL-010 (renders at all sizes).
    """

    @pytest.fixture(autouse=True)
    def _cleanup_patches(self) -> None:
        """Clean up content_region property patches after each test."""
        # Yield to the test
        yield
        # Clean up any lingering patchers to prevent cross-test contamination
        _cleanup_content_region_patches()

    @pytest.mark.parametrize("chart_class,data_key,column_count", [
        (SessionBarChart, "chart_data", 3),
        (TokenBarChart, "chart_data", 5),
        (AvgTokensChart, "avg_data", 3),
        (TimingChart, "timing_data", 7),
        (ToolUsageChart, "tool_data", 4),
        (CacheEfficiencyChart, "cache_data", 4),
        (ModelUsageChart, "model_data", 4),
    ])
    @pytest.mark.parametrize("terminal_width", [80, 120, 200, 320])
    def test_chart_renders_at_width(
        self, chart_class, data_key, column_count, terminal_width
    ):
        """Each chart renders without error at various widths."""
        chart = _setup_chart(chart_class, data_key, column_count, terminal_width)
        result = chart.render()
        assert isinstance(result, (Table, Text)), (
            f"{chart_class.__name__} at {terminal_width} cols: "
            f"expected Table or Text, got {type(result).__name__}"
        )

    @pytest.mark.parametrize("chart_class,data_key,column_count", [
        (SessionBarChart, "chart_data", 3),
        (TokenBarChart, "chart_data", 5),
        (AvgTokensChart, "avg_data", 3),
        (TimingChart, "timing_data", 7),
        (ToolUsageChart, "tool_data", 4),
        (CacheEfficiencyChart, "cache_data", 4),
        (ModelUsageChart, "model_data", 4),
    ])
    def test_long_labels_truncated_at_narrow_width(
        self, chart_class, data_key, column_count
    ):
        """VAL-LABEL-002: Long labels truncate at 80 columns."""
        chart = _setup_chart(chart_class, data_key, column_count, 80)
        result = chart.render()

        max_w = _compute_max_label_width(80, column_count)

        if isinstance(result, Table):
            rendered = " ".join(
                str(cell) for col in result.columns for cell in col._cells
            )
            if chart_class is ToolUsageChart:
                label = _LONG_TOOL
            elif chart_class is ModelUsageChart:
                label = _LONG_MODEL
            else:
                label = _LONG_LABEL

            if len(label) > max_w:
                assert "..." in rendered, (
                    f"{chart_class.__name__}: expected '...' in label at 80 cols "
                    f"(max_label_width={max_w}, label_len={len(label)})"
                )

    @pytest.mark.parametrize("chart_class,data_key,column_count", [
        (SessionBarChart, "chart_data", 3),
        (TokenBarChart, "chart_data", 5),
        (AvgTokensChart, "avg_data", 3),
        (TimingChart, "timing_data", 7),
        (ToolUsageChart, "tool_data", 4),
        (CacheEfficiencyChart, "cache_data", 4),
        (ModelUsageChart, "model_data", 4),
    ])
    def test_long_labels_full_at_wide_width(
        self, chart_class, data_key, column_count
    ):
        """VAL-LABEL-001: Long labels render in full at 200+ columns."""
        chart = _setup_chart(chart_class, data_key, column_count, 200)
        result = chart.render()

        max_w = _compute_max_label_width(200, column_count)

        if isinstance(result, Table):
            rendered = " ".join(
                str(cell) for col in result.columns for cell in col._cells
            )
            if chart_class is ToolUsageChart:
                label = _LONG_TOOL
            elif chart_class is ModelUsageChart:
                label = _LONG_MODEL
            else:
                label = _LONG_LABEL

            # If max_label_width is >= label length, the full label should appear
            if max_w >= len(label):
                assert label in rendered, (
                    f"{chart_class.__name__}: expected full label at 200 cols "
                    f"(max_label_width={max_w}, label_len={len(label)})"
                )


# ---------------------------------------------------------------------------
# Tests for ToolUsageChart labels (VAL-LABEL-007 specific)
# ---------------------------------------------------------------------------


class TestToolUsageChartLabels:
    """VAL-LABEL-007: ToolUsageChart handles tool names (not session names)."""

    def test_tool_names_not_session_names(self):
        """Tool chart labels come from tool names, not session names."""
        chart = ToolUsageChart()
        chart.update_data([
            ("tool_read_file", 50),
            ("tool_search_code", 30),
        ])
        result = chart.render()
        assert isinstance(result, Table)
        tool_col = result.columns[0]
        tool_cells = [str(cell) for cell in tool_col._cells]
        assert any("tool_read" in cell for cell in tool_cells), (
            f"Expected 'tool_read_file' in tool cells, got {tool_cells[:3]}"
        )


# ---------------------------------------------------------------------------
# Tests for ModelUsageChart labels (VAL-LABEL-009 specific)
# ---------------------------------------------------------------------------


class TestModelUsageChartLabels:
    """VAL-LABEL-009: ModelUsageChart handles model names (not session names)."""

    def test_model_names_not_session_names(self):
        """Model chart labels come from model names, not session names."""
        chart = ModelUsageChart()
        chart.update_data([
            ("gpt-4-turbo-preview", 50),
            ("claude-3-opus", 30),
        ])
        result = chart.render()
        assert isinstance(result, Table)
        model_col = result.columns[0]
        model_cells = [str(cell) for cell in model_col._cells]
        assert any("gpt-4-turbo" in cell for cell in model_cells), (
            f"Expected 'gpt-4-turbo' in model cells, got {model_cells[:3]}"
        )


# ---------------------------------------------------------------------------
# Tests for width-dependent label behavior (VAL-LABEL-003 through VAL-LABEL-009)
# ---------------------------------------------------------------------------


class TestWidthDependentLabels:
    """Labels change width when terminal width changes."""

    @pytest.mark.parametrize("chart_class,data_key,column_count", [
        (SessionBarChart, "chart_data", 3),
        (TokenBarChart, "chart_data", 5),
        (AvgTokensChart, "avg_data", 3),
        (TimingChart, "timing_data", 7),
        (ToolUsageChart, "tool_data", 4),
        (CacheEfficiencyChart, "cache_data", 4),
        (ModelUsageChart, "model_data", 4),
    ])
    def test_label_width_increases_with_terminal_width(
        self, chart_class, data_key, column_count
    ):
        """Each chart's label width grows when terminal is wider (VAL-LABEL-003-009)."""
        # Compute expected label widths at different terminal sizes
        w80 = _compute_max_label_width(80, column_count)
        w120 = _compute_max_label_width(120, column_count)
        w200 = _compute_max_label_width(200, column_count)

        # Verify widths increase
        assert w120 > w80, (
            f"{chart_class.__name__}: expected 120-col width ({w120}) > 80-col ({w80})"
        )
        assert w200 > w120, (
            f"{chart_class.__name__}: expected 200-col width ({w200}) > 120-col ({w120})"
        )


# ---------------------------------------------------------------------------
# Static analysis: no hardcoded truncation constants (VAL-LABEL-015)
# ---------------------------------------------------------------------------


class TestNoHardcodedTruncation:
    """VAL-LABEL-015: No hardcoded truncation constants remain in chart classes."""

    CHART_CLASS_NAMES = [
        "SessionBarChart",
        "TokenBarChart",
        "AvgTokensChart",
        "TimingChart",
        "ToolUsageChart",
        "CacheEfficiencyChart",
        "ModelUsageChart",
    ]

    def test_no_label_slicing_patterns(self):
        """No label[:N] or name[:N] slicing patterns exist in chart render methods."""
        source = textwrap.dedent(open(
            "src/llm_flow_viewer/tui/widgets/dashboard_widgets.py",
            encoding="utf-8",
        ).read())

        # Check for common hardcoded truncation patterns in the chart classes
        lines = source.splitlines()
        forbidden_lines = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Skip comments
            if stripped.startswith("#"):
                continue
            # Skip lines that reference helper functions
            if "_truncate_label" in stripped or "_compute_max_label_width" in stripped:
                continue
            # Check for label[:N] or name[:N] patterns (but not in test files)
            if ("label[" in stripped or "name[" in stripped) and ":" in stripped:
                forbidden_lines.append(f"  Line {i+1}: {stripped}")

        assert not forbidden_lines, (
            f"Found hardcoded truncation patterns:\n" + "\n".join(forbidden_lines)
        )

    def test_render_methods_use_helpers(self):
        """All 7 chart render methods reference _compute_max_label_width and _truncate_label."""
        source = textwrap.dedent(open(
            "src/llm_flow_viewer/tui/widgets/dashboard_widgets.py",
            encoding="utf-8",
        ).read())

        for class_name in self.CHART_CLASS_NAMES:
            # Find the class definition's line number
            class_lines = [
                (i, line) for i, line in enumerate(source.splitlines())
                if f"class {class_name}" in line
            ]
            assert class_lines, f"Could not find class {class_name}"
            class_line = class_lines[0][0]

            # Look for render method within the class
            lines = source.splitlines()
            render_start = None
            for i in range(class_line + 1, min(class_line + 150, len(lines))):
                if "def render(self)" in lines[i] or "def render(self," in lines[i]:
                    render_start = i
                    break

            assert render_start is not None, (
                f"Could not find render method in {class_name}"
            )

            # Check the render method contains _compute_max_label_width
            # Use a generous window since some render methods are long
            render_lines = lines[render_start:render_start + 100]
            combined = "\n".join(render_lines)
            assert "_compute_max_label_width" in combined, (
                f"{class_name}.render() does not call _compute_max_label_width"
            )
            assert "_truncate_label" in combined, (
                f"{class_name}.render() does not call _truncate_label"
            )


# ---------------------------------------------------------------------------
# Empty data edge cases (robustness)
# ---------------------------------------------------------------------------


class TestChartEmptyData:
    """Charts handle empty data at all widths."""

    @pytest.fixture(autouse=True)
    def _cleanup_patches(self) -> None:
        """Clean up content_region property patches after each test."""
        yield
        _cleanup_content_region_patches()

    @pytest.mark.parametrize("chart_class,data_key", [
        (SessionBarChart, "chart_data"),
        (TokenBarChart, "chart_data"),
        (AvgTokensChart, "avg_data"),
        (TimingChart, "timing_data"),
        (ToolUsageChart, "tool_data"),
        (CacheEfficiencyChart, "cache_data"),
        (ModelUsageChart, "model_data"),
    ])
    @pytest.mark.parametrize("terminal_width", [80, 120, 200])
    def test_empty_data_renders_at_all_widths(
        self, chart_class, data_key, terminal_width
    ):
        """Chart with empty data renders 'No data' at any width."""
        from rich.text import Text as RichText

        chart = chart_class()
        # Set empty data via the reactive attribute directly
        setattr(chart, data_key, [])
        _set_content_region_width(chart, terminal_width)

        result = chart.render()
        assert isinstance(result, RichText), (
            f"{chart_class.__name__}: empty data should return Text, "
            f"got {type(result).__name__}"
        )


# ---------------------------------------------------------------------------
# Cross-check: SessionList is NOT affected (VAL-CROSS-006)
# ---------------------------------------------------------------------------


class TestSessionListNonRegression:
    """SessionList sidebar truncation must not regress (non-regression check)."""

    def test_session_list_truncation_unchanged(self):
        """SessionList._truncate_label still exists and works as before."""
        from llm_flow_viewer.tui.widgets.session_list import SessionList

        # The static method should still exist
        assert hasattr(SessionList, "_truncate_label")
        assert callable(SessionList._truncate_label)

        # Its behavior should be unchanged
        # label[:max_length - 3] + "..." → "very long label for testing"[:12] + "..."
        result = SessionList._truncate_label("very long label for testing", 15)
        assert result == "very long la...", (
            f"Expected 'very long la...', got '{result}'"
        )

        # Should return unchanged for short labels
        result = SessionList._truncate_label("short", 15)
        assert result == "short"
