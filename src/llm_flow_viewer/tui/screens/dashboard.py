"""Dashboard screen showing cross-session metrics with bar charts.

Displays:
- Aggregate metrics panel (total sessions, total calls, total tokens)
- Bar chart of call counts per session
- Bar chart of token usage per session
- Average tokens per call per session
- Timing chart (min/avg/max RTT per session)
- Session metrics DataTable
- Single-session and all-sessions view toggling
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.worker import Worker
from textual.widgets import Header, DataTable, Input, Label, Static

from llm_flow_viewer.tui.widgets.app_footer import AppFooter
from llm_flow_viewer.tui.widgets.dashboard_widgets import (
    AvgTokensChart,
    CacheEfficiencyChart,
    ComparisonPanel,
    MetricsPanel,
    ModelUsageChart,
    SessionBarChart,
    TimingChart,
    TokenBarChart,
    ToolUsageChart,
)
from llm_flow_viewer.tui.widgets.session_list import (
    SessionInfo,
    discover_sessions,
)

logger = logging.getLogger(__name__)


class DashboardScreen(Screen):
    """Dashboard view showing session metrics, bar charts, and comparisons.

    Displays:
    - Aggregate metrics panel (total sessions, calls, tokens)
    - Bar chart of call counts per session
    - Bar chart of token usage per session
    - Average tokens per call per session chart
    - Timing chart (min/avg/max RTT per session) with fastest/slowest
      session highlighting
    - Session metrics DataTable with detailed counts

    Supports toggling between all-sessions and single-session views.
    When a session is focused (Enter on its table row), all charts
    narrow to that session only.  Pressing Escape or Backspace returns
    to the all-sessions overview.

    The session that was selected in Browse view (if any) is visually
    distinguished (bold, accent color) in the table.
    """

    BINDINGS = [
        Binding("escape", "handle_escape", "Browse", show=True),
        Binding("b", "switch_to_browse", "Browse", show=False),
        Binding("backspace", "clear_session_focus", "All Sessions", show=False),
        Binding("c", "toggle_comparison", "Compare", show=True),
        Binding("q", "quit", "Quit", show=True),
        Binding("tab", "focus_next", "Next Panel", show=False),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("r", "refresh_dashboard", "Refresh", show=True),
    ]

    CSS = """
    DashboardScreen {
        align: center top;
    }

    #dashboard-content {
        layout: vertical;
        width: 100%;
        height: 1fr;
        overflow-y: auto;
    }

    #dashboard-title {
        width: 100%;
        height: 3;
        content-align: center middle;
        text-style: bold;
        padding: 0 1;
        margin: 0;
    }

    #dashboard-metrics {
        width: 100%;
        height: auto;
        margin: 0 0 1 0;
    }

    #chart-call-counts {
        width: 1fr;
        height: auto;
        margin: 0 0 1 0;
    }

    #chart-token-usage {
        width: 1fr;
        height: auto;
        margin: 0 0 1 0;
    }

    #avg-tokens-header {
        width: 100%;
        height: auto;
        margin: 0 0 1 0;
    }

    #session-filter {
        width: 100%;
        height: 3;
        margin: 0 0 1 0;
        border: solid $primary;
    }

    #session-filter:focus {
        border: solid $accent;
    }

    #sessions-table {
        width: 100%;
        height: auto;
        min-height: 6;
        border: solid $primary;
        border-title-color: $primary;
    }

    #sessions-table:focus-within {
        border: solid $accent;
    }

    #dashboard-hint {
        width: 100%;
        height: 1;
        content-align: center middle;
        color: $text-muted;
        padding: 0 1;
    }

    #view-mode-label {
        width: 100%;
        height: 1;
        content-align: center middle;
        color: $accent;
        text-style: italic;
        padding: 0 1;
    }

    /* ---- Comparison mode ---- */
    #comparison-container {
        layout: vertical;
        width: 100%;
        height: 1fr;
        border: solid $secondary;
        margin: 0 0 1 0;
        padding: 0 1;
        display: none;
    }

    #comparison-container.visible {
        display: block;
    }

    #comparison-header {
        width: 100%;
        height: 1;
        content-align: center middle;
        text-style: bold;
        color: $accent;
        padding: 0 1;
        margin: 0 0 1 0;
    }

    #comparison-panels {
        layout: horizontal;
        width: 100%;
        height: auto;
    }

    #comparison-hint {
        width: 100%;
        height: 1;
        content-align: center middle;
        color: $text-muted;
        padding: 0 1;
    }

    /* Drill-down panel */
    #drill-down-panel {
        width: 100%;
        height: 1fr;
        border: solid $warning;
        margin: 0 0 1 0;
        padding: 0 1;
        display: none;
    }

    #drill-down-panel.visible {
        display: block;
    }

    /* Session errors warning */
    #session-errors-warning {
        width: 100%;
        height: 1;
        content-align: left middle;
        color: $warning;
        text-style: bold;
        padding: 0 1;
        display: none;
    }

    #session-errors-warning.visible {
        display: block;
    }

    /* Focus indicator for comparison panels handled in widget CSS */

    /* ---- Empty state ---- */
    #dashboard-empty-state {
        width: 100%;
        height: 1fr;
        align: center middle;
        display: none;
    }

    #dashboard-empty-state.visible {
        display: block;
    }

    #dashboard-empty-state > #empty-state-title {
        width: 100%;
        height: 3;
        content-align: center middle;
        text-style: bold;
        color: $text;
    }

    #dashboard-empty-state > #empty-state-hint {
        width: 100%;
        height: 3;
        content-align: center middle;
        color: $text-muted;
        text-style: italic;
    }

    /* ---- Loading indicator ---- */
    #dashboard-loading-indicator {
        width: 100%;
        height: 1fr;
        align: center middle;
        display: none;
    }

    #dashboard-loading-indicator.visible {
        display: block;
    }

    #dashboard-loading-indicator > #loading-spinner {
        width: 100%;
        height: 3;
        content-align: center middle;
        text-style: bold;
        color: $accent;
    }

    #dashboard-loading-indicator > #loading-message {
        width: 100%;
        height: 1;
        content-align: center middle;
        color: $text-muted;
    }

    Header {
        dock: top;
    }
    """

    def __init__(
        self,
        flows_dir: str = "./flows",
        selected_session_index: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._flows_dir = flows_dir
        self._selected_session_index = selected_session_index
        self._focused_session_index: int | None = None
        # Comparison mode state
        self._comparison_mode: bool = False
        self._compare_session_indices: set[int] = set()
        self._drill_down_session_index: int | None = None
        # Cached session data: list of (SessionInfo, call_count, total_input, total_output)
        self._session_data: List[Tuple[SessionInfo, int, int, int]] = []
        # Cached timing data: list of (label, min_rtt, avg_rtt, max_rtt) in seconds
        self._timing_data: List[Tuple[str, Optional[float], Optional[float], Optional[float]]] = []
        # Cached tool usage data: list of dict[tool_name -> count] per session
        self._tool_usage_data: List[dict[str, int]] = []
        # Cached cache efficiency data: list of (cache_read, input_tokens) per session
        self._cache_efficiency_data: List[Tuple[Optional[int], Optional[int]]] = []
        # Cached model usage data: list of dict[model_name -> count] per session
        self._model_usage_data: List[dict[str, int]] = []
        # Per-session error tracking: maps session index -> error message string
        self._session_errors: Dict[int, str] = {}
        # Track async load state
        self._data_loaded: bool = False
        # Track active workers so they can be cancelled during shutdown
        # (VAL-GEN-001, VAL-GEN-002, VAL-GEN-005)
        self._workers: list[Worker] = []
        # Filter text for session name type-to-filter (empty = show all)
        self._filter_text: str = ""

    # ------------------------------------------------------------------
    # Dashboard metrics cache (JSON — .dashboard_metrics.json)
    # ------------------------------------------------------------------

    @property
    def _cache_path(self) -> str:
        """Absolute path to the dashboard metrics cache JSON file.

        The cache file is stored in the flows directory as
        ``.dashboard_metrics.json``.
        """
        return os.path.join(self._flows_dir, ".dashboard_metrics.json")

    def _compute_cache_key(self) -> str:
        """Compute a checksum of all session source files and parquet files.

        For each discovered session, hashes:
        1. The source flow file (path, size, mtime_ns) — ensures deletion or
           modification of any source file changes the key.
        2. All associated parquet cache files (name, size, mtime_ns).

        If any source file is deleted, the key changes because that session's
        (path, size, mtime) tuple is absent from the hash input.  This also
        catches sessions with zero LLM calls (no parquet files), which were
        invisible under the old parquet-only hashing.

        Returns:
            A SHA-256 hex digest string.
        """
        sessions = discover_sessions(self._flows_dir)
        hasher = hashlib.sha256()

        for sinfo in sessions:
            # Hash source flow file info (path, size, mtime_ns)
            try:
                stat_info = os.stat(sinfo.file_path)
                entry = f"src:{sinfo.file_path}:{stat_info.st_size}:{stat_info.st_mtime_ns}"
                hasher.update(entry.encode("utf-8"))
            except OSError:
                continue

            # Hash parquet cache files as before
            flow_dir = os.path.dirname(sinfo.file_path)
            flow_name = os.path.basename(sinfo.file_path)
            try:
                for fname in sorted(os.listdir(flow_dir)):
                    if fname.endswith(".parquet") and flow_name in fname:
                        fpath = os.path.join(flow_dir, fname)
                        try:
                            stat_info = os.stat(fpath)
                            entry = f"{fname}:{stat_info.st_size}:{stat_info.st_mtime_ns}"
                            hasher.update(entry.encode("utf-8"))
                        except OSError:
                            continue
            except OSError:
                continue

        return hasher.hexdigest()

    def _load_metrics_from_cache(self) -> bool:
        """Load dashboard metrics from the JSON cache file.

        Checks cache freshness by comparing the stored cache key against
        the current checksum of session parquet files.  If the cache is
        fresh, all ``_session_data``, ``_timing_data``,
        ``_tool_usage_data``, ``_cache_efficiency_data``,
        ``_model_usage_data`` and ``_session_errors`` are restored.

        Returns:
            True if metrics were successfully loaded from cache,
            False if cache is missing, stale, or corrupted.
        """
        cache_path = self._cache_path
        if not os.path.isfile(cache_path):
            return False

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)

            # Check cache key freshness
            current_key = self._compute_cache_key()
            if cache.get("cache_key") != current_key:
                logger.info("Dashboard cache stale — cache key mismatch")
                return False

            # Restore session data
            self._session_data = []
            for entry in cache["session_data"]:
                sinfo = SessionInfo(
                    index=entry[0],
                    task_name=entry[1],
                    file_path=entry[2],
                )
                self._session_data.append((sinfo, entry[3], entry[4], entry[5]))

            # Restore timing data: [label, min, avg, max]
            self._timing_data = [
                (entry[0], entry[1], entry[2], entry[3])
                for entry in cache["timing_data"]
            ]

            # Restore tool usage data
            self._tool_usage_data = list(cache["tool_usage_data"])

            # Restore cache efficiency data
            self._cache_efficiency_data = [
                (entry[0], entry[1])
                for entry in cache["cache_efficiency_data"]
            ]

            # Restore model usage data
            self._model_usage_data = list(cache["model_usage_data"])

            # Restore session errors (JSON keys are strings → convert back)
            self._session_errors = {
                int(k): v for k, v in cache.get("session_errors", {}).items()
            }

            # --- Source file existence validation ---
            # Drop any cached entry whose source flow file no longer exists.
            # Also remove corresponding entries from all parallel data arrays
            # so that _session_data, _timing_data, _tool_usage_data,
            # _cache_efficiency_data, and _model_usage_data remain aligned.
            valid_indices = [
                i
                for i, (sinfo, _, _, _) in enumerate(self._session_data)
                if os.path.isfile(sinfo.file_path)
            ]
            if len(valid_indices) < len(self._session_data):
                dropped = len(self._session_data) - len(valid_indices)
                logger.info(
                    "Dropping %d cached entries with missing source files",
                    dropped,
                )
                self._session_data = [self._session_data[i] for i in valid_indices]
                self._timing_data = [self._timing_data[i] for i in valid_indices]
                self._tool_usage_data = [self._tool_usage_data[i] for i in valid_indices]
                self._cache_efficiency_data = [self._cache_efficiency_data[i] for i in valid_indices]
                self._model_usage_data = [self._model_usage_data[i] for i in valid_indices]

            logger.info("Loaded dashboard metrics from cache: %s", cache_path)
            return True
        except Exception as exc:
            logger.warning(
                "Failed to load dashboard metrics cache: %s", exc,
            )
            return False

    def _save_metrics_to_cache(self) -> None:
        """Save current dashboard metrics to the JSON cache file.

        Computes the current cache key from session parquet files,
        serialises all metric data structures, and writes
        ``.dashboard_metrics.json`` to the flows directory.
        """
        cache_key = self._compute_cache_key()

        # Serialise session data (SessionInfo → plain tuples)
        session_data_serialized = [
            [sinfo.index, sinfo.task_name, sinfo.file_path, ccount, inp, out]
            for sinfo, ccount, inp, out in self._session_data
        ]

        cache = {
            "cache_key": cache_key,
            "session_data": session_data_serialized,
            "timing_data": self._timing_data,
            "tool_usage_data": self._tool_usage_data,
            "cache_efficiency_data": self._cache_efficiency_data,
            "model_usage_data": self._model_usage_data,
            "session_errors": {str(k): v for k, v in self._session_errors.items()},
        }

        cache_path = self._cache_path
        try:
            # Write atomically: write to temp file, then rename
            tmp_path = cache_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)
            os.replace(tmp_path, cache_path)
            logger.info("Saved dashboard metrics cache: %s", cache_path)
        except Exception as exc:
            logger.warning(
                "Failed to save dashboard metrics cache: %s", exc,
            )

    def compose(self) -> ComposeResult:
        """Create child widgets for the dashboard screen."""
        yield Header(show_clock=True)
        yield Label(
            "Session Metrics Dashboard",
            id="dashboard-title",
        )
        yield Label("", id="view-mode-label")
        # --- Loading indicator (visible during async data loading) ---
        with Vertical(id="dashboard-loading-indicator"):
            yield Label("Loading dashboard data...", id="loading-spinner")
            yield Label(
                "Parsing session files and computing metrics",
                id="loading-message",
            )
        # --- Empty state (visible when no sessions exist) ---
        with Vertical(id="dashboard-empty-state"):
            yield Label(
                "No sessions found",
                id="empty-state-title",
            )
            yield Label(
                "Point to a flows directory with --flows-dir to get started",
                id="empty-state-hint",
            )
        # --- Regular dashboard content ---
        with VerticalScroll(id="dashboard-content"):
            yield MetricsPanel(id="dashboard-metrics")
            yield SessionBarChart(
                title="API Calls per Session",
                value_label="Calls",
                id="chart-call-counts",
            )
            yield TokenBarChart(
                title="Token Usage per Session",
                id="chart-token-usage",
            )
            yield AvgTokensChart(id="avg-tokens-chart")
            yield TimingChart(id="chart-timing")
            yield ToolUsageChart(
                title="Tool Usage (All Sessions Combined)",
                top_n=10,
                id="chart-tool-usage",
            )
            yield CacheEfficiencyChart(id="chart-cache-efficiency")
            yield ModelUsageChart(id="chart-model-usage")
            yield Label("", id="session-errors-warning")
            yield Input(placeholder="Filter sessions by name...", id="session-filter")
            yield DataTable(id="sessions-table", cursor_type="row")
        # --- Comparison mode container (hidden by default) ---
        with VerticalScroll(id="comparison-container"):
            yield Label(
                "  [bold]Comparison Mode[/]  Select sessions with Enter on filtered table  ",
                id="comparison-header",
            )
            yield Horizontal(id="comparison-panels")
        # --- Drill-down panel (hidden by default) ---
        yield ComparisonPanel(
            session_index=0,
            session_label="",
            id="drill-down-panel",
        )
        yield Label(
            "Enter:Focus Session  Backspace:All Sessions  r:Refresh  ?:Help  b/Escape:Browse  q:Quit  Up/Down/Tab:Navigate",
            id="dashboard-hint",
        )
        yield AppFooter()

    def on_mount(self) -> None:
        """Set up the screen after mounting.

        Shows a loading indicator and starts a background worker to
        populate session metrics, charts, and the DataTable.
        The worker runs the data computation in a thread pool executor
        (separated from UI updates) so the TUI remains responsive.
        """
        self._show_loading()
        self._load_data_async()

    def on_unmount(self) -> None:
        """Clean up when the screen is removed.

        Cancels all tracked background workers so the app can shut
        down promptly when the user presses ``q`` instead of waiting
        for blocked parquet reads to complete (VAL-GEN-001, VAL-GEN-005).
        """
        self._cancel_all_workers()

    def _cancel_all_workers(self) -> None:
        """Cancel all tracked background workers and clear the list.

        Each worker's cancel() method sets the underlying asyncio task
        as cancelled so the event loop does not wait for it to finish.
        Already-finished or already-cancelled workers are skipped.
        """
        for worker in list(self._workers):
            if not worker.is_finished and not worker.is_cancelled:
                try:
                    worker.cancel()
                except Exception:
                    logger.debug(
                        "Exception cancelling worker '%s'", worker.name,
                    )
        self._workers.clear()

    # ------------------------------------------------------------------
    # Edge state management
    # ------------------------------------------------------------------

    def _show_loading(self) -> None:
        """Show the loading indicator and hide regular content."""
        try:
            self.query_one("#dashboard-content").display = False
        except Exception:
            pass
        try:
            self.query_one("#dashboard-empty-state", Vertical).display = False
        except Exception:
            pass
        try:
            indicator = self.query_one("#dashboard-loading-indicator", Vertical)
            indicator.display = True
            indicator.add_class("visible")
        except Exception:
            pass

    def _hide_loading(self) -> None:
        """Hide the loading indicator."""
        try:
            indicator = self.query_one("#dashboard-loading-indicator", Vertical)
            indicator.display = False
            indicator.remove_class("visible")
        except Exception:
            pass

    def _show_empty_state(self) -> None:
        """Show the empty state message and hide regular content."""
        if self.app.screen is not self:
            return
        try:
            self.query_one("#dashboard-content").display = False
        except Exception:
            pass
        self._hide_loading()
        try:
            empty_state = self.query_one("#dashboard-empty-state", Vertical)
            empty_state.display = True
            empty_state.add_class("visible")
        except Exception:
            pass

    def _hide_empty_state(self) -> None:
        """Hide the empty state message."""
        if self.app.screen is not self:
            return
        try:
            empty_state = self.query_one("#dashboard-empty-state", Vertical)
            empty_state.display = False
            empty_state.remove_class("visible")
        except Exception:
            pass

    def _show_dashboard_content(self) -> None:
        """Show the regular dashboard content and hide loading/empty states."""
        if self.app.screen is not self:
            return
        self._hide_loading()
        self._hide_empty_state()
        try:
            self.query_one("#dashboard-content").display = True
        except Exception:
            pass

    def _update_errors_warning(self) -> None:
        """Show or hide the warning about sessions excluded due to errors.

        Displays a bold warning message when one or more sessions had
        parse errors and were excluded from aggregate metrics.
        The warning includes a count of affected sessions.
        """
        try:
            warning_label = self.query_one("#session-errors-warning", Label)
            if self._session_errors:
                error_count = len(self._session_errors)
                sessions = discover_sessions(self._flows_dir)
                total_sessions = len(sessions)
                excluded_names = []
                for sinfo in sessions:
                    if sinfo.index in self._session_errors:
                        excluded_names.append(f"{sinfo.index:02d} | {sinfo.task_name}")

                if error_count == 1:
                    msg = (
                        f"\u26a0 1 session excluded due to parse errors: "
                        f"{excluded_names[0] if excluded_names else 'unknown'}"
                    )
                else:
                    excluded_str = ", ".join(excluded_names) if excluded_names else "unknown"
                    msg = (
                        f"\u26a0 {error_count} of {total_sessions} sessions "
                        f"excluded due to parse errors: {excluded_str}"
                    )
                warning_label.update(msg)
                warning_label.display = True
                warning_label.add_class("visible")
            else:
                warning_label.display = False
                warning_label.remove_class("visible")
        except Exception:
            logger.exception("Failed to update errors warning")

    def _load_data_async(self) -> None:
        """Start a background worker to load dashboard data.

        The worker runs the data computation (parquet reads, session
        discovery) in a thread pool executor to keep the TUI responsive.
        UI updates (chart renders, table population) are done on the
        main thread after the data is ready.
        """

        async def _do_load() -> None:
            loop = asyncio.get_running_loop()
            try:
                # Fast path: try loading from cache first
                loaded_from_cache = self._load_metrics_from_cache()
                if loaded_from_cache:
                    logger.info(
                        "Dashboard metrics loaded from cache (fast path)",
                    )
                else:
                    # Slow path: compute metrics from parquet files
                    await loop.run_in_executor(
                        None, self._populate_data,
                    )
                    # Save to cache for next time (in executor to avoid blocking)
                    await loop.run_in_executor(
                        None, self._save_metrics_to_cache,
                    )
                # UI updates on main thread (not thread-safe)
                self._update_metrics_panel()
                self._update_charts()
                self._populate_table()
            except Exception as exc:
                logger.error("Failed to load dashboard data: %s", exc)
                self._session_errors[-1] = f"Dashboard load error: {exc}"
            finally:
                self._on_data_loaded()

        worker = self.run_worker(_do_load(), name="dashboard-load")
        self._workers.append(worker)

    def _on_data_loaded(self) -> None:
        """Handle completion of async data loading.

        Transitions to the appropriate visual state based on whether
        sessions were discovered and loaded successfully.
        Guards against stale updates if the screen was popped during
        loading.
        """
        self._data_loaded = True
        if self.app.screen is not self:
            return

        sessions = discover_sessions(self._flows_dir)

        if not sessions:
            self._show_empty_state()
        elif self._session_errors and len(self._session_errors) >= len(sessions):
            self._show_dashboard_content()
        else:
            self._show_dashboard_content()

        # Show warning about excluded sessions (VAL-CROSS-023)
        self._update_errors_warning()

        # Set focus to the DataTable for keyboard interaction
        try:
            self.query_one("#sessions-table", DataTable).focus()
        except Exception:
            pass

    def _refresh_all(self) -> None:
        """Refresh all dashboard data: metrics, charts, and table.

        Recomputes all metrics, updates the cache file, and refreshes
        all UI widgets.  Used by synchronous callers (comparison mode
        toggle, etc.) and the manual refresh action.
        """
        self._populate_data()
        self._save_metrics_to_cache()
        self._update_metrics_panel()
        self._update_charts()
        self._populate_table()
        self._update_errors_warning()

    def action_refresh_dashboard(self) -> None:
        """Force a full refresh of all dashboard data.

        Re-discovers sessions, re-computes all metrics from the
        underlying parquet cache files, and updates all widgets
        (metrics panel, charts, DataTable) with the latest data.

        This action is bound to the ``r`` key and provides a manual
        live-data refresh mechanism (VAL-CROSS-012).  The loading
        indicator is shown before data computation starts and hidden
        when data is ready.  Computation runs in a thread pool
        executor so the TUI remains responsive and the loading
        indicator is rendered before blocking work begins.
        """
        self._show_loading()

        async def _do_refresh() -> None:
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(
                    None, self._populate_data,
                )
                await loop.run_in_executor(
                    None, self._save_metrics_to_cache,
                )
                self._update_metrics_panel()
                self._update_charts()
                self._populate_table()
                self._update_errors_warning()
            except Exception as exc:
                logger.error(
                    "Failed to refresh dashboard: %s", exc,
                )
            finally:
                self._on_data_loaded()

        worker = self.run_worker(_do_refresh(), name="dashboard-refresh")
        self._workers.append(worker)

    def _populate_data(self) -> None:
        """Discover sessions and compute metrics for each.

        Populates self._session_data and self._timing_data caches.
        Per-session errors are captured in self._session_errors.
        """
        sessions = discover_sessions(self._flows_dir)
        self._session_data = []
        self._timing_data = []
        self._tool_usage_data = []
        self._cache_efficiency_data = []
        self._model_usage_data = []
        self._session_errors = {}
        for sinfo in sessions:
            try:
                call_count, total_input, total_output = self._compute_metrics_fast(sinfo)
            except Exception as exc:
                logger.warning("Failed to compute metrics for session %s: %s",
                               sinfo.task_name, exc)
                self._session_errors[sinfo.index] = str(exc)
                self._session_data.append((sinfo, 0, 0, 0))
            else:
                self._session_data.append((sinfo, call_count, total_input, total_output))

            # Compute timing metrics
            try:
                label = f"{sinfo.index:02d} | {sinfo.task_name}"
                min_rtt, avg_rtt, max_rtt = self._compute_timing_metrics_fast(sinfo)
            except Exception as exc:
                logger.warning("Failed to compute timing for session %s: %s",
                               sinfo.task_name, exc)
                if sinfo.index not in self._session_errors:
                    self._session_errors[sinfo.index] = f"Timing error: {exc}"
                label = f"{sinfo.index:02d} | {sinfo.task_name}"
                min_rtt, avg_rtt, max_rtt = None, None, None
            self._timing_data.append((label, min_rtt, avg_rtt, max_rtt))

            # Compute tool usage
            try:
                self._tool_usage_data.append(self._compute_tool_usage_fast(sinfo))
            except Exception as exc:
                logger.warning("Failed to compute tool usage for session %s: %s",
                               sinfo.task_name, exc)
                if sinfo.index not in self._session_errors:
                    self._session_errors[sinfo.index] = f"Tool usage error: {exc}"
                self._tool_usage_data.append({})

            # Compute cache efficiency
            try:
                cache_read, input_tokens = self._compute_cache_efficiency_fast(sinfo)
            except Exception as exc:
                logger.warning("Failed to compute cache efficiency for session %s: %s",
                               sinfo.task_name, exc)
                if sinfo.index not in self._session_errors:
                    self._session_errors[sinfo.index] = f"Cache error: {exc}"
                cache_read, input_tokens = None, None
            self._cache_efficiency_data.append((cache_read, input_tokens))

            # Compute model usage
            try:
                self._model_usage_data.append(self._compute_model_usage_fast(sinfo))
            except Exception as exc:
                logger.warning("Failed to compute model usage for session %s: %s",
                               sinfo.task_name, exc)
                if sinfo.index not in self._session_errors:
                    self._session_errors[sinfo.index] = f"Model usage error: {exc}"
                self._model_usage_data.append({})

    def _compute_overall_metrics(self) -> Dict[str, int]:
        """Compute aggregate metrics across all (or focused) sessions.

        Returns:
            A dict with ``total_sessions``, ``total_calls``,
            ``total_input_tokens``, ``total_output_tokens``.
        """
        if not self._session_data:
            return {
                "total_sessions": 0,
                "total_calls": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
            }

        # If a single session is focused, show only that session's data
        if self._focused_session_index is not None:
            for sinfo, ccount, inp, out in self._session_data:
                if sinfo.index == self._focused_session_index:
                    return {
                        "total_sessions": 1,
                        "total_calls": ccount,
                        "total_input_tokens": inp,
                        "total_output_tokens": out,
                    }
            # Fallback: focused session not found in data
            return {
                "total_sessions": 0,
                "total_calls": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
            }

        # All sessions
        total_calls = sum(d[1] for d in self._session_data)
        total_input = sum(d[2] for d in self._session_data)
        total_output = sum(d[3] for d in self._session_data)
        return {
            "total_sessions": len(self._session_data),
            "total_calls": total_calls,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
        }

    def _update_metrics_panel(self) -> None:
        """Update the MetricsPanel with current aggregate metrics."""
        metrics = self._compute_overall_metrics()
        metrics_panel = self.query_one("#dashboard-metrics", MetricsPanel)
        metrics_panel.metrics_data = (
            metrics["total_sessions"],
            metrics["total_calls"],
            metrics["total_input_tokens"],
            metrics["total_output_tokens"],
        )

    def _get_filtered_data(
        self,
    ) -> List[Tuple[SessionInfo, int, int, int]]:
        """Return session data filtered by focus state.

        If a single session is focused, returns only that session's data.
        Otherwise returns all sessions.

        Returns:
            A list of ``(SessionInfo, call_count, input_tokens, output_tokens)``.
        """
        if self._focused_session_index is not None:
            result = []
            for sinfo, ccount, inp, out in self._session_data:
                if sinfo.index == self._focused_session_index:
                    result.append((sinfo, ccount, inp, out))
                    break
            return result
        return list(self._session_data)

    def _update_charts(self) -> None:
        """Update all chart widgets with current session data."""
        filtered = self._get_filtered_data()

        if not filtered:
            # No data — clear charts
            chart_calls = self.query_one("#chart-call-counts", SessionBarChart)
            chart_calls.chart_data = []
            chart_tokens = self.query_one("#chart-token-usage", TokenBarChart)
            chart_tokens.chart_data = []
            avg_chart = self.query_one("#avg-tokens-chart", AvgTokensChart)
            avg_chart.avg_data = []
            timing_chart = self.query_one("#chart-timing", TimingChart)
            timing_chart.timing_data = []
            tool_chart = self.query_one("#chart-tool-usage", ToolUsageChart)
            tool_chart.update_data([])
            cache_chart = self.query_one("#chart-cache-efficiency", CacheEfficiencyChart)
            cache_chart.cache_data = []
            model_chart = self.query_one("#chart-model-usage", ModelUsageChart)
            model_chart.update_data([])
            return

        # Call counts bar chart
        calls_data: List[Tuple[str, int]] = []
        for sinfo, ccount, _, _ in filtered:
            calls_data.append((f"{sinfo.index:02d} | {sinfo.task_name}", ccount))
        chart_calls = self.query_one("#chart-call-counts", SessionBarChart)
        chart_calls.update_data(calls_data)

        # Token usage bar chart (split input / output — VAL-DASH-003)
        tokens_data: List[Tuple[str, int, int]] = []
        for sinfo, _, inp, out in filtered:
            tokens_data.append((f"{sinfo.index:02d} | {sinfo.task_name}", inp, out))
        chart_tokens = self.query_one("#chart-token-usage", TokenBarChart)
        chart_tokens.update_data(tokens_data)

        # Average tokens per call chart
        avg_data: List[Tuple[str, float]] = []
        total_avg_dividend = 0
        total_avg_divisor = 0
        for sinfo, ccount, inp, out in filtered:
            if ccount > 0:
                total_tok = inp + out
                avg = total_tok / ccount
                avg_data.append((f"{sinfo.index:02d} | {sinfo.task_name}", avg))
                total_avg_dividend += total_tok
                total_avg_divisor += ccount
        overall_avg = total_avg_dividend / total_avg_divisor if total_avg_divisor > 0 else 0

        avg_chart = self.query_one("#avg-tokens-chart", AvgTokensChart)
        avg_chart.update_data(avg_data, overall_avg=overall_avg)

        # Timing chart
        if self._focused_session_index is not None:
            timing_filtered = []
            for label, mn, avg, mx in self._timing_data:
                if str(self._focused_session_index).zfill(2) in label:
                    timing_filtered.append((label, mn, avg, mx))
                    break
        else:
            timing_filtered = list(self._timing_data)

        timing_chart = self.query_one("#chart-timing", TimingChart)
        timing_chart.update_data(timing_filtered)

        # Tool usage chart
        tool_chart = self.query_one("#chart-tool-usage", ToolUsageChart)

        if self._focused_session_index is not None:
            # Show tool usage for the focused session only
            for sinfo, _ccount, _inp, _out in filtered:
                if sinfo.index == self._focused_session_index:
                    # Find tool usage for this session
                    for i, s in enumerate(discover_sessions(self._flows_dir)):
                        if s.index == self._focused_session_index and i < len(self._tool_usage_data):
                            tool_counts = self._tool_usage_data[i]
                            tool_items: list[tuple[str, int]] = list(tool_counts.items())
                            tool_chart.border_title = f"Tool Usage — {sinfo.task_name}"
                            tool_chart.update_data(tool_items)
                            break
                    break
            else:
                tool_chart.border_title = "Tool Usage (All Sessions Combined)"
                tool_chart.update_data([])
        else:
            # Aggregate tool usage across all sessions
            aggregated: dict[str, int] = {}
            for tu_data in self._tool_usage_data:
                for tool_name, count in tu_data.items():
                    aggregated[tool_name] = aggregated.get(tool_name, 0) + count
            tool_items = list(aggregated.items())
            tool_chart.border_title = "Tool Usage (All Sessions Combined)"
            tool_chart.update_data(tool_items)

        # --- Cache Efficiency chart ---
        cache_chart = self.query_one("#chart-cache-efficiency", CacheEfficiencyChart)
        if self._focused_session_index is not None:
            # Show cache efficiency for the focused session only
            cache_items: list[tuple[str, int | None, int | None]] = []
            for sinfo, _ccount, _inp, _out in filtered:
                if sinfo.index == self._focused_session_index:
                    for i, s in enumerate(discover_sessions(self._flows_dir)):
                        if s.index == self._focused_session_index and i < len(self._cache_efficiency_data):
                            cache_read, input_tok = self._cache_efficiency_data[i]
                            cache_items.append((
                                f"{sinfo.index:02d} | {sinfo.task_name}",
                                cache_read,
                                input_tok,
                            ))
                            break
                    break
            cache_chart.cache_data = cache_items
        else:
            # Show cache efficiency for all sessions
            cache_all: list[tuple[str, int | None, int | None]] = []
            for i, sinfo in enumerate(discover_sessions(self._flows_dir)):
                if i < len(self._cache_efficiency_data):
                    cache_read, input_tok = self._cache_efficiency_data[i]
                    cache_all.append((
                        f"{sinfo.index:02d} | {sinfo.task_name}",
                        cache_read,
                        input_tok,
                    ))
            cache_chart.cache_data = cache_all

        # --- Model Usage chart ---
        model_chart = self.query_one("#chart-model-usage", ModelUsageChart)
        if self._focused_session_index is not None:
            # Show model usage for the focused session only
            for sinfo, _ccount, _inp, _out in filtered:
                if sinfo.index == self._focused_session_index:
                    for i, s in enumerate(discover_sessions(self._flows_dir)):
                        if s.index == self._focused_session_index and i < len(self._model_usage_data):
                            model_counts = self._model_usage_data[i]
                            model_items = list(model_counts.items())
                            model_chart.border_title = f"Model Usage — {sinfo.task_name}"
                            model_chart.update_data(model_items)
                            break
                    break
            else:
                model_chart.border_title = "Model Usage (All Sessions Combined)"
                model_chart.update_data([])
        else:
            # Aggregate model usage across all sessions
            aggregated_models: dict[str, int] = {}
            for mu_data in self._model_usage_data:
                for model_name, count in mu_data.items():
                    aggregated_models[model_name] = aggregated_models.get(model_name, 0) + count
            model_items = list(aggregated_models.items())
            model_chart.border_title = "Model Usage (All Sessions Combined)"
            model_chart.update_data(model_items)

        # Update view mode label
        view_label = self.query_one("#view-mode-label", Label)
        if self._focused_session_index is not None:
            focused_name = ""
            for sinfo, _, _, _ in filtered:
                if sinfo.index == self._focused_session_index:
                    focused_name = sinfo.task_name
                    break
            view_label.update(
                f"  Viewing: [bold]{focused_name}[/]  "
                "  [italic](Press Backspace for All Sessions)[/]  "
            )
        else:
            view_label.update("  Viewing: [bold]All Sessions[/]  ")

    def _populate_table(self) -> None:
        """Populate the session metrics DataTable.

        Uses cached session data for speed.  The focused session (if any)
        is highlighted with bold accent styling.  Sessions with errors
        show an error icon and the error message in lieu of metrics.
        """
        from rich.text import Text

        data_table = self.query_one("#sessions-table", DataTable)
        data_table.border_title = "Session Metrics"

        data_table.clear(columns=True)
        data_table.add_columns(
            "Session",
            "Calls",
            "Input Tokens",
            "Output Tokens",
            "Total Tokens",
        )

        # Use the overall sessions so the table always shows all sessions
        # (the charts are what narrows)
        sessions = discover_sessions(self._flows_dir)
        if not sessions:
            data_table.add_row("No sessions found", "", "", "", "")
            return

        # Determine the highlighted session from either constructor or app
        selected_idx = self._selected_session_index
        if selected_idx is None:
            try:
                selected_idx = self.app.selected_session_index  # type: ignore[union-attr]
            except AttributeError:
                pass

        # Recompute metrics for each session for consistency
        session_metrics = []
        for sinfo in sessions:
            ccount, inp, out = self._compute_metrics_fast(sinfo)
            session_metrics.append((sinfo, ccount, inp, out))

        for sinfo, ccount, inp, out in session_metrics:
            total_tokens = inp + out
            label = f"{sinfo.index:02d} | {sinfo.task_name}"

            # Apply type-to-filter: case-insensitive substring match
            if self._filter_text and self._filter_text.lower() not in label.lower():
                continue

            # Check if this session has an error
            error_msg = self._session_errors.get(sinfo.index)
            if error_msg:
                input_str = "—"
                output_str = "—"
                total_str = "—"
                ccount_str = "—"
                label = f"⚠ {label}"
            else:
                input_str = f"{inp:,}" if inp else "0"
                output_str = f"{out:,}" if out else "0"
                total_str = f"{total_tokens:,}" if total_tokens else "0"
                ccount_str = str(ccount)

            # Highlight: either the selected-from-browse session or the focused session
            is_selected = (
                (selected_idx is not None and sinfo.index == selected_idx)
                or (self._focused_session_index is not None
                    and sinfo.index == self._focused_session_index)
            )

            if is_selected:
                label_cell = Text(label, style="bold $accent")
                calls_cell = Text(ccount_str, style="bold $accent")
                input_cell = Text(input_str, style="bold $accent")
                output_cell = Text(output_str, style="bold $accent")
                total_cell = Text(total_str, style="bold $accent")
            else:
                if error_msg:
                    label_cell = Text(label, style="red")
                    calls_cell = Text(ccount_str, style="red")
                    input_cell = Text(input_str, style="red")
                    output_cell = Text(output_str, style="red")
                    total_cell = Text(total_str, style="red")
                else:
                    label_cell = label
                    calls_cell = ccount_str
                    input_cell = input_str
                    output_cell = output_str
                    total_cell = total_str

            row = [label_cell, calls_cell, input_cell, output_cell, total_cell]
            data_table.add_row(*row)

        # Add error detail rows if any sessions have errors
        if self._session_errors:
            for sinfo, _, _, _ in session_metrics:
                error_msg = self._session_errors.get(sinfo.index)
                if error_msg:
                    error_text = Text(
                        f"  Error: {error_msg}",
                        style="italic red",
                    )
                    data_table.add_row(
                        error_text, "", "", "", "",
                    )
            # Add a blank separator after error details
            data_table.add_row("", "", "", "", "")

    def on_input_changed(self, event: Input.Changed) -> None:
        """Filter the DataTable rows in real-time based on input text.

        Matches the input value case-insensitively against session labels
        (e.g. ``"01 | analyze_codebase"``).  Cleared input shows all sessions.

        Args:
            event: The ``Input.Changed`` message from the filter Input.
        """
        self._filter_text = event.value
        # Re-populate table immediately (no debounce — real-time filtering)
        self._populate_table()
        # Re-apply session focus highlighting
        self._update_charts()

    def action_clear_session_focus(self) -> None:
        """Clear the focused session and show all sessions (Backspace key)."""
        if self._focused_session_index is not None:
            self._focused_session_index = None
            self._update_metrics_panel()
            self._update_charts()
            self._populate_table()

    # ------------------------------------------------------------------
    # Session focus actions
    # ------------------------------------------------------------------

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle Enter on a DataTable row — focus that session.

        When a session row is selected (Enter key), all charts narrow
        to that session only.  Pressing Backspace returns to the
        all-sessions overview.

        Args:
            event: The ``RowSelected`` message from the DataTable.
        """
        cursor_row = event.cursor_row
        if cursor_row is None:
            return

        table = self.query_one("#sessions-table", DataTable)
        row_data = table.get_row_at(cursor_row)
        if not row_data or not row_data[0]:
            return

        label_cell = str(row_data[0])

        # Parse session index from label (format: "01 | task_name")
        import re as _re
        m = _re.match(r"^(\d+)", label_cell)
        if m:
            idx = int(m.group(1))
            if self._focused_session_index != idx:
                self._focused_session_index = idx
                self._update_metrics_panel()
                self._update_charts()
                self._populate_table()
                # Propagate the focused session to Browse for cross-view
                try:
                    self.app.selected_session_index = idx  # type: ignore[union-attr]
                except AttributeError:
                    pass

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def _compute_metrics_fast(
        self,
        session_info: SessionInfo,
    ) -> Tuple[int, int, int]:
        """Compute session metrics from parquet cache files (fast path).

        Reads the response parquet file for the given session and sums
        token columns, counts rows (calls).  Falls back to full session
        loading if parquet is unavailable.

        Args:
            session_info: The session to compute metrics for.

        Returns:
            A tuple ``(call_count, total_input_tokens, total_output_tokens)``.
        """
        import pyarrow.parquet as pq

        flow_dir = os.path.dirname(session_info.file_path)
        flow_name = os.path.basename(session_info.file_path)
        for fname in os.listdir(flow_dir):
            if fname.endswith("_responses.parquet") and flow_name in fname:
                resp_path = os.path.join(flow_dir, fname)
                try:
                    table = pq.read_table(resp_path, columns=[
                        "input_tokens", "output_tokens",
                    ])
                    call_count = table.num_rows
                    total_input = sum(
                        v for v in table.column("input_tokens").to_pylist()
                        if v is not None
                    )
                    total_output = sum(
                        v for v in table.column("output_tokens").to_pylist()
                        if v is not None
                    )
                    return call_count, total_input, total_output
                except Exception as exc:
                    logger.debug(
                        "Could not read parquet for %s: %s",
                        flow_name, exc,
                    )
                    break

        # Fallback: full session loading
        session = self._load_session_data(session_info)
        if session is None:
            return 0, 0, 0
        call_count = len(session.calls)
        total_input = 0
        total_output = 0
        for call in session.calls:
            if call.response:
                total_input += call.response.input_tokens or 0
                total_output += call.response.output_tokens or 0
        return call_count, total_input, total_output

    def _compute_timing_metrics_fast(
        self,
        session_info: SessionInfo,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Compute min/avg/max RTT from parquet cache files (fast path).

        Reads both request and response parquet files for the given session,
        computes round-trip time (RTT = response_end - request_start) per
        call, and returns min, avg, max.  Falls back to full session loading
        if parquet is unavailable.

        Args:
            session_info: The session to compute timing metrics for.

        Returns:
            A tuple ``(min_rtt_sec, avg_rtt_sec, max_rtt_sec)``.
            Any value may be ``None`` if timing data is unavailable.
        """
        import pyarrow.parquet as pq

        flow_dir = os.path.dirname(session_info.file_path)
        flow_name = os.path.basename(session_info.file_path)

        req_path: str | None = None
        resp_path: str | None = None
        for fname in os.listdir(flow_dir):
            if fname.endswith("_requests.parquet") and flow_name in fname:
                req_path = os.path.join(flow_dir, fname)
            elif fname.endswith("_responses.parquet") and flow_name in fname:
                resp_path = os.path.join(flow_dir, fname)

        if req_path is not None and resp_path is not None:
            try:
                req_table = pq.read_table(
                    req_path, columns=["request_id", "timestamp_start"],
                )
                resp_table = pq.read_table(
                    resp_path, columns=["request_id", "timestamp_end"],
                )

                req_dict = req_table.to_pydict()
                resp_dict = resp_table.to_pydict()

                # Build request lookup: request_id → timestamp_start
                req_start_by_id: Dict[str, Optional[float]] = {}
                for i in range(len(req_dict["request_id"])):
                    rid = str(req_dict["request_id"][i])
                    req_start_by_id[rid] = req_dict["timestamp_start"][i]

                # Build response lookup: request_id → timestamp_end
                resp_end_by_id: Dict[str, Optional[float]] = {}
                for i in range(len(resp_dict["request_id"])):
                    rid = str(resp_dict["request_id"][i])
                    resp_end_by_id[rid] = resp_dict["timestamp_end"][i]

                # Compute RTT for each paired request/response
                rtts: List[float] = []
                for rid, req_start in req_start_by_id.items():
                    resp_end = resp_end_by_id.get(rid)
                    if req_start is not None and resp_end is not None:
                        rtt = resp_end - req_start
                        if rtt >= 0:
                            rtts.append(rtt)

                if rtts:
                    return min(rtts), sum(rtts) / len(rtts), max(rtts)
            except Exception as exc:
                logger.debug(
                    "Could not read timing parquet for %s: %s",
                    flow_name, exc,
                )

        # Fallback: full session loading
        session = self._load_session_data(session_info)
        if session is None or not session.calls:
            return None, None, None

        rtts = []
        for call in session.calls:
            if call.timing is not None:
                req_start = call.timing.request_start
                resp_end = call.timing.response_end
                if req_start is not None and resp_end is not None:
                    rtt = resp_end - req_start
                    if rtt >= 0:
                        rtts.append(rtt)

        if not rtts:
            return None, None, None

        return min(rtts), sum(rtts) / len(rtts), max(rtts)

    def _compute_tool_usage_fast(
        self,
        session_info: SessionInfo,
    ) -> dict[str, int]:
        """Count tool invocations for a session from parquet (fast path).

        Reads the ``tool_uses`` JSON column from the response parquet file
        and parses it to extract tool names.  Falls back to full session
        loading if the parquet cache is unavailable.

        Args:
            session_info: The session to compute tool usage for.

        Returns:
            A dict mapping tool name → invocation count (e.g.
            ``{"Read": 42, "Grep": 17, ...}``).  Returns an empty dict if
            the session has no data or an error occurred.
        """
        import json
        import pyarrow.parquet as pq

        flow_dir = os.path.dirname(session_info.file_path)
        flow_name = os.path.basename(session_info.file_path)

        for fname in os.listdir(flow_dir):
            if fname.endswith("_responses.parquet") and flow_name in fname:
                resp_path = os.path.join(flow_dir, fname)
                try:
                    table = pq.read_table(resp_path, columns=["tool_uses"])
                    tool_counts: dict[str, int] = {}
                    for row in table.column("tool_uses").to_pylist():
                        if row:
                            try:
                                tool_uses = json.loads(row)
                                for tu in tool_uses:
                                    name = tu.get("name", "")
                                    if name:
                                        tool_counts[name] = tool_counts.get(name, 0) + 1
                            except json.JSONDecodeError:
                                pass
                    return tool_counts
                except Exception as exc:
                    logger.debug(
                        "Could not read tool_uses parquet for %s: %s",
                        flow_name, exc,
                    )
                    break

        # Fallback: full session loading
        from collections import Counter
        session = self._load_session_data(session_info)
        if session is None:
            return {}
        counter: Counter = Counter()
        for call in session.calls:
            if call.response:
                for tool_use in call.response.tool_uses:
                    counter[tool_use.name] += 1
        return dict(counter)

    def _compute_cache_efficiency_fast(
        self,
        session_info: SessionInfo,
    ) -> Tuple[Optional[int], Optional[int]]:
        """Compute cache read and total input tokens from parquet (fast path).

        Reads the ``cache_read_input_tokens`` and ``input_tokens`` columns
        from the response parquet file for the given session and sums them.
        Falls back to full session loading if parquet is unavailable.

        Args:
            session_info: The session to compute cache efficiency for.

        Returns:
            A tuple ``(total_cache_read, total_input_tokens)``.  Either value
            may be ``None`` if the data is unavailable.
        """
        import pyarrow.parquet as pq

        flow_dir = os.path.dirname(session_info.file_path)
        flow_name = os.path.basename(session_info.file_path)
        for fname in os.listdir(flow_dir):
            if fname.endswith("_responses.parquet") and flow_name in fname:
                resp_path = os.path.join(flow_dir, fname)
                try:
                    table = pq.read_table(resp_path, columns=[
                        "cache_read_input_tokens", "input_tokens",
                    ])
                    cache_read_values = table.column("cache_read_input_tokens").to_pylist()
                    input_values = table.column("input_tokens").to_pylist()
                    total_cache_read = sum(
                        v for v in cache_read_values if v is not None
                    )
                    total_input = sum(
                        v for v in input_values if v is not None
                    )
                    return total_cache_read, total_input
                except Exception as exc:
                    logger.debug(
                        "Could not read cache parquet for %s: %s",
                        flow_name, exc,
                    )
                    break

        # Fallback: full session loading
        session = self._load_session_data(session_info)
        if session is None:
            return None, None
        total_cache_read = 0
        total_input = 0
        for call in session.calls:
            if call.response:
                cr = call.response.cache_read_input_tokens
                if cr is not None:
                    total_cache_read += cr
                inp = call.response.input_tokens
                if inp is not None:
                    total_input += inp
        if total_cache_read == 0 and total_input == 0:
            return None, None
        return total_cache_read, total_input

    def _compute_model_usage_fast(
        self,
        session_info: SessionInfo,
    ) -> dict[str, int]:
        """Count model invocations from request parquet (fast path).

        Reads the ``model`` column from the request parquet file and
        counts occurrences of each model.  Falls back to full session
        loading if parquet is unavailable.

        Args:
            session_info: The session to compute model usage for.

        Returns:
            A dict mapping model name → call count (e.g.
            ``{"deepseek-v4-flash": 10}``).  Returns an empty dict if no
            data is available.
        """
        import pyarrow.parquet as pq

        flow_dir = os.path.dirname(session_info.file_path)
        flow_name = os.path.basename(session_info.file_path)
        for fname in os.listdir(flow_dir):
            if fname.endswith("_requests.parquet") and flow_name in fname:
                req_path = os.path.join(flow_dir, fname)
                try:
                    table = pq.read_table(req_path, columns=["model"])
                    model_counts: dict[str, int] = {}
                    for model_val in table.column("model").to_pylist():
                        if model_val:
                            model_counts[model_val] = model_counts.get(model_val, 0) + 1
                    return model_counts
                except Exception as exc:
                    logger.debug(
                        "Could not read model parquet for %s: %s",
                        flow_name, exc,
                    )
                    break

        # Fallback: full session loading
        from collections import Counter
        session = self._load_session_data(session_info)
        if session is None:
            return {}
        counter: Counter = Counter()
        for call in session.calls:
            if call.request and call.request.model:
                counter[call.request.model] += 1
        return dict(counter)

    def _load_session_data(self, session_info: SessionInfo):
        """Load session data for metric computation.

        Args:
            session_info: The session to load.

        Returns:
            The parsed :class:`~llm_flow_viewer.parser.models.Session`, or
            ``None`` if loading fails.
        """
        try:
            from llm_flow_viewer.parser.session import flow_file_to_session
            return flow_file_to_session(
                session_info.file_path,
                session_info.index,
                session_info.task_name,
            )
        except Exception as exc:
            logger.warning(
                "Failed to load session %s for dashboard: %s",
                session_info.task_name,
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Comparison Mode
    # ------------------------------------------------------------------

    def action_toggle_comparison(self) -> None:
        """Toggle comparison mode on/off.

        When activated, hides the regular dashboard charts and table,
        and shows side-by-side comparison panels.  Number keys (1-7)
        toggle session selection instead of focusing individual sessions.
        """
        self._comparison_mode = not self._comparison_mode

        if self._comparison_mode:
            # Clear any existing focused session and drill-down
            self._focused_session_index = None
            self._drill_down_session_index = None

            # Show comparison container, hide dashboard content
            self.query_one("#comparison-container").add_class("visible")
            self.query_one("#dashboard-content").display = False
            self.query_one("#dashboard-hint").display = False
            self.query_one("#dashboard-metrics").display = False

            self._rebuild_comparison_panels()
        else:
            # Hide comparison container, show dashboard content
            self.query_one("#comparison-container").remove_class("visible")
            self.query_one("#drill-down-panel", ComparisonPanel).remove_class("visible")
            self.query_one("#dashboard-content").display = True
            self.query_one("#dashboard-hint").display = True
            self.query_one("#dashboard-metrics").display = True
            self._compare_session_indices.clear()
            self._drill_down_session_index = None

            # Refresh dashboard data
            self._refresh_all()

    def _rebuild_comparison_panels(self) -> None:
        """Rebuild the comparison panels based on selected sessions.

        Removes existing panels and creates new ones for each selected
        session index.  Stores the panels inside ``#comparison-panels``.
        """
        # Skip if not in comparison mode
        if not self._comparison_mode:
            return

        panels_container = self.query_one("#comparison-panels")
        # Remove existing panels
        panels_container.remove_children()

        # No sessions selected yet — show a hint
        if not self._compare_session_indices:
            from textual.widgets import Label
            hint = Label(
                "  Select sessions using Enter on filtered DataTable rows  ",
                id="comparison-empty-hint",
            )
            panels_container.mount(hint)
            return

        # Create a panel for each selected session
        for idx in sorted(self._compare_session_indices):
            panel = self._build_comparison_panel(idx)
            if panel is not None:
                panels_container.mount(panel)

        # Focus the first panel
        first_panel = panels_container.query("ComparisonPanel").first()
        if first_panel:
            first_panel.focus()

    def _build_comparison_panel(
        self, session_index: int,
    ) -> ComparisonPanel | None:
        """Build a ComparisonPanel for the given session index.

        Gathers metrics from the session data cache and returns a
        :class:`ComparisonPanel` widget.

        Args:
            session_index: The session index (1-7) to build a panel for.

        Returns:
            A ``ComparisonPanel`` instance, or ``None`` if the session is
            not found.
        """
        from llm_flow_viewer.tui.widgets.dashboard_widgets import (
            ComparisonPanel,
        )

        sessions = discover_sessions(self._flows_dir)
        sinfo = None
        for s in sessions:
            if s.index == session_index:
                sinfo = s
                break

        if sinfo is None:
            return None

        label = f"{sinfo.index:02d} | {sinfo.task_name}"

        # Find cached metrics
        call_count = 0
        total_input = 0
        total_output = 0
        avg_rtt: float | None = None

        for s_entry, cc, inp, out in self._session_data:
            if s_entry.index == session_index:
                call_count = cc
                total_input = inp
                total_output = out
                break

        for t_entry, mn, avg, mx in self._timing_data:
            if str(session_index).zfill(2) in t_entry:
                avg_rtt = avg
                break

        # Find tool usage
        tool_usage: dict[str, int] = {}
        for i, s in enumerate(sessions):
            if s.index == session_index and i < len(self._tool_usage_data):
                tool_usage = self._tool_usage_data[i]
                break

        return ComparisonPanel(
            session_index=session_index,
            session_label=label,
            call_count=call_count,
            input_tokens=total_input,
            output_tokens=total_output,
            avg_rtt=avg_rtt,
            tool_usage=tool_usage,
        )

    # ------------------------------------------------------------------
    # Drill-down support
    # ------------------------------------------------------------------

    def _exit_drill_down(self) -> None:
        """Exit drill-down mode back to comparison overview."""
        if self._drill_down_session_index is not None:
            self._drill_down_session_index = None
            # Hide drill-down panel
            drill_panel = self.query_one("#drill-down-panel", ComparisonPanel)
            drill_panel.remove_class("visible")
            # Show comparison panels
            self.query_one("#comparison-container").add_class("visible")
            self.query_one("#comparison-panels").display = True

    def action_handle_escape(self) -> None:
        """Handle Escape key based on current mode.

        Priority:
        1. If in drill-down mode → exit drill-down (back to comparison)
        2. If in comparison mode → exit comparison (back to dashboard)
        3. Otherwise → switch to browse (pop screen)
        """
        if self._drill_down_session_index is not None:
            self._exit_drill_down()
        elif self._comparison_mode:
            # Exit comparison mode
            self._comparison_mode = False
            self._compare_session_indices.clear()
            self._drill_down_session_index = None

            self.query_one("#comparison-container").remove_class("visible")
            self.query_one("#drill-down-panel", ComparisonPanel).remove_class("visible")
            self.query_one("#dashboard-content").display = True
            self.query_one("#dashboard-hint").display = True
            self.query_one("#dashboard-metrics").display = True
            self._refresh_all()
        else:
            # Pop to browse
            if self.app.screen_stack:
                self.app.pop_screen()
                self.app._restore_call_tree_focus()
                # Propagate dashboard-focused session to browse
                if self._focused_session_index is not None:
                    try:
                        from llm_flow_viewer.tui.screens.browse import BrowseScreen
                        browse = self.app.screen
                        if isinstance(browse, BrowseScreen):
                            browse.load_session_by_index(self._focused_session_index)
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    # Drill-down support
    # ------------------------------------------------------------------

    def _enter_drill_down(self, session_index: int) -> None:
        """Enter drill-down mode for the given session.

        Hides the comparison panels and shows a single drill-down panel
        with detailed session metrics.

        Args:
            session_index: The session index to drill into.
        """
        self._drill_down_session_index = session_index

        # Hide comparison panels
        self.query_one("#comparison-container").remove_class("visible")
        self.query_one("#comparison-panels").display = False

        # Build detailed panel
        drill_panel = self.query_one("#drill-down-panel", ComparisonPanel)
        sessions = discover_sessions(self._flows_dir)
        sinfo = None
        for s in sessions:
            if s.index == session_index:
                sinfo = s
                break

        if sinfo:
            label = f"{sinfo.index:02d} | {sinfo.task_name}"
            # Find metrics
            call_count = 0
            total_input = 0
            total_output = 0
            avg_rtt = None
            for s_entry, cc, inp, out in self._session_data:
                if s_entry.index == session_index:
                    call_count = cc
                    total_input = inp
                    total_output = out
                    break
            for t_entry, mn, avg, mx in self._timing_data:
                if str(session_index).zfill(2) in t_entry:
                    avg_rtt = avg
                    break

            tool_usage = {}
            for i, s in enumerate(sessions):
                if s.index == session_index and i < len(self._tool_usage_data):
                    tool_usage = self._tool_usage_data[i]
                    break

            drill_panel.session_index = session_index
            drill_panel.session_label = label
            drill_panel.call_count = call_count
            drill_panel.input_tokens = total_input
            drill_panel.output_tokens = total_output
            drill_panel.avg_rtt = avg_rtt
            drill_panel.tool_usage = tool_usage
            drill_panel.border_title = f"Drill-Down — {label}"
            drill_panel.set_drill_down(True)

        drill_panel.add_class("visible")
        drill_panel.focus()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_switch_to_browse(self) -> None:
        """Return to the Browse view.

        Pops the current screen (Dashboard) from the screen stack,
        revealing the Browse screen that was pushed beneath it.
        After the pop, restores focus to the CallTree widget on the
        Browse screen for seamless keyboard navigation.

        If a session was focused in the dashboard (via Enter), that
        session is propagated to the Browse screen so pressing ``b``
        shows the same session's tree (VAL-CROSS-006).
        """
        if self.app.screen_stack:
            self.app.pop_screen()
            self.app._restore_call_tree_focus()
            # Propagate dashboard-focused session to browse
            if self._focused_session_index is not None:
                try:
                    from llm_flow_viewer.tui.screens.browse import BrowseScreen
                    browse = self.app.screen
                    if isinstance(browse, BrowseScreen):
                        browse.load_session_by_index(self._focused_session_index)
                except Exception:
                    pass
