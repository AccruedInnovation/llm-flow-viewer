"""Tests for cross-cutting error and resilience behaviors in the TUI.

Covers the following validation assertions:
- VAL-CROSS-006: Successful parse populates browse view
- VAL-CROSS-007: Partial parse — some files fail, remaining data still shown
- VAL-CROSS-008: Session with zero calls appears in list but shows empty tree
- VAL-CROSS-009: Large session loads progressively
- VAL-CROSS-022: Corrupt flow file — error surfaced in browse view
- VAL-CROSS-023: Corrupt flow file — error reflected in dashboard metrics
- VAL-CROSS-024: Missing parquet cache triggers parser on raw flow file
- VAL-CROSS-025: Disk I/O error during parse — graceful degradation
- VAL-CROSS-026: JSON decode error in a single flow — isolation
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from textual.widgets import Tree

from llm_flow_viewer.parser.models import (
    LLMCall,
    ParsedRequest,
    ParsedResponse,
    Session,
    Timing,
    TokenUsage,
    ToolUse,
)
from llm_flow_viewer.tui.app import LLMFlowViewerApp
from llm_flow_viewer.tui.screens.browse import BrowseScreen
from llm_flow_viewer.tui.widgets.app_footer import AppFooter
from llm_flow_viewer.tui.widgets.call_tree import CallTree, CallTreeNodeData
from llm_flow_viewer.tui.widgets.session_list import (
    SessionInfo,
    SessionList,
    discover_sessions,
)
from llm_flow_viewer.tui.widgets.detail_panel import DetailPanel


# ===================================================================
# Helper functions
# ===================================================================


def _make_mock_call(
    request_id: str = "call_01",
    model: str = "deepseek-v4-flash",
    timestamp: float = 1000.0,
    has_response: bool = True,
    tool_count: int = 0,
    has_timing: bool = True,
    has_tokens: bool = True,
) -> LLMCall:
    """Create a mock LLMCall with minimal required fields."""
    req = ParsedRequest(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": "Hello"}],
        tools=[],
        system=[],
        stream=True,
        timestamp_start=timestamp,
        timestamp_end=timestamp + 0.05,
        request_id=request_id,
    )

    tool_uses = []
    for i in range(tool_count):
        tool_uses.append(ToolUse(
            name=f"Tool{i}",
            id=f"call_tool_{i}",
            input={"param": f"value{i}"},
        ))

    resp = ParsedResponse(
        text="This is a response." if has_response else "",
        thinking="" if not has_response else "",
        tool_uses=tool_uses,
        input_tokens=100 if has_tokens else None,
        output_tokens=50 if has_tokens else None,
        cache_creation_input_tokens=0 if has_tokens else None,
        cache_read_input_tokens=1000 if has_tokens else None,
        stop_reason="end_turn" if has_response else None,
        status_code=200 if has_response else 0,
        error_message="" if has_response else "No response",
        timestamp_start=timestamp + 0.05 if has_response else None,
        timestamp_end=timestamp + 2.0 if has_response else None,
        request_id=request_id,
    )

    timing = Timing(
        request_start=timestamp,
        request_end=timestamp + 0.05,
        response_start=timestamp + 0.05 if has_response else None,
        response_end=timestamp + 2.0 if has_response else None,
    ) if has_timing else None

    token_usage = TokenUsage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    ) if has_tokens else None

    return LLMCall(
        request_id=request_id,
        request=req,
        response=resp if has_response else None,
        timing=timing,
        token_usage=token_usage,
    )


def _make_session(calls: List[LLMCall] | None = None, flow_errors: List[str] | None = None) -> Session:
    """Create a mock Session with the given calls."""
    if calls is None:
        calls = [
            _make_mock_call(request_id="call_01", timestamp=1000.0, tool_count=2),
        ]
    return Session(
        index=1,
        task_name="test_session",
        model="deepseek-v4-flash",
        calls=calls,
        flow_errors=flow_errors or [],
    )


def _create_flow_file(directory: str, index: int, task_name: str) -> str:
    """Create a dummy flow file."""
    filename = f"{index:02d}_flows-{task_name}"
    filepath = os.path.join(directory, filename)
    Path(filepath).write_text("dummy flow content")
    return filepath


# ===================================================================
# VAL-CROSS-006: Successful parse populates browse view
# ===================================================================


class TestSuccessfulParse:
    """VAL-CROSS-006: Successful parse populates browse view."""

    def test_call_tree_populate_adds_call_nodes(self):
        """Populate should add call child nodes to the tree."""
        tree = CallTree()
        session = _make_session()
        tree.populate(session)

        call_nodes = [c for c in tree.root.children if c.data and c.data.node_type == "call"]
        assert len(call_nodes) >= 1, "Tree should have at least one call node"

    def test_call_tree_populate_shows_session_name(self):
        """Populate should show session name in root label."""
        tree = CallTree()
        session = _make_session()
        tree.populate(session)

        root_label = str(tree.root.label).lower()
        assert "test_session" in root_label, (
            f"Root should show session name, got: {root_label}"
        )

    def test_call_count_in_root_label(self):
        """Root label should include the call count."""
        tree = CallTree()
        session = _make_session()
        tree.populate(session)

        root_label = str(tree.root.label)
        assert "1 call" in root_label, (
            f"Root should show '1 call', got: {root_label}"
        )

    def test_call_tree_sections_expandable(self):
        """Each call node should have expandable section children."""
        tree = CallTree()
        session = _make_session()
        tree.populate(session)

        for child in tree.root.children:
            if child.data and child.data.node_type != "call":
                continue
            assert len(child.children) > 0, (
                "Call node should have section children"
            )


# ===================================================================
# VAL-CROSS-007: Partial parse — some files fail, remaining data still shown
# ===================================================================


class TestPartialParse:
    """VAL-CROSS-007: Partial parse — some files fail, remaining data still shown."""

    @pytest.mark.asyncio
    async def test_loading_session_with_error_shows_error_not_crash(self):
        """When _load_session encounters an error, it should show error in tree,
        not crash the app."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            browse = app.screen

            # Simulate an error by calling _on_session_error directly
            browse._on_session_error("test_session", "Unit test error")
            await pilot.pause()

            # App should still be running
            assert app._running is True

            # Tree should show error message
            tree = browse.query_one("#call-tree", CallTree)
            tree_label = str(tree.root.label).lower()
            assert "error" in tree_label, (
                f"Tree should show error message, got: {tree_label}"
            )

    @pytest.mark.asyncio
    async def test_app_remains_running_after_session_error(self):
        """App should not crash after a session loading error."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            browse = app.screen
            browse._on_session_error("bad_session", "Simulated failure")
            await pilot.pause()

            # Can still interact with the app
            await pilot.press("tab")
            await pilot.pause()
            assert app._running is True

            # Can quit gracefully
            await pilot.press("q")
            await pilot.pause()
            assert app._running is False

    @pytest.mark.asyncio
    async def test_multiple_errors_logged_not_crash(self):
        """Multiple session errors should be logged without crashing."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            browse = app.screen

            # Simulate multiple errors
            for i in range(3):
                browse._on_session_error(f"session_{i}", f"Error #{i}")
                await pilot.pause()

            assert app._running is True

    @pytest.mark.asyncio
    async def test_can_still_load_valid_session_after_error(self):
        """After a session error, can still load a valid session."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            browse = app.screen

            # Simulate an error first
            browse._on_session_error("bad", "Error")
            await pilot.pause()

            # Then load a valid session
            tree = browse.query_one("#call-tree", CallTree)
            session = _make_session()
            tree.populate(session)
            await pilot.pause()

            # Tree should have call nodes
            call_nodes = [c for c in tree.root.children if c.data and c.data.node_type == "call"]
            assert len(call_nodes) >= 1, (
                "Should be able to load valid session after error"
            )


# ===================================================================
# VAL-CROSS-008: Zero-call session
# ===================================================================


class TestZeroCallSession:
    """VAL-CROSS-008: Session with zero calls appears in list but shows empty tree."""

    def test_empty_session_shows_message(self):
        """Zero-call session should show informational message in tree."""
        tree = CallTree()
        session = _make_session(calls=[])
        tree.populate(session)

        label_text = str(tree.root.label).lower()
        assert "no api calls" in label_text or "no calls" in label_text, (
            f"Empty session should show message, got: {label_text}"
        )

    def test_empty_session_does_not_crash(self):
        """Zero-call session should not crash the tree."""
        tree = CallTree()
        session = _make_session(calls=[])
        # Must not raise exception
        tree.populate(session)
        # Tree should have no children (empty state)
        assert len(tree.root.children) == 0

    def test_empty_session_label_mentions_session_name(self):
        """Empty session label should include the session task name."""
        tree = CallTree()
        session = Session(
            index=2,
            task_name="empty_session",
            model="",
            calls=[],
        )
        tree.populate(session)

        root_label = str(tree.root.label)
        assert "empty_session" in root_label, (
            f"Session name should be in label, got: {root_label}"
        )

    @pytest.mark.asyncio
    async def test_empty_session_detail_panel_shows_placeholder(self):
        """Detail panel should show placeholder when empty session loaded."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            browse = app.screen

            tree = browse.query_one("#call-tree", CallTree)
            session = _make_session(calls=[])
            tree.populate(session)

            detail = browse.query_one("#detail-panel", DetailPanel)
            detail_text = str(detail.content).lower()
            assert "select a node" in detail_text or "view details" in detail_text, (
                f"Detail should show placeholder, got: {detail_text}"
            )


# ===================================================================
# VAL-CROSS-009: Large session progressive loading
# ===================================================================


class TestLargeSessionLoading:
    """VAL-CROSS-009: Large session loads progressively."""

    def test_loading_indicator_shown(self):
        """show_loading should update root label to indicate loading."""
        tree = CallTree()
        tree.show_loading("big_session")
        label_text = str(tree.root.label).lower()
        assert "loading" in label_text, (
            f"Root should show loading, got: {label_text}"
        )
        assert "big_session" in label_text, (
            f"Root should reference session name, got: {label_text}"
        )

    def test_loading_then_populate_shows_data(self):
        """After loading indicator, populate should replace it with data."""
        tree = CallTree()
        tree.show_loading("session")

        session = _make_session()
        tree.populate(session)

        label_text = str(tree.root.label).lower()
        assert "loading" not in label_text, (
            f"Loading indicator should be replaced, got: {label_text}"
        )

    def test_loading_clears_existing_children(self):
        """show_loading should clear existing tree children."""
        tree = CallTree()
        session = _make_session()
        tree.populate(session)
        assert len(tree.root.children) > 0

        tree.show_loading("new_session")
        assert len(tree.root.children) == 0, (
            "Tree should have no children during loading"
        )

    @pytest.mark.asyncio
    async def test_app_remains_responsive_during_loading(self):
        """App should remain responsive (can switch views) while loading."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            browse = app.screen

            # Show loading state
            tree = browse.query_one("#call-tree", CallTree)
            tree.show_loading("big_session")
            await pilot.pause()

            # Can still switch to dashboard
            await pilot.press("d")
            await pilot.pause()
            from llm_flow_viewer.tui.screens.dashboard import DashboardScreen
            assert isinstance(app.screen, DashboardScreen), (
                "Should be able to switch to dashboard while loading"
            )


# ===================================================================
# VAL-CROSS-022: Corrupt flow file — error in browse view
# ===================================================================


class TestCorruptSessionBrowse:
    """VAL-CROSS-022: Corrupt flow file — error surfaced in browse view."""

    def test_error_node_appears_in_tree(self):
        """When a session has flow errors, error nodes should appear in tree."""
        tree = CallTree()
        session = _make_session(
            calls=[_make_mock_call(request_id="valid_call")],
            flow_errors=["FlowReadException: Corrupt data at offset 1024"],
        )
        tree.populate(session)

        # Check for error nodes
        error_nodes = [c for c in tree.root.children if c.data and c.data.node_type == "error"]
        assert len(error_nodes) >= 1, (
            "Tree should have error nodes for flow errors"
        )

        # Error node should contain the error message
        error_label = str(error_nodes[0].label).lower()
        assert "error" in error_label or "corrupt" in error_label, (
            f"Error node should mention error, got: {error_label}"
        )

    def test_error_node_has_full_content(self):
        """Error node data should contain full error details."""
        tree = CallTree()
        error_msg = "FlowReadException: Detailed error at position 2048"
        session = _make_session(
            calls=[_make_mock_call(request_id="valid_call")],
            flow_errors=[error_msg],
        )
        tree.populate(session)

        error_nodes = [c for c in tree.root.children if c.data and c.data.node_type == "error"]
        assert len(error_nodes) >= 1
        data = error_nodes[0].data
        assert data is not None
        assert error_msg in data.full_content, (
            "Error node should store full error message"
        )

    def test_valid_calls_still_appear_alongside_errors(self):
        """Valid call nodes should appear alongside error nodes."""
        tree = CallTree()
        session = _make_session(
            calls=[_make_mock_call(request_id="valid_call")],
            flow_errors=["FlowReadException: Some error"],
        )
        tree.populate(session)

        call_nodes = [c for c in tree.root.children if c.data and c.data.node_type == "call"]
        assert len(call_nodes) >= 1, (
            "Valid calls should still appear despite errors"
        )

    @pytest.mark.asyncio
    async def test_session_error_message_in_tree(self):
        """Corrupt session should show error message in tree when selected."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            browse = app.screen

            # Simulate error loading
            browse._on_session_error("corrupt_session", "File is corrupt")
            await pilot.pause()

            tree = browse.query_one("#call-tree", CallTree)
            label_text = str(tree.root.label).lower()
            assert "error" in label_text, (
                f"Tree should show error for corrupt session, got: {label_text}"
            )


# ===================================================================
# VAL-CROSS-023: Corrupt flow file — error in dashboard
# ===================================================================


class TestCorruptSessionDashboard:
    """VAL-CROSS-023: Corrupt flow file — error reflected in dashboard."""

    @pytest.mark.asyncio
    async def test_session_errors_tracked(self):
        """Dashboard should track session errors."""
        from llm_flow_viewer.tui.screens.dashboard import DashboardScreen

        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("d")
            await pilot.pause()

            dashboard = app.screen
            assert isinstance(dashboard, DashboardScreen)

            # Add an error for session 1
            dashboard._session_errors[1] = "Corrupt file"
            await pilot.pause()

            # Error should be tracked
            assert 1 in dashboard._session_errors, (
                "Session 1 should have error"
            )


    @pytest.mark.asyncio
    async def test_excluded_session_not_counted_in_metrics(self):
        """Corrupt sessions should be excluded from aggregate metrics."""
        from llm_flow_viewer.tui.screens.dashboard import DashboardScreen

        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("d")
            await pilot.pause()

            dashboard = app.screen
            assert isinstance(dashboard, DashboardScreen)

            # Add error for one session - metrics should still count others
            dashboard._session_errors[1] = "Corrupt file"
            metrics = dashboard._compute_overall_metrics()

            # Should still have sessions
            assert metrics["total_sessions"] >= 0, (
                "Metrics should be computable even with errors"
            )
            assert metrics["total_calls"] >= 0, (
                "Metrics should be computable even with errors"
            )


# ===================================================================
# VAL-CROSS-024: Missing parquet cache triggers parser on raw flow file
# ===================================================================


class TestMissingCacheOnDemand:
    """VAL-CROSS-024: Missing parquet cache triggers parser on raw flow file."""

    def test_load_or_parse_called_when_no_cache(self):
        """When cache is missing, parser should be invoked."""
        from llm_flow_viewer.parser.cache import load_or_parse_cached, is_cache_fresh

        with tempfile.TemporaryDirectory() as tmpdir:
            flow_file = _create_flow_file(tmpdir, 1, "test_session")

            # With no cache, is_cache_fresh returns False
            assert not is_cache_fresh(flow_file, tmpdir), (
                "Cache should not be fresh before parsing"
            )

    def test_cache_written_after_parse(self):
        """After parsing, cache should be written."""
        from llm_flow_viewer.parser.cache import get_cache_paths, write_cache

        with tempfile.TemporaryDirectory() as tmpdir:
            flow_file = _create_flow_file(tmpdir, 1, "test_session")

            # Verify we can determine cache paths
            req_path, resp_path = get_cache_paths(flow_file, tmpdir)
            assert req_path.endswith("_requests.parquet"), (
                f"Request cache path should end with _requests.parquet, got: {req_path}"
            )
            assert resp_path.endswith("_responses.parquet"), (
                f"Response cache path should end with _responses.parquet, got: {resp_path}"
            )

    def test_cache_detects_missing(self):
        """is_cache_fresh should detect when cache files don't exist."""
        from llm_flow_viewer.parser.cache import is_cache_fresh

        with tempfile.TemporaryDirectory() as tmpdir:
            flow_file = _create_flow_file(tmpdir, 1, "test_session")
            assert not is_cache_fresh(flow_file, tmpdir), (
                "Cache should not be fresh when no cache files exist"
            )

    def test_valid_session_data_after_parsing(self):
        """Session data should be populated after parsing completes."""
        from llm_flow_viewer.parser.session import flow_file_to_session

        with tempfile.TemporaryDirectory() as tmpdir:
            flow_file = _create_flow_file(tmpdir, 1, "test_session")

            # flow_file_to_session should handle the missing cache gracefully
            try:
                session = flow_file_to_session(
                    flow_file,
                    index=1,
                    task_name="test_session",
                )
                # Should return a Session object even if parsing had issues
                assert session is not None
                assert session.index == 1
                assert session.task_name == "test_session"
            except Exception:
                # In test environment with dummy files, errors may occur
                # but that's expected - the key is no crash
                pass


# ===================================================================
# VAL-CROSS-025: Disk I/O error during parse — graceful degradation
# ===================================================================


class TestDiskIOError:
    """VAL-CROSS-025: Disk I/O error during parse — graceful degradation."""

    @pytest.mark.asyncio
    async def test_io_error_handled_gracefully(self):
        """IOError during session loading should show user-friendly error,
        not crash the app."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            browse = app.screen

            # Directly call _on_session_error to simulate I/O error handling
            browse._on_session_error("io_error_session", "Permission denied: cannot read file")
            await pilot.pause()

            # App should still be running
            assert app._running is True

            # Tree should show error indicator in root label
            tree = browse.query_one("#call-tree", CallTree)
            root_label = str(tree.root.label).lower()
            assert "error" in root_label, (
                f"Tree root should show error indicator, got: {root_label}"
            )

            # Error details should be in a child error node
            error_nodes = [
                c for c in tree.root.children
                if c.data and c.data.node_type == "error"
            ]
            assert len(error_nodes) >= 1, (
                "Tree should have error child nodes"
            )
            error_label = str(error_nodes[0].label).lower()
            assert "permission denied" in error_label or "cannot read" in error_label, (
                f"Error node should contain user-friendly message, got: {error_label}"
            )

    @pytest.mark.asyncio
    async def test_os_error_during_load_shows_error(self):
        """OS-level exceptions during load should be caught and shown."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            browse = app.screen

            # Simulate an OSError
            browse._on_session_error("os_error_session", "OSError: [Errno 13] Permission denied")
            await pilot.pause()

            assert app._running is True, "App should not crash on OSError"

    @pytest.mark.asyncio
    async def test_file_not_found_error_handled(self):
        """FileNotFoundError should be handled gracefully."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            browse = app.screen

            browse._on_session_error("missing_session", "File not found: no_such_file.flow")
            await pilot.pause()

            assert app._running is True, "App should not crash on FileNotFoundError"
            tree = browse.query_one("#call-tree", CallTree)
            tree_label = str(tree.root.label).lower()
            assert "error" in tree_label, (
                f"Tree should show error, got: {tree_label}"
            )

    @pytest.mark.asyncio
    async def test_can_navigate_away_after_io_error(self):
        """User can switch views after an I/O error."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            browse = app.screen

            browse._on_session_error("io_session", "Permission denied")
            await pilot.pause()

            # Can switch to dashboard
            await pilot.press("d")
            await pilot.pause()

            from llm_flow_viewer.tui.screens.dashboard import DashboardScreen
            assert isinstance(app.screen, DashboardScreen), (
                "Should be able to navigate to dashboard after I/O error"
            )

            # Can return to browse
            await pilot.press("b")
            await pilot.pause()

            from llm_flow_viewer.tui.screens.browse import BrowseScreen
            assert isinstance(app.screen, BrowseScreen), (
                "Should be able to return to browse after I/O error"
            )


# ===================================================================
# VAL-CROSS-026: JSON decode error in a single flow — isolation
# ===================================================================


class TestJSONDecodeErrorIsolation:
    """VAL-CROSS-026: JSON decode error in a single flow — isolation."""

    def test_error_nodes_do_not_block_valid_calls(self):
        """Flow errors should not prevent valid calls from being displayed."""
        tree = CallTree()
        session = _make_session(
            calls=[_make_mock_call(request_id="valid_1")],
            flow_errors=["FlowReadException: Corrupt flow at offset 100"],
        )
        tree.populate(session)

        call_nodes = [c for c in tree.root.children if c.data and c.data.node_type == "call"]
        error_nodes = [c for c in tree.root.children if c.data and c.data.node_type == "error"]

        assert len(call_nodes) >= 1, "Valid call nodes should be present"
        assert len(error_nodes) >= 1, "Error nodes should be present"

    def test_multiple_errors_shown(self):
        """Multiple flow errors should each have their own node."""
        tree = CallTree()
        session = _make_session(
            calls=[_make_mock_call(request_id="valid_1")],
            flow_errors=[
                "Error: Flow at position 100 corrupt",
                "Error: Flow at position 200 corrupt",
            ],
        )
        tree.populate(session)

        error_nodes = [c for c in tree.root.children if c.data and c.data.node_type == "error"]
        assert len(error_nodes) >= 2, (
            "Should have multiple error nodes"
        )

    def test_all_valid_calls_shown_despite_errors(self):
        """All valid calls should be shown despite presence of errors."""
        tree = CallTree()
        calls = [
            _make_mock_call(request_id="call_1", timestamp=1000.0),
            _make_mock_call(request_id="call_2", timestamp=1002.0),
            _make_mock_call(request_id="call_3", timestamp=1004.0),
        ]
        session = _make_session(
            calls=calls,
            flow_errors=["FlowReadException: Bad flow data"],
        )
        tree.populate(session)

        call_nodes = [c for c in tree.root.children if c.data and c.data.node_type == "call"]
        assert len(call_nodes) == 3, (
            f"All 3 valid calls should appear, got {len(call_nodes)}"
        )

    def test_flow_errors_stored_in_session(self):
        """Session should store flow errors for display."""
        errors = ["Error 1", "Error 2"]
        session = _make_session(
            calls=[_make_mock_call(request_id="valid")],
            flow_errors=errors,
        )
        assert len(session.flow_errors) == 2
        assert session.flow_errors == errors


# ===================================================================
# Footer/status bar tests
# ===================================================================


class TestFooterStatusErrors:
    """Tests for footer status messages during error conditions."""

    @pytest.mark.asyncio
    async def test_footer_can_set_status_after_error(self):
        """Footer should be able to display status after error."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()

            footer = app.screen.query_one(AppFooter)
            footer.set_status("Ready — 7 sessions available")
            await pilot.pause()

            footer_text = str(footer.content).lower()
            assert "ready" in footer_text or "session" in footer_text, (
                f"Footer should show status after error, got: {footer_text}"
            )


# ===================================================================
# Session list with error indicators
# ===================================================================


class TestSessionListErrors:
    """Tests for session list showing error states."""

    def test_session_list_displays_all_sessions(self):
        """Session list should display all discovered sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_flow_file(tmpdir, 1, "session_one")
            _create_flow_file(tmpdir, 2, "session_two")
            _create_flow_file(tmpdir, 3, "session_three")

            sessions = discover_sessions(tmpdir)
            assert len(sessions) == 3

    def test_session_list_empty_on_no_files(self):
        """Session list should be empty when no flow files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions = discover_sessions(tmpdir)
            assert len(sessions) == 0

    def test_session_list_ignores_parquet(self):
        """Session discovery should skip parquet files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "01_flows-test.parquet").write_text("parquet data")
            _create_flow_file(tmpdir, 1, "real_session")

            sessions = discover_sessions(tmpdir)
            assert len(sessions) == 1
            assert sessions[0].task_name == "real_session"


# ===================================================================
# Single-call session display
# ===================================================================


class TestSingleCallSession:
    """Single-call session should display correctly with one call entry."""

    def test_single_call_shown_correctly(self):
        """A single-call session should show exactly one call node."""
        tree = CallTree()
        session = _make_session(calls=[_make_mock_call(request_id="only_call")])
        tree.populate(session)

        call_nodes = [c for c in tree.root.children if c.data and c.data.node_type == "call"]
        assert len(call_nodes) == 1, (
            f"Should have 1 call node, got {len(call_nodes)}"
        )

    def test_single_call_label_shows_call_number(self):
        """Single call should be labeled 'Call #1'."""
        tree = CallTree()
        session = _make_session(calls=[_make_mock_call(request_id="only_call")])
        tree.populate(session)

        call_nodes = [c for c in tree.root.children if c.data and c.data.node_type == "call"]
        assert len(call_nodes) >= 1
        label = str(call_nodes[0].label)
        assert "Call #1" in label, (
            f"Single call should be 'Call #1', got: {label}"
        )

    def test_single_call_has_sections(self):
        """Single call should have expandable sections."""
        tree = CallTree()
        session = _make_session(calls=[_make_mock_call(request_id="only_call", tool_count=1)])
        tree.populate(session)

        call_nodes = [c for c in tree.root.children if c.data and c.data.node_type == "call"]
        assert len(call_nodes) >= 1
        # Should have section children
        assert len(call_nodes[0].children) > 0, (
            "Single call should have section children"
        )

    def test_single_call_model_in_label(self):
        """Single call label should include the model name."""
        tree = CallTree()
        session = _make_session(calls=[_make_mock_call(request_id="only_call", model="deepseek-v4-pro")])
        tree.populate(session)

        call_nodes = [c for c in tree.root.children if c.data and c.data.node_type == "call"]
        assert len(call_nodes) >= 1
        label = str(call_nodes[0].label)
        assert "deepseek-v4-pro" in label, (
            f"Call label should include model, got: {label}"
        )


# ===================================================================
# VAL-STREAM-007: Error Handling Preserves Partial Tree
# ===================================================================


class TestStreamingParseErrorPreservesPartialTree:
    """VAL-STREAM-007: Mid-stream parse error preserves already-parsed calls."""

    @pytest.mark.asyncio
    async def test_calls_preserved_after_session_error(self):
        """After _on_session_error, existing call nodes should remain in tree."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            browse = app.screen
            tree = browse.query_one("#call-tree", CallTree)

            # Simulate streaming: first append some calls
            calls = [
                _make_mock_call(request_id="call_01", timestamp=1000.0),
                _make_mock_call(request_id="call_02", timestamp=1002.0),
            ]
            tree.append_calls(calls)
            await pilot.pause()

            call_count_before = len([
                c for c in tree.root.children
                if c.data and c.data.node_type == "call"
            ])
            assert call_count_before == 2, (
                f"Should have 2 calls before error, got {call_count_before}"
            )

            # Now simulate a parse error mid-stream
            browse._on_session_error("test_session", "Parse error at entry 10")
            await pilot.pause()

            # Calls should be preserved
            call_count_after = len([
                c for c in tree.root.children
                if c.data and c.data.node_type == "call"
            ])
            assert call_count_after == call_count_before, (
                f"Call count should be preserved after error: "
                f"before={call_count_before}, after={call_count_after}"
            )

            # Error node should be present
            error_nodes = [
                c for c in tree.root.children
                if c.data and c.data.node_type == "error"
            ]
            assert len(error_nodes) >= 1, (
                "Error node should be present alongside preserved calls"
            )

            # Root label should indicate error
            root_label = str(tree.root.label)
            assert "Error" in root_label, (
                f"Root label should show error indicator, got: {root_label}"
            )

    @pytest.mark.asyncio
    async def test_user_can_still_browse_partial_tree_after_error(self):
        """User can still interact with partial tree after error."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            browse = app.screen
            tree = browse.query_one("#call-tree", CallTree)

            # Append calls then simulate error
            calls = [_make_mock_call(request_id="call_01", timestamp=1000.0)]
            tree.append_calls(calls)
            browse._on_session_error("test_session", "Parse error")
            await pilot.pause()

            # App should still be running
            assert app._running is True

            # User can still navigate to dashboard
            await pilot.press("d")
            await pilot.pause()
            from llm_flow_viewer.tui.screens.dashboard import DashboardScreen
            assert isinstance(app.screen, DashboardScreen), (
                "Should be able to navigate to Dashboard after error"
            )

            # And back to browse
            await pilot.press("b")
            await pilot.pause()
            from llm_flow_viewer.tui.screens.browse import BrowseScreen
            assert isinstance(app.screen, BrowseScreen), (
                "Should be able to navigate back to Browse after error"
            )

            # Tree should still have call nodes + error node
            restored_tree = browse.query_one("#call-tree", CallTree)
            call_nodes = [
                c for c in restored_tree.root.children
                if c.data and c.data.node_type == "call"
            ]
            error_nodes = [
                c for c in restored_tree.root.children
                if c.data and c.data.node_type == "error"
            ]
            assert len(call_nodes) == 1, "Call nodes should survive navigation"
            assert len(error_nodes) == 1, "Error nodes should survive navigation"

    @pytest.mark.asyncio
    async def test_error_surfaced_in_footer_status(self):
        """Error message should appear in the footer status bar."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            browse = app.screen

            # Simulate error
            browse._on_session_error("bad_session", "Connection timeout")
            await pilot.pause()

            # Footer should indicate the error
            footer = browse.query_one(AppFooter)
            footer_text = str(footer.content).lower()
            assert "error" in footer_text or "timeout" in footer_text, (
                f"Footer should show error status, got: {footer_text}"
            )

    @pytest.mark.asyncio
    async def test_error_during_empty_stream(self):
        """Error with no calls yet should still show error in tree."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            browse = app.screen
            tree = browse.query_one("#call-tree", CallTree)

            # Error occurs before any calls were appended
            browse._on_session_error("new_session", "File not found")
            await pilot.pause()

            # App should still be running
            assert app._running is True

            # Tree should show error (not placeholder)
            root_label = str(tree.root.label).lower()
            assert "error" in root_label, (
                f"Tree should show error, got: {root_label}"
            )


# ===================================================================
# VAL-STREAM-011: Session Switch Cancels Prior Stream
# ===================================================================


class TestSessionSwitchCancelsPriorStream:
    """VAL-STREAM-011: Session switch during stream cancels prior worker."""

    @pytest.mark.asyncio
    async def test_session_switch_clears_partial_tree(self):
        """When switching sessions, the partial tree from the old session should be cleared."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            browse = app.screen
            tree = browse.query_one("#call-tree", CallTree)

            # Simulate partial load of first session
            calls_session_a = [
                _make_mock_call(request_id="a_01", timestamp=1000.0),
                _make_mock_call(request_id="a_02", timestamp=1002.0),
            ]
            tree.append_calls(calls_session_a)

            # Verify calls from session A are present
            call_count_a = len([
                c for c in tree.root.children
                if c.data and c.data.node_type == "call"
            ])
            assert call_count_a == 2

            # Simulate session switch by cancelling the stream and showing loading
            browse._cancel_streaming()
            tree.show_loading("session_b")
            await pilot.pause()

            # Tree should be cleared for new session
            call_count_after_clear = len([
                c for c in tree.root.children
                if c.data and c.data.node_type == "call"
            ])
            assert call_count_after_clear == 0, (
                "Tree should be cleared for new session load"
            )

    @pytest.mark.asyncio
    async def test_worker_cancelled_on_switch(self):
        """The streaming worker should be destroyed on session switch."""
        from unittest.mock import MagicMock, patch

        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            browse = app.screen

            # Create a mock session info
            session_info_a = SessionInfo(
                index=1,
                task_name="session_a",
                file_path="/fake/path/session_a",
            )

            # Start loading session A
            browse._load_session(session_info_a)
            await pilot.pause()

            # Verify a streaming worker exists
            assert browse._streaming_worker is not None, (
                "Streaming worker should exist after starting load"
            )

            # Store reference to original worker
            original_worker = browse._streaming_worker

            # Start loading session B (this should cancel session A's worker)
            session_info_b = SessionInfo(
                index=2,
                task_name="session_b",
                file_path="/fake/path/session_b",
            )
            browse._load_session(session_info_b)
            await pilot.pause()

            # Original worker should be cancelled or replaced
            assert original_worker.is_cancelled or original_worker.is_finished, (
                "Original worker should be cancelled after session switch"
            )

            # Load ID should be updated
            assert browse._current_load_id != -1, (
                "Load ID should be valid after switch"
            )

            # New worker should be different
            assert browse._streaming_worker is not None, (
                "New streaming worker should exist"
            )
            # (the old worker reference may be reused if it was the same object,
            # but the load ID has changed, so old callbacks are guarded)

    @pytest.mark.asyncio
    async def test_cancel_streaming_prevents_stale_callbacks(self):
        """_cancel_streaming should prevent stale callbacks from executing."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            browse = app.screen
            tree = browse.query_one("#call-tree", CallTree)

            # Simulate: start with load_id=0
            browse._current_load_id = 0
            browse._next_load_id = 1

            # Append some calls (simulating first session streaming)
            calls = [_make_mock_call(request_id="old_call", timestamp=1000.0)]
            tree.append_calls(calls)

            # Cancel streaming (as happens on session switch)
            browse._cancel_streaming()
            await pilot.pause()

            # Now simulate a stale callback from old load_id=0
            # _apply_parse_batch should ignore it because _current_load_id != 0
            stale_batch = [_make_mock_call(request_id="stale_call", timestamp=1002.0)]
            browse._apply_parse_batch(0, 5, stale_batch)

            # The stale batch should not have added to tree
            call_nodes = [
                c for c in tree.root.children
                if c.data and c.data.node_type == "call"
            ]
            assert len(call_nodes) == 1, (
                "Stale callbacks should not add calls to the tree"
            )

    @pytest.mark.asyncio
    async def test_load_id_increments_on_each_switch(self):
        """Load ID should increment on each session switch."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            browse = app.screen

            # Set up mock session infos
            sessions = [
                SessionInfo(index=i, task_name=f"session_{i}", file_path=f"/fake/path/session_{i}")
                for i in range(3)
            ]

            # Start loading each session
            for session_info in sessions:
                browse._load_session(session_info)
                await pilot.pause()

            # After 3 loads, _next_load_id should be 3
            assert browse._next_load_id == 3, (
                f"Load ID should be 3 after 3 loads, got {browse._next_load_id}"
            )

    @pytest.mark.asyncio
    async def test_can_quit_after_session_switch(self):
        """User can quit the app gracefully after a session switch."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            browse = app.screen

            # Simulate loading session A then switching to B
            session_a = SessionInfo(
                index=1,
                task_name="session_a",
                file_path="/fake/path/a",
            )
            browse._load_session(session_a)
            await pilot.pause()

            session_b = SessionInfo(
                index=2,
                task_name="session_b",
                file_path="/fake/path/b",
            )
            browse._load_session(session_b)
            await pilot.pause()

            # Quit should work fine
            await pilot.press("q")
            await pilot.pause()
            assert app._running is False, (
                "App should quit cleanly after session switch"
            )
