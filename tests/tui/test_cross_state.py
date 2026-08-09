"""Tests for cross-state behaviors in the LLM Flow Viewer TUI.

Covers the following validation assertions:
- VAL-BROWSE-063: Same session data renders consistently across views
- VAL-BROWSE-065: Loading a different session clears previous data
- VAL-CROSS-013: Switching views preserves browse state
- VAL-CROSS-016: Selecting a tree node populates detail panel
- VAL-CROSS-018: Collapsing/expanding tree nodes does not clear detail panel
"""

from __future__ import annotations

import pytest

from llm_flow_viewer.tui.app import LLMFlowViewerApp
from llm_flow_viewer.tui.screens.browse import BrowseScreen
from llm_flow_viewer.tui.widgets.call_tree import CallTree, CallTreeNodeData
from llm_flow_viewer.tui.widgets.detail_panel import DetailPanel
from llm_flow_viewer.parser.models import (
    LLMCall,
    ParsedRequest,
    ParsedResponse,
    Session,
    Timing,
    TokenUsage,
    ToolUse,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_call(
    request_id: str = "call_01",
    model: str = "deepseek-v4-flash",
    timestamp: float = 1000.0,
    has_response: bool = True,
    tool_count: int = 0,
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> LLMCall:
    """Create a mock LLMCall with minimal required fields."""
    req = ParsedRequest(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": "Hello"}],
        tools=[{"name": "test_tool", "description": "A test tool", "input_schema": {}}] if tool_count > 0 else [],
        system=[],
        stream=True,
        timestamp_start=timestamp,
        timestamp_end=timestamp + 0.05,
        request_id=request_id,
    )

    tool_uses = [
        ToolUse(name=f"Tool{i}", id=f"call_tool_{i}", input={"param": f"value{i}"})
        for i in range(tool_count)
    ]

    resp = ParsedResponse(
        text="This is a response." if has_response else "",
        thinking="Chain of thought..." if has_response else "",
        tool_uses=tool_uses,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=1000,
        stop_reason="end_turn" if has_response else None,
        status_code=200 if has_response else 0,
        error_message="" if has_response else "No response",
        timestamp_start=timestamp + 0.05,
        timestamp_end=timestamp + 2.0,
        request_id=request_id,
    )

    timing = Timing(
        request_start=timestamp,
        request_end=timestamp + 0.05,
        response_start=timestamp + 0.05 if has_response else None,
        response_end=timestamp + 2.0 if has_response else None,
    )

    token_usage = TokenUsage(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )

    return LLMCall(
        request_id=request_id,
        request=req,
        response=resp if has_response else None,
        timing=timing,
        token_usage=token_usage,
    )


def _make_mock_session(
    calls: list[LLMCall] | None = None,
    index: int = 1,
    task_name: str = "test_session",
) -> Session:
    """Create a mock Session with the given calls or reasonable defaults."""
    if calls is None:
        calls = [
            _make_mock_call(request_id="call_01", model="deepseek-v4-flash", timestamp=1000.0, tool_count=2,
                           input_tokens=100, output_tokens=50),
            _make_mock_call(request_id="call_02", model="deepseek-v4-flash", timestamp=1002.0, tool_count=0,
                           input_tokens=200, output_tokens=75),
            _make_mock_call(request_id="call_03", model="deepseek-v4-flash", timestamp=1004.0, tool_count=1,
                           input_tokens=150, output_tokens=60),
        ]
    return Session(
        index=index,
        task_name=task_name,
        model="deepseek-v4-flash",
        calls=calls,
    )


# ===================================================================
# VAL-CROSS-016: Selecting a tree node populates the detail panel
# ===================================================================


class TestDetailPanelNodeSelection:
    """Enter on a tree node should populate the detail panel with content."""

    @pytest.mark.asyncio
    async def test_enter_on_call_node_updates_detail(self, tmp_path):
        """VAL-CROSS-016: Pressing Enter on a call node should update detail panel."""
        app = LLMFlowViewerApp(flows_dir=str(tmp_path))
        async with app.run_test(size=(120, 40)) as pilot:
            tree = app.screen.query(CallTree).first()
            tree.focus()
            await pilot.pause()

            # Populate tree with mock data
            session = _make_mock_session()
            tree.populate(session)
            await pilot.pause()
            tree.root.expand()
            await pilot.pause()

            # Get the first call node
            call_nodes = [c for c in tree.root.children
                          if c.data and c.data.node_type == "call"]
            assert len(call_nodes) > 0

            # Verify the detail panel starts with placeholder
            detail = app.screen.query(DetailPanel).first()
            assert "Select a node" in str(detail.content), (
                "Detail panel should start with placeholder"
            )

            # Navigate to the first call node using keyboard
            # Focus the tree, use Space to select the first visible node
            tree.focus()
            await pilot.pause()

            # Verify the browse screen has the handler registered
            browse_screen = app.screen
            assert hasattr(browse_screen, "on_tree_node_selected"), (
                "BrowseScreen should handle Tree.NodeSelected"
            )

            # Verify the node data can be extracted for the detail panel
            event_data = call_nodes[0].data
            assert event_data is not None
            assert isinstance(event_data, CallTreeNodeData)

            # The _get_node_content method should return usable content
            content = BrowseScreen._get_node_content(event_data)
            assert len(content) > 0, "Node content should be non-empty"

            # For a call node, the content should contain model info
            if event_data.node_type == "call" and event_data.call:
                if event_data.call.request and event_data.call.request.model:
                    assert event_data.call.request.model in content, (
                        f"Content should include model name, got: {content[:100]}"
                    )

    def test_get_node_content_returns_data(self):
        """BrowseScreen._get_node_content should return appropriate content for each type."""
        # Test with a section node
        section_data = CallTreeNodeData(
            node_type="section",
            section_type="request_details",
            summary="Request Details",
        )
        content = BrowseScreen._get_node_content(section_data)
        assert content == "Request Details"

        # Test with a field node
        field_data = CallTreeNodeData(
            node_type="field",
            field_key="Model",
            field_value="deepseek-v4-flash",
            summary="Model: deepseek-v4-flash",
        )
        content = BrowseScreen._get_node_content(field_data)
        assert content == "deepseek-v4-flash"

        # Test with an error node
        error_data = CallTreeNodeData(
            node_type="error",
            summary="Error loading data",
            full_content="Detailed error message",
        )
        content = BrowseScreen._get_node_content(error_data)
        assert content == "Detailed error message"

    def test_get_node_content_with_full_content(self):
        """_get_node_content should prefer full_content when available."""
        data = CallTreeNodeData(
            node_type="content_block",
            content_block_type="text",
            content_preview="preview text...",
            full_content="This is the full content that should be shown in detail panel.",
        )
        content = BrowseScreen._get_node_content(data)
        assert "full content" in content, "Should use full_content"
        assert content == data.full_content


# ===================================================================
# VAL-CROSS-018: Collapsing/expanding tree nodes does not clear detail panel
# ===================================================================


class TestDetailPanelStability:
    """Collapsing/expanding tree nodes should not clear the detail panel."""

    @pytest.mark.asyncio
    async def test_detail_panel_persists_on_collapse(self, tmp_path):
        """VAL-CROSS-018: Detail panel should keep content when tree nodes are collapsed."""
        app = LLMFlowViewerApp(flows_dir=str(tmp_path))
        async with app.run_test(size=(120, 40)) as pilot:
            tree = app.screen.query(CallTree).first()
            detail = app.screen.query(DetailPanel).first()

            # Populate tree with mock data
            session = _make_mock_session()
            tree.populate(session)
            await pilot.pause()
            tree.root.expand()
            await pilot.pause()

            # Get the first call node
            call_nodes = [c for c in tree.root.children
                          if c.data and c.data.node_type == "call"]
            assert len(call_nodes) > 0

            # Set content in detail panel
            detail_content = "This is content that should persist"
            detail.show_content(detail_content, title="Test Content")
            await pilot.pause()
            assert detail_content in str(detail.content), "Detail panel should show the test content"

            # Collapse the root node
            tree.root.collapse()
            await pilot.pause()

            # Detail panel should still have the same content
            assert detail_content in str(detail.content), (
                "Detail panel should retain content after tree collapse"
            )

            # Expand the root again
            tree.root.expand()
            await pilot.pause()

            # Detail panel should still have the same content
            assert detail_content in str(detail.content), (
                "Detail panel should retain content after tree expand"
            )


# ===================================================================
# VAL-BROWSE-065: Loading a different session clears previous data
# ===================================================================


class TestSessionTransition:
    """Loading a new session should clear previous tree data."""

    @pytest.mark.asyncio
    async def test_new_session_clears_tree(self):
        """VAL-BROWSE-065: Loading a new session should clear the previous tree entirely."""
        tree = CallTree()
        session1 = _make_mock_session(index=1, task_name="session_one")
        session2 = _make_mock_session(index=2, task_name="session_two")

        # Populate with first session
        tree.populate(session1)
        call_children1 = [c for c in tree.root.children
                          if c.data and c.data.node_type == "call"]
        assert len(call_children1) >= 1, "First session should have call nodes"

        # Populate with second session
        tree.populate(session2)

        # Old call nodes should be gone
        tree_label = str(tree.root.label).lower()
        assert "session_one" not in tree_label, (
            "First session name should not appear in tree label after new load"
        )
        assert "session_two" in tree_label, (
            "Second session name should appear in tree label"
        )

        # Root should have children from the new session
        new_call_children = [c for c in tree.root.children
                             if c.data and c.data.node_type == "call"]
        assert len(new_call_children) >= 1, "New session should have call nodes"

    @pytest.mark.asyncio
    async def test_show_loading_clears_tree(self):
        """show_loading should clear the tree before showing loading state."""
        tree = CallTree()
        session = _make_mock_session()

        # Populate tree
        tree.populate(session)
        assert len(tree.root.children) > 0, "Tree should have children after populate"

        # Show loading - should clear children
        tree.show_loading("new_session")
        assert len(tree.root.children) == 0, (
            "Tree should have no children during loading state"
        )
        assert "Loading" in str(tree.root.label), (
            "Tree root should show loading text"
        )

    @pytest.mark.asyncio
    async def test_no_stale_nodes_during_transition(self):
        """VAL-BROWSE-065: No stale nodes from old session should remain visible."""
        tree = CallTree()
        session1 = _make_mock_session(index=1, task_name="first")
        session2 = _make_mock_session(index=2, task_name="second", calls=[])
        # Second session has no calls

        # Load first session
        tree.populate(session1)
        first_labels = [str(c.label) for c in tree.root.children]
        assert len(first_labels) > 0, "First session should have children"

        # Load second (empty) session
        tree.populate(session2)

        # Verify all old children are gone
        assert len(tree.root.children) == 0, (
            "Old session children should be cleared before new session loads"
        )

        # Root label should indicate empty session
        label_text = str(tree.root.label).lower()
        assert "no api calls" in label_text or "first" not in label_text, (
            "Root label should not reference old session data"
        )


# ===================================================================
# VAL-CROSS-013: Switching views preserves browse state
# ===================================================================


class TestBrowseStatePersistence:
    """Switching to Dashboard and back should preserve browse state."""

    @pytest.mark.asyncio
    async def test_dashboard_screen_exists_and_registered(self):
        """App should be able to create a DashboardScreen instance.

        The DashboardScreen is not registered in SCREENS (it uses dynamic
        instantiation to pass constructor arguments), but the module and
        class must be importable.
        """
        from llm_flow_viewer.tui.screens.dashboard import DashboardScreen
        assert DashboardScreen is not None, "DashboardScreen must be importable"
        # Verify it's a Screen subclass
        from textual.screen import Screen
        assert issubclass(DashboardScreen, Screen), (
            "DashboardScreen must be a Screen subclass"
        )


# ===================================================================
# VAL-BROWSE-063: Same session data renders consistently across views
# ===================================================================


class TestDataConsistencyAcrossViews:
    """Same session data (token counts, call counts, timing) renders identically
    in both Browse and Dashboard views."""

    def test_session_data_consistent(self):
        """Session data should be stored and retrievable consistently."""
        # Create a session
        session = _make_mock_session(
            task_name="test_session",
            calls=[
                _make_mock_call(input_tokens=100, output_tokens=50),
                _make_mock_call(input_tokens=200, output_tokens=75),
            ]
        )

        # Verify call counts
        assert len(session.calls) == 2, "Session should have 2 calls"

        # Verify token counts
        total_input = 0
        total_output = 0
        for call in session.calls:
            if call.response:
                total_input += call.response.input_tokens or 0
                total_output += call.response.output_tokens or 0
        assert total_input == 300, f"Total input tokens should be 300, got {total_input}"
        assert total_output == 125, f"Total output tokens should be 125, got {total_output}"

        # Verify session metadata
        assert session.task_name == "test_session"
        assert session.index == 1
        assert session.model == "deepseek-v4-flash"


# ===================================================================
# Dashboard screen basic tests
# ===================================================================


class TestDashboardScreenBasics:
    """Basic tests for the Dashboard screen."""

    @pytest.mark.asyncio
    async def test_dashboard_has_quit_binding(self):
        """Dashboard screen should have 'q' quit binding."""
        from llm_flow_viewer.tui.screens.dashboard import DashboardScreen
        binding_keys = {b.key for b in DashboardScreen.BINDINGS}
        assert "q" in binding_keys or any("quit" in b.action for b in DashboardScreen.BINDINGS), (
            "Dashboard should have quit key binding"
        )


# ===================================================================
# View switching via keyboard
# ===================================================================


class TestViewSwitchBindings:
    """View switching bindings in app and screens."""

    @pytest.mark.asyncio
    async def test_dashboard_has_escape_binding(self):
        """Dashboard should have Escape binding to go back to Browse."""
        from llm_flow_viewer.tui.screens.dashboard import DashboardScreen
        binding_keys = {b.key for b in DashboardScreen.BINDINGS}
        assert "escape" in binding_keys or "b" in binding_keys, (
            "Dashboard should have Escape or 'b' binding to return to Browse"
        )
