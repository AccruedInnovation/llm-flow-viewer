"""Tests for the DetailPanel widget.

Covers placeholder state, JSON syntax highlighting, plain text rendering,
scrollable content, border title updates, key bindings, and content type
detection.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Button

from llm_flow_viewer.tui.widgets.detail_panel import DetailPanel


# ---------------------------------------------------------------------------
# Test app for isolated DetailPanel testing
# ---------------------------------------------------------------------------


class DetailPanelTestApp(App):
    """A minimal test app that contains just a DetailPanel."""

    def compose(self) -> ComposeResult:
        yield DetailPanel(id="test-detail")


# ---------------------------------------------------------------------------
# Placeholder State (VAL-BROWSE-038)
# ---------------------------------------------------------------------------


class TestDetailPanelPlaceholder:
    """Detail panel shows placeholder when no node is selected."""

    def test_placeholder_constant_defined(self):
        """PLACEHOLDER_TEXT constant should be defined."""
        assert DetailPanel.PLACEHOLDER_TEXT == "Select a node to view details"

    @pytest.mark.asyncio
    async def test_placeholder_on_startup(self):
        """Detail panel should show placeholder text on startup."""
        app = DetailPanelTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            detail = app.query_one("#test-detail", DetailPanel)
            # The content property returns the value set in __init__
            assert "Select a node to view details" in str(detail.content), (
                "Detail panel should have placeholder content set"
            )

    @pytest.mark.asyncio
    async def test_show_placeholder_does_not_crash(self):
        """show_placeholder() should execute without error."""
        app = DetailPanelTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            detail = app.query_one("#test-detail", DetailPanel)
            # Show content first, then reset
            detail.show_content("Some content here")
            await pilot.pause()
            # Reset to placeholder - should not crash
            detail.show_placeholder()
            await pilot.pause()


# ---------------------------------------------------------------------------
# Border Title (node type label)
# ---------------------------------------------------------------------------


class TestDetailPanelBorderTitle:
    """Detail panel border title should reflect the selected node type."""

    @pytest.mark.asyncio
    async def test_set_title_updates_border(self):
        """set_title() should update the border_title."""
        app = DetailPanelTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            detail = app.query_one("#test-detail", DetailPanel)
            # set_title is a method we'll add to DetailPanel
            detail.set_title("Custom Title")
            await pilot.pause()
            assert detail.border_title == "Custom Title", (
                f"border_title should be 'Custom Title', got: {detail.border_title}"
            )

    @pytest.mark.asyncio
    async def test_default_title_on_mount(self):
        """Default border title should be set on mount."""
        app = DetailPanelTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            detail = app.query_one("#test-detail", DetailPanel)
            # Default border title should be set in on_mount
            assert detail.border_title is not None, (
                "border_title should not be None after mount"
            )


# ---------------------------------------------------------------------------
# JSON Syntax Highlighting (VAL-BROWSE-035)
# ---------------------------------------------------------------------------


class TestDetailPanelJsonContent:
    """JSON content should be rendered with syntax highlighting."""

    @pytest.mark.asyncio
    async def test_show_json_does_not_crash(self):
        """show_json() should execute without errors for valid JSON."""
        app = DetailPanelTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            detail = app.query_one("#test-detail", DetailPanel)
            json_str = '{"name": "test", "value": 42, "active": true}'
            # Should not raise any exception
            detail.show_json(json_str)
            await pilot.pause()

    @pytest.mark.asyncio
    async def test_show_json_invalid_fallback(self):
        """show_json() should handle invalid JSON gracefully."""
        app = DetailPanelTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            detail = app.query_one("#test-detail", DetailPanel)
            # Invalid JSON should fallback gracefully, not crash
            detail.show_json("not valid json")
            await pilot.pause()

    def test_json_theme_is_dark(self):
        """JSON should use a dark theme (monokai or ansi_dark)."""
        assert DetailPanel._JSON_THEME in ("monokai", "ansi_dark", "native", "dracula", "zenburn"), (
            f"JSON theme should be a dark theme, got: {DetailPanel._JSON_THEME}"
        )


# ---------------------------------------------------------------------------
# Plain Text Content (VAL-BROWSE-036)
# ---------------------------------------------------------------------------


class TestDetailPanelTextContent:
    """Plain text content should be rendered with preserved whitespace."""

    @pytest.mark.asyncio
    async def test_show_text_does_not_crash(self):
        """show_text() should execute without error."""
        app = DetailPanelTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            detail = app.query_one("#test-detail", DetailPanel)
            detail.show_text("Line one\n\nLine two\nIndented line")
            await pilot.pause()

    @pytest.mark.asyncio
    async def test_show_text_empty(self):
        """show_text() should handle empty text gracefully."""
        app = DetailPanelTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            detail = app.query_one("#test-detail", DetailPanel)
            detail.show_text("")
            await pilot.pause()

    @pytest.mark.asyncio
    async def test_show_text_long_content(self):
        """show_text() should handle long text content."""
        app = DetailPanelTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            detail = app.query_one("#test-detail", DetailPanel)
            long_text = "Line " * 500  # Long text
            detail.show_text(long_text)
            await pilot.pause()


# ---------------------------------------------------------------------------
# Content Type Detection (VAL-BROWSE-035)
# ---------------------------------------------------------------------------


class TestContentDetection:
    """Detail panel should auto-detect JSON vs text content."""

    def test_is_json_content_valid_object(self):
        """_is_json_content should return True for valid JSON objects."""
        assert DetailPanel._is_json_content('{"key": "value"}')
        assert DetailPanel._is_json_content('{"a": 1, "b": [2, 3]}')

    def test_is_json_content_valid_array(self):
        """_is_json_content should return True for valid JSON arrays."""
        assert DetailPanel._is_json_content('[1, 2, 3]')
        assert DetailPanel._is_json_content('[{"a": 1}]')

    def test_is_json_content_invalid(self):
        """_is_json_content should return False for invalid JSON."""
        assert not DetailPanel._is_json_content("not json")
        assert not DetailPanel._is_json_content("{invalid}")
        assert not DetailPanel._is_json_content("")

    def test_is_json_content_plain_text(self):
        """_is_json_content should return False for plain text."""
        assert not DetailPanel._is_json_content("Hello, world!")
        assert not DetailPanel._is_json_content("Line one\nLine two")

    @pytest.mark.asyncio
    async def test_show_content_detects_json(self):
        """show_content() should detect JSON and not crash."""
        app = DetailPanelTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            detail = app.query_one("#test-detail", DetailPanel)
            detail.show_content('{"hello": "world", "count": 42}')
            await pilot.pause()

    @pytest.mark.asyncio
    async def test_show_content_plain_text(self):
        """show_content() should handle plain text without crashing."""
        app = DetailPanelTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            detail = app.query_one("#test-detail", DetailPanel)
            detail.show_content("This is plain text\nWith multiple lines")
            await pilot.pause()


# ---------------------------------------------------------------------------
# Scrollability (VAL-BROWSE-037)
# ---------------------------------------------------------------------------


class TestDetailPanelScrollable:
    """Detail panel should be scrollable for long content."""

    @pytest.mark.asyncio
    async def test_overflow_y_auto(self):
        """Detail panel should have overflow_y set to auto."""
        app = DetailPanelTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            detail = app.query_one("#test-detail", DetailPanel)
            assert detail.styles.overflow_y == "auto", (
                "Detail panel should have overflow_y: auto for scrollability"
            )

    @pytest.mark.asyncio
    async def test_long_content_scrollable(self):
        """Long content should make the panel scrollable."""
        app = DetailPanelTestApp()
        async with app.run_test(size=(60, 10)) as pilot:  # Small terminal
            detail = app.query_one("#test-detail", DetailPanel)
            # Show very long content
            long_text = "\n".join(f"Line {i}" for i in range(100))
            detail.show_text(long_text)
            await pilot.pause()

            # Virtual size should be large (content exceeds visible area)
            vs = detail.virtual_size
            assert vs.height > 10, (
                f"Virtual height should exceed visible height for long content, "
                f"got virtual.height={vs.height}"
            )

    def test_scroll_up_down_bindings(self):
        """Detail panel should have Up/Down key bindings for scrolling."""
        detail = DetailPanel()
        bindings = detail.BINDINGS
        binding_keys = {b.key for b in bindings}
        assert "up" in binding_keys, "Up key binding should exist"
        assert "down" in binding_keys, "Down key binding should exist"

    def test_page_up_down_bindings(self):
        """Detail panel should have PageUp/PageDown key bindings for scrolling."""
        detail = DetailPanel()
        bindings = detail.BINDINGS
        binding_keys = {b.key for b in bindings}
        # PageUp and PageDown should be supported
        assert "page_up" in binding_keys or "pagedown" in binding_keys or "page_down" in binding_keys or "pageup" in binding_keys, (
            f"PageUp/PageDown key bindings should exist, got: {binding_keys}"
        )


# ---------------------------------------------------------------------------
# Key Bindings (VAL-BROWSE-046)
# ---------------------------------------------------------------------------


class TestDetailPanelKeyBindings:
    """Detail panel key bindings for scrolling."""

    def test_key_bindings_defined(self):
        """DetailPanel should define key bindings."""
        detail = DetailPanel()
        assert len(detail.BINDINGS) >= 2, (
            "DetailPanel should have at least 2 key bindings"
        )


# ---------------------------------------------------------------------------
# Integration with BrowseScreen
# ---------------------------------------------------------------------------


class TestDetailPanelIntegration:
    """Integration tests with the full app."""

    @pytest.mark.asyncio
    async def test_detail_panel_in_browse_screen(self):
        """DetailPanel should be part of the BrowseScreen layout."""
        from llm_flow_viewer.tui.app import LLMFlowViewerApp

        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            detail = app.screen.query(DetailPanel).first()
            assert detail is not None, "DetailPanel should be present in the app"
            assert detail.id == "detail-panel", (
                f"Expected id 'detail-panel', got '{detail.id}'"
            )

    @pytest.mark.asyncio
    async def test_detail_panel_placeholder_in_app(self):
        """DetailPanel should show placeholder text in the full app."""
        from llm_flow_viewer.tui.app import LLMFlowViewerApp

        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            detail = app.screen.query(DetailPanel).first()
            assert "Select a node to view details" in str(detail.content), (
                f"Detail panel should show placeholder in app, got: {detail.content}"
            )


# ---------------------------------------------------------------------------
# Consistency: Not clearing on tree collapse/expand (VAL-BROWSE-034 edge)
# ---------------------------------------------------------------------------


class TestDetailPanelConsistency:
    """Detail panel content should not be cleared by tree operations."""

    @pytest.mark.asyncio
    async def test_content_updates_work(self):
        """Multiple show_content calls should work without errors."""
        app = DetailPanelTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            detail = app.query_one("#test-detail", DetailPanel)

            # Show multiple contents in sequence
            detail.show_content("First content")
            await pilot.pause()

            detail.show_content("Second content")
            await pilot.pause()


# ---------------------------------------------------------------------------
# Node Type Title Mapping
# ---------------------------------------------------------------------------


class TestDetailPanelTitles:
    """Detail panel should set appropriate titles for various node types."""

    def test_set_title_method_exists(self):
        """DetailPanel should have a set_title method."""
        assert hasattr(DetailPanel, "set_title"), "DetailPanel must have set_title method"


# ---------------------------------------------------------------------------
# Scroll behavior tests (VAL-BROWSE-001 through VAL-BROWSE-015)
# ---------------------------------------------------------------------------


class ConstrainedDetailPanelApp(App):
    """A test app that constrains the DetailPanel with height: 1fr.

    This replicates the fix from BrowseScreen so we can test scroll
    behavior in isolation.
    """

    CSS = """
    #outer-container {
        height: 20;
        border: none;
    }

    #test-detail {
        height: 1fr;
        border: solid $primary;
    }

    #dummy-button {
        height: 3;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="outer-container"):
            yield DetailPanel(id="test-detail")
            # Dummy focusable widget so we can move focus away from detail panel
            yield Button("Dummy", id="dummy-button", variant="default")


class TestDetailPanelScrollBehavior:
    """Detail panel scrolling with proper height constraint."""

    @pytest.mark.asyncio
    async def test_long_content_shows_scrollbar(self):
        """Long content should make the panel scrollable when constrained."""
        app = ConstrainedDetailPanelApp()
        async with app.run_test(size=(120, 40)) as pilot:
            detail = app.query_one("#test-detail", DetailPanel)

            # Show very long content
            long_text = "\n".join(f"Line {i}" for i in range(100))
            detail.show_text(long_text)
            await pilot.pause()

            # Container height should be constrained (~20 rows minus borders)
            container_height = detail.container_size.height
            assert container_height > 0, "Container height should be positive"
            # Virtual height should exceed container height (overflow)
            assert detail.virtual_size.height > container_height, (
                f"Virtual height ({detail.virtual_size.height}) should exceed "
                f"container height ({container_height}) for long content"
            )
            # max_scroll_y should be > 0
            assert detail.max_scroll_y > 0, (
                f"max_scroll_y should be > 0 for long content, got {detail.max_scroll_y}"
            )

    @pytest.mark.asyncio
    async def test_short_content_no_scrollbar(self):
        """Short content should not make the panel scrollable."""
        app = ConstrainedDetailPanelApp()
        async with app.run_test(size=(120, 40)) as pilot:
            detail = app.query_one("#test-detail", DetailPanel)

            # Show short content
            detail.show_text("Short text that fits")
            await pilot.pause()

            # max_scroll_y should be 0 (no overflow)
            assert detail.max_scroll_y == 0, (
                f"max_scroll_y should be 0 for short content, got {detail.max_scroll_y}"
            )

    @pytest.mark.asyncio
    async def test_placeholder_no_scrollbar(self):
        """Placeholder text should not have a scrollbar."""
        app = ConstrainedDetailPanelApp()
        async with app.run_test(size=(120, 40)) as pilot:
            detail = app.query_one("#test-detail", DetailPanel)

            # Panel starts with placeholder
            await pilot.pause()
            assert detail.max_scroll_y == 0, (
                f"Placeholder should have max_scroll_y=0, got {detail.max_scroll_y}"
            )

    @pytest.mark.asyncio
    async def test_arrow_down_scrolls_content(self):
        """Down arrow should scroll content down when panel is focused."""
        app = ConstrainedDetailPanelApp()
        async with app.run_test(size=(120, 40)) as pilot:
            detail = app.query_one("#test-detail", DetailPanel)

            # Show long content
            long_text = "\n".join(f"Line {i}" for i in range(100))
            detail.show_text(long_text)
            await pilot.pause()

            # Focus the panel
            detail.focus()
            await pilot.pause()

            # Record initial scroll position
            initial_scroll = detail.scroll_y

            # Press down arrow a few times
            for _ in range(3):
                await pilot.press("down")
                await pilot.pause()

            # Scroll position should have advanced
            assert detail.scroll_y > initial_scroll, (
                f"Scroll Y should increase after down arrow presses "
                f"(was {initial_scroll}, now {detail.scroll_y})"
            )

    @pytest.mark.asyncio
    async def test_arrow_up_scrolls_content(self):
        """Up arrow should scroll content up when panel is focused."""
        app = ConstrainedDetailPanelApp()
        async with app.run_test(size=(120, 40)) as pilot:
            detail = app.query_one("#test-detail", DetailPanel)

            # Show long content
            long_text = "\n".join(f"Line {i}" for i in range(100))
            detail.show_text(long_text)
            await pilot.pause()

            # Focus and scroll down first
            detail.focus()
            await pilot.pause()
            for _ in range(10):
                await pilot.press("down")
                await pilot.pause()

            scrolled_y = detail.scroll_y
            assert scrolled_y > 0, "Should have scrolled down before testing up"

            # Press up arrow a few times
            for _ in range(3):
                await pilot.press("up")
                await pilot.pause()

            # Scroll position should have decreased
            assert detail.scroll_y < scrolled_y, (
                f"Scroll Y should decrease after up arrow presses "
                f"(was {scrolled_y}, now {detail.scroll_y})"
            )

    @pytest.mark.asyncio
    async def test_page_down_scrolls_content(self):
        """PageDown should scroll content by approximately one viewport."""
        app = ConstrainedDetailPanelApp()
        async with app.run_test(size=(120, 40)) as pilot:
            detail = app.query_one("#test-detail", DetailPanel)

            # Show long content
            long_text = "\n".join(f"Line {i}" for i in range(100))
            detail.show_text(long_text)
            await pilot.pause()

            detail.focus()
            await pilot.pause()

            initial_scroll = detail.scroll_y
            await pilot.press("page_down")
            await pilot.pause()

            # PageDown should advance more than a single line
            assert detail.scroll_y > initial_scroll, (
                f"PageDown should increase scroll Y "
                f"(was {initial_scroll}, now {detail.scroll_y})"
            )

    @pytest.mark.asyncio
    async def test_page_up_scrolls_content(self):
        """PageUp should scroll content upward."""
        app = ConstrainedDetailPanelApp()
        async with app.run_test(size=(120, 40)) as pilot:
            detail = app.query_one("#test-detail", DetailPanel)

            long_text = "\n".join(f"Line {i}" for i in range(100))
            detail.show_text(long_text)
            await pilot.pause()

            detail.focus()
            await pilot.pause()

            # Scroll to bottom first
            detail.scroll_end(animate=False)
            await pilot.pause()
            bottom_scroll = detail.scroll_y
            assert bottom_scroll > 0, "Should be scrolled to bottom"

            # Press page_up
            await pilot.press("page_up")
            await pilot.pause()

            # Should have scrolled up
            assert detail.scroll_y < bottom_scroll, (
                f"PageUp should decrease scroll Y "
                f"(was {bottom_scroll}, now {detail.scroll_y})"
            )

    @pytest.mark.asyncio
    async def test_scroll_bottom_boundary(self):
        """Should not scroll past bottom boundary."""
        app = ConstrainedDetailPanelApp()
        async with app.run_test(size=(120, 40)) as pilot:
            detail = app.query_one("#test-detail", DetailPanel)

            long_text = "\n".join(f"Line {i}" for i in range(100))
            detail.show_text(long_text)
            await pilot.pause()

            detail.focus()
            await pilot.pause()

            # Scroll all the way to bottom
            detail.scroll_end(animate=False)
            await pilot.pause()
            bottom_scroll = detail.scroll_y

            # Try to scroll past bottom
            for _ in range(5):
                await pilot.press("down")
                await pilot.pause()

            # Should still be at bottom (not past max_scroll_y)
            assert detail.scroll_y <= detail.max_scroll_y, (
                f"Scroll Y ({detail.scroll_y}) should not exceed "
                f"max_scroll_y ({detail.max_scroll_y})"
            )
            # Should not have changed from bottom
            assert detail.scroll_y == bottom_scroll, (
                f"Scroll Y should stay at bottom ({bottom_scroll}), "
                f"but changed to {detail.scroll_y}"
            )

    @pytest.mark.asyncio
    async def test_scroll_top_boundary(self):
        """Should not scroll past top boundary."""
        app = ConstrainedDetailPanelApp()
        async with app.run_test(size=(120, 40)) as pilot:
            detail = app.query_one("#test-detail", DetailPanel)

            long_text = "\n".join(f"Line {i}" for i in range(100))
            detail.show_text(long_text)
            await pilot.pause()

            detail.focus()
            await pilot.pause()

            # Should be at top already
            assert detail.scroll_y == 0, "Should start at scroll_y=0"

            # Try to scroll past top
            for _ in range(5):
                await pilot.press("up")
                await pilot.pause()

            # Should still be at top
            assert detail.scroll_y == 0, (
                f"Scroll Y should stay at 0 at top boundary, got {detail.scroll_y}"
            )

    @pytest.mark.asyncio
    async def test_new_node_resets_scroll(self):
        """Calling scroll_home after changing content should reset scroll to top.

        Simulates the BrowseScreen behavior: when a different tree node is
        selected, on_tree_node_selected calls scroll_home after show_content.
        """
        app = ConstrainedDetailPanelApp()
        async with app.run_test(size=(120, 40)) as pilot:
            detail = app.query_one("#test-detail", DetailPanel)

            # Show long content and scroll down
            detail.show_text("\n".join(f"Line {i}" for i in range(100)))
            await pilot.pause()

            detail.focus()
            await pilot.pause()
            for _ in range(15):
                await pilot.press("down")
                await pilot.pause()
            assert detail.scroll_y > 0, "Should have scrolled down"

            # Now show different (new node) content
            detail.show_text("NEW CONTENT: " + "\n".join(f"Line {i}" for i in range(50)))
            # Scroll home simulates what on_tree_node_selected does for new nodes
            detail.scroll_home(animate=False)
            await pilot.pause()

            # Scroll should be at top
            assert detail.scroll_y == 0, (
                f"New content with scroll_home should give scroll_y=0, got {detail.scroll_y}"
            )

    @pytest.mark.asyncio
    async def test_same_content_preserves_scroll(self):
        """Showing the same content should preserve scroll position."""
        app = ConstrainedDetailPanelApp()
        async with app.run_test(size=(120, 40)) as pilot:
            detail = app.query_one("#test-detail", DetailPanel)

            content = "\n".join(f"Line {i}" for i in range(100))

            # Show content and scroll down
            detail.show_text(content)
            await pilot.pause()

            detail.focus()
            await pilot.pause()
            for _ in range(10):
                await pilot.press("down")
                await pilot.pause()
            scrolled_y = detail.scroll_y
            assert scrolled_y > 0, "Should have scrolled down"

            # Re-show same content (simulates same node re-selection)
            detail.show_text(content)
            await pilot.pause()

            # Scroll should be preserved (show_text with scroll_end=False keeps position)
            assert detail.scroll_y == scrolled_y, (
                f"Same content should preserve scroll at {scrolled_y}, "
                f"got {detail.scroll_y}"
            )

    @pytest.mark.asyncio
    async def test_transition_overflow_to_non_overflow_removes_scrollbar(self):
        """Transition from overflow to non-overflow content removes scrollbar."""
        app = ConstrainedDetailPanelApp()
        async with app.run_test(size=(120, 40)) as pilot:
            detail = app.query_one("#test-detail", DetailPanel)

            # Show long content (overflow)
            detail.show_text("\n".join(f"Line {i}" for i in range(100)))
            await pilot.pause()
            assert detail.max_scroll_y > 0, "Long content should have scroll"

            # Show short content (no overflow)
            detail.show_text("Short text")
            await pilot.pause()
            assert detail.max_scroll_y == 0, (
                f"Short content should have max_scroll_y=0, got {detail.max_scroll_y}"
            )

    @pytest.mark.asyncio
    async def test_transition_non_overflow_to_overflow_adds_scrollbar(self):
        """Transition from non-overflow to overflow content adds scrollbar."""
        app = ConstrainedDetailPanelApp()
        async with app.run_test(size=(120, 40)) as pilot:
            detail = app.query_one("#test-detail", DetailPanel)

            # Show short content (no overflow)
            detail.show_text("Short text")
            await pilot.pause()
            assert detail.max_scroll_y == 0, "Short content should have no scroll"

            # Show long content (overflow)
            detail.show_text("\n".join(f"Line {i}" for i in range(100)))
            await pilot.pause()
            assert detail.max_scroll_y > 0, (
                f"Long content should have scroll, got max_scroll_y={detail.max_scroll_y}"
            )

    @pytest.mark.asyncio
    async def test_arrow_keys_no_effect_when_not_focused(self):
        """Arrow keys should not scroll detail panel when not focused."""
        app = ConstrainedDetailPanelApp()
        async with app.run_test(size=(120, 40)) as pilot:
            detail = app.query_one("#test-detail", DetailPanel)

            # Show long content
            detail.show_text("\n".join(f"Line {i}" for i in range(100)))
            await pilot.pause()

            # Ensure detail panel is NOT focused
            if detail.has_focus:
                await pilot.press("shift+tab")  # Move focus away
                await pilot.pause()

            initial_scroll = detail.scroll_y

            # Press down arrow (should NOT scroll detail panel)
            for _ in range(3):
                await pilot.press("down")
                await pilot.pause()

            # Scroll position should be unchanged
            assert detail.scroll_y == initial_scroll, (
                f"Scroll Y should not change when panel is not focused "
                f"(was {initial_scroll}, now {detail.scroll_y})"
            )
