"""Tests for the Help overlay screen ('?' key).

Covers the following validation assertions:
- VAL-DASH-012: Keyboard shortcut reference — help screen
"""

from __future__ import annotations

import pytest

from llm_flow_viewer.tui.app import LLMFlowViewerApp
from llm_flow_viewer.tui.screens.help_screen import HelpScreen
from llm_flow_viewer.tui.screens.browse import BrowseScreen


# ===================================================================
# VAL-DASH-012: Help screen via '?' key
# ===================================================================


class TestHelpKeyBinding:
    """The '?' key should be bound on the app for showing help."""

    @pytest.mark.asyncio
    async def test_question_mark_key_bound(self):
        """App should have '?' key bound for help."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        binding_keys = {b.key for b in app.BINDINGS}
        assert "?" in binding_keys, (
            "'?' key should be bound for showing the help overlay"
        )

    @pytest.mark.asyncio
    async def test_help_action_defined(self):
        """App should have action_show_help method."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        assert hasattr(app, "action_show_help"), (
            "App must have action_show_help method"
        )

    @pytest.mark.asyncio
    async def test_help_screen_registered(self):
        """App should have 'help' screen registered."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        assert "help" in app.SCREENS, (
            "HelpScreen should be registered in app.SCREENS"
        )

    @pytest.mark.asyncio
    async def test_help_screen_importable(self):
        """HelpScreen should be importable and instantiable."""
        screen = HelpScreen()
        assert screen is not None, (
            "HelpScreen should be instantiable"
        )

    @pytest.mark.asyncio
    async def test_help_screen_is_modal(self):
        """HelpScreen should be a ModalScreen."""
        from textual.screen import ModalScreen
        assert issubclass(HelpScreen, ModalScreen), (
            "HelpScreen should inherit from ModalScreen"
        )


class TestHelpScreenContent:
    """Help screen should contain keyboard shortcut listings."""

    @pytest.mark.asyncio
    async def test_help_overlay_appears_on_question_mark(self):
        """VAL-DASH-012: Pressing '?' should open the help overlay."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("?")
            await pilot.pause()

            # The top screen should now be HelpScreen
            assert isinstance(app.screen, HelpScreen), (
                f"Expected HelpScreen after pressing '?', got {type(app.screen).__name__}"
            )

    @pytest.mark.asyncio
    async def test_help_overlay_dismissed_by_escape(self):
        """VAL-DASH-012: Pressing Escape should dismiss the help overlay."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("?")
            await pilot.pause()
            assert isinstance(app.screen, HelpScreen), (
                "HelpScreen should be visible after pressing '?'"
            )

            await pilot.press("escape")
            await pilot.pause()

            # Should be back to BrowseScreen
            assert isinstance(app.screen, BrowseScreen), (
                f"Expected BrowseScreen after dismissing help, got {type(app.screen).__name__}"
            )

    @pytest.mark.asyncio
    async def test_help_overlay_dismissed_by_question_mark_again(self):
        """VAL-DASH-012: Pressing '?' again should dismiss the help overlay."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("?")
            await pilot.pause()
            assert isinstance(app.screen, HelpScreen), (
                "HelpScreen should be visible after pressing '?'"
            )

            await pilot.press("?")
            await pilot.pause()

            # Should be back to BrowseScreen
            assert isinstance(app.screen, BrowseScreen), (
                f"Expected BrowseScreen after dismissing help, got {type(app.screen).__name__}"
            )

    @pytest.mark.asyncio
    async def test_help_overlay_has_title(self):
        """Help screen should have a title."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("?")
            await pilot.pause()

            # The help screen should have a title label
            title_label = app.screen.query_one("#help-title")
            assert title_label is not None, (
                "Help screen should have a #help-title element"
            )
            title_text = str(title_label.content)
            assert "Keyboard" in title_text or "Shortcuts" in title_text, (
                f"Help title should contain 'Keyboard' or 'Shortcuts', got: {title_text}"
            )

    @pytest.mark.asyncio
    async def test_help_overlay_has_footer(self):
        """Help screen should have a footer with dismiss instructions."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("?")
            await pilot.pause()

            footer = app.screen.query_one("#help-footer")
            assert footer is not None, (
                "Help screen should have a #help-footer element"
            )
            footer_text = str(footer.content)
            assert "Escape" in footer_text, (
                f"Help footer should mention 'Escape' to close, got: {footer_text}"
            )

    @pytest.mark.asyncio
    async def test_help_overlay_has_global_shortcuts(self):
        """Help screen should list global shortcuts (b, d, ?, q, Escape)."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("?")
            await pilot.pause()

            help_container = app.screen.query_one("#help-container")
            all_text = []
            for node in help_container.walk_children(with_self=False):
                if hasattr(node, 'content'):
                    all_text.append(str(node.content))
            content_text = " ".join(all_text)

            # Check for global shortcut mentions
            assert any(kw in content_text for kw in ["Switch to Browse", "Switch to Dashboard", "Browse", "Dashboard"]), (
                f"Help should mention Browse/Dashboard switching shortcuts"
            )

    @pytest.mark.asyncio
    async def test_help_overlay_has_browse_shortcuts(self):
        """Help screen should list Browse view shortcuts."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("?")
            await pilot.pause()

            help_container = app.screen.query_one("#help-container")
            # Walk through ALL descendant widgets to gather text
            all_text = []
            for node in help_container.walk_children(with_self=False):
                if hasattr(node, 'content'):
                    all_text.append(str(node.content))
            content_text = " ".join(all_text)

            # Should mention tree navigation
            assert any(kw in content_text for kw in ["Cycle focus", "Navigate", "Expand", "Browse View"]), (
                "Help should mention browse view navigation shortcuts"
            )

    @pytest.mark.asyncio
    async def test_help_overlay_has_dashboard_shortcuts(self):
        """Help screen should list Dashboard view shortcuts."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("?")
            await pilot.pause()

            help_container = app.screen.query_one("#help-container")
            all_text = []
            for node in help_container.walk_children(with_self=False):
                if hasattr(node, 'content'):
                    all_text.append(str(node.content))
            content_text = " ".join(all_text)

            # Should mention dashboard-specific shortcuts
            assert any(kw in content_text for kw in ["Focus a session", "comparison", "Drill", "Dashboard"]), (
                "Help should mention dashboard view navigation shortcuts"
            )

    @pytest.mark.asyncio
    async def test_help_content_is_scrollable(self):
        """Help content should be scrollable for long content."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("?")
            await pilot.pause()

            help_content = app.screen.query_one("#help-content")
            # VerticalScroll should support scrolling
            assert hasattr(help_content, 'scroll_up'), (
                "Help content container should support scrolling"
            )

    @pytest.mark.asyncio
    async def test_help_binding_on_browse_screen(self):
        """Help shortcut should work in Browse view."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            # Start in Browse (default)
            assert isinstance(app.screen, BrowseScreen), (
                f"Expected BrowseScreen on startup, got {type(app.screen).__name__}"
            )

            await pilot.press("?")
            await pilot.pause()
            assert isinstance(app.screen, HelpScreen), (
                f"Expected HelpScreen after pressing '?' in Browse, got {type(app.screen).__name__}"
            )

    @pytest.mark.asyncio
    async def test_help_binding_on_dashboard_screen(self):
        """Help shortcut should work in Dashboard view."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            # Switch to Dashboard
            await pilot.press("d")
            await pilot.pause()
            from llm_flow_viewer.tui.screens.dashboard import DashboardScreen
            assert isinstance(app.screen, DashboardScreen), (
                f"Expected DashboardScreen, got {type(app.screen).__name__}"
            )

            await pilot.press("?")
            await pilot.pause()
            assert isinstance(app.screen, HelpScreen), (
                f"Expected HelpScreen after pressing '?' in Dashboard, got {type(app.screen).__name__}"
            )


class TestHelpFooterHints:
    """Footer should show the '?' help hint."""

    @pytest.mark.asyncio
    async def test_footer_mentions_help_in_browse(self):
        """AppFooter should mention '?' help key in Browse view."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        async with app.run_test(size=(120, 40)) as pilot:
            from llm_flow_viewer.tui.widgets.app_footer import AppFooter
            footer = app.screen.query_one(AppFooter)
            footer_text = str(footer.content)
            assert "?" in footer_text or "Help" in footer_text, (
                f"Footer should mention '?' help key, got: {footer_text}"
            )

    @pytest.mark.asyncio
    async def test_help_binding_show_in_app_bindings(self):
        """The '?' binding should have show=True so it appears in the footer."""
        app = LLMFlowViewerApp(flows_dir="./flows")
        help_binding = next((b for b in app.BINDINGS if b.key == "?"), None)
        assert help_binding is not None, (
            "'?' binding should exist"
        )
        assert help_binding.show is True, (
            "'?' binding should be visible (show=True)"
        )
