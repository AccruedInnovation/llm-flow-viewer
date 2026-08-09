"""Main TUI application for LLM Flow Viewer."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from textual.app import App
from textual.binding import Binding
from textual.reactive import var

from llm_flow_viewer.tui.screens.browse import BrowseScreen
from llm_flow_viewer.tui.screens.help_screen import HelpScreen

logger = logging.getLogger(__name__)


class LLMFlowViewerApp(App):
    """Textual TUI application for browsing LLM API call flows.

    The application provides a three-panel Browse view
    (session sidebar, call tree, detail panel) with keyboard-driven
    navigation and a Dashboard view for cross-session metrics.
    """

    TITLE = "LLM Flow Viewer"
    SUB_TITLE = "Browse LLM API Calls"

    CSS = """
    /* App-level theming using Textual CSS variables for consistent colors
       and sufficient contrast on dark backgrounds. These variables are
       inherited by all child screens and widgets. */

    Screen {
        background: $surface;
    }
    """

    SCREENS = {
        "browse": BrowseScreen,
        "help": HelpScreen,
    }

    # Track the currently selected session index across views.
    # Set by BrowseScreen when a session is loaded, read by DashboardScreen
    # to highlight the active session.  ``None`` means no session is loaded.
    selected_session_index: int | None = var(None)

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("b", "switch_to_browse", "Browse", show=False),
        Binding("d", "switch_to_dashboard", "Dashboard", show=False),
        Binding("tab", "focus_next", "Next Panel", show=False),
        Binding("?", "show_help", "Help", show=True),
    ]

    def __init__(self, flows_dir: str = "./flows", session: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self._flows_dir = os.path.abspath(flows_dir)
        self._session_arg = session

    @property
    def preselected_session(self) -> str | None:
        """The session name passed via ``--session``, or ``None``.

        This is used by the Browse screen to pre-select a session
        on startup when the user provides ``--session <name>``.
        """
        return self._session_arg

    def on_mount(self) -> None:
        """Set up the application after mounting."""
        self.push_screen("browse")

    def get_flows_dir(self, screen_name: str = "browse") -> str | None:
        """Return the flows directory for the given screen.

        This method is consulted by screens that need the flows dir
        but cannot receive it via the constructor.

        Args:
            screen_name: The name of the screen requesting the path.

        Returns:
            The absolute flows directory path, or ``None`` if unknown.
        """
        return self._flows_dir

    def action_show_help(self) -> None:
        """Show the keyboard shortcut help overlay.

        Pushes a :class:`HelpScreen` on top of the current screen so
        that the underlying screen's state is preserved.  The help
        overlay can be dismissed with Escape or ``?``.
        """
        try:
            help_screen = self.SCREENS.get("help")
            if help_screen:
                self.push_screen(help_screen())
        except Exception:
            pass

    def action_switch_to_browse(self) -> None:
        """Switch to the Browse view.

        If we are currently on a screen that is not Browse (e.g. Dashboard),
        pop back to the Browse screen and restore focus to the CallTree
        widget for seamless keyboard interaction.  If Browse is already
        showing, this is a no-op.
        """
        if not isinstance(self.screen, BrowseScreen):
            # Pop the Dashboard screen to reveal Browse beneath
            if self.screen_stack:
                self.pop_screen()
                # After popping the top screen (Dashboard), restore focus
                # to the CallTree widget on the revealed BrowseScreen so
                # keyboard navigation works immediately.
                self._restore_call_tree_focus()

    def _restore_call_tree_focus(self) -> None:
        """Restore focus to the CallTree widget on the current screen.

        Called after ``pop_screen()`` returns to Browse to ensure the
        CallTree (the primary interactive widget) receives focus.
        """
        try:
            call_tree = self.screen.query_one("#call-tree")
            if call_tree:
                call_tree.focus()
        except Exception:
            pass

    def action_switch_to_dashboard(self) -> None:
        """Switch to the Dashboard view, preserving Browse state.

        Pushes a new Dashboard screen on top of the Browse screen so
        that Browse state (tree expansion, selected node, detail panel)
        is preserved in the screen stack.  Pressing ``b`` or Escape pops
        the Dashboard to return to Browse with full state restored.
        """
        if not self.screen_stack:
            # Browse should be on the stack; if not, push it first
            self.push_screen("browse")

        try:
            from llm_flow_viewer.tui.screens.dashboard import DashboardScreen
        except ImportError:
            self.notify(
                "Dashboard view coming soon!",
                title="Dashboard",
                severity="information",
                timeout=3,
            )
            return

        dashboard = DashboardScreen(
            flows_dir=self._flows_dir,
            selected_session_index=self.selected_session_index,
        )
        self.push_screen(dashboard)

    @property
    def flows_dir(self) -> str:
        """The absolute path to the flows directory."""
        return self._flows_dir

    def on_unmount(self) -> None:
        """Safety net: cancel workers on all screens during app shutdown.

        Iterates through the screen stack to cancel any tracked
        background workers, ensuring the app exits promptly
        (VAL-GEN-001, VAL-GEN-002, VAL-GEN-005).

        Note: we cannot reference ``self.screen`` here because during
        ``_shutdown`` the screen stack may already be empty.
        """
        for screen in list(self.screen_stack):
            workers = getattr(screen, "_workers", None)
            if workers is not None:
                for worker in list(workers):
                    if not worker.is_finished and not worker.is_cancelled:
                        try:
                            worker.cancel()
                        except Exception:
                            logger.debug(
                                "Exception cancelling worker '%s' "
                                "during app shutdown",
                                worker.name,
                            )
                workers.clear()


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="llm-flow-viewer",
        description="LLM Flow Viewer — browse and analyze LLM API call flows.",
    )
    parser.add_argument(
        "--flows-dir",
        default="./flows",
        help="Directory containing flow files (default: ./flows)",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Load a specific session by name on startup (e.g. 01_flows-analyze_codebase)",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit",
    )
    return parser


def _discover_session_names(flows_dir: str) -> list[str]:
    """Discover available session names in a flows directory.

    Args:
        flows_dir: Absolute path to the flows directory.

    Returns:
        A sorted list of session file names (e.g.
        ``["01_flows-analyze_codebase", …]``).
    """
    try:
        from llm_flow_viewer.tui.widgets.session_list import discover_sessions

        sessions = discover_sessions(flows_dir)
        # Return the original file names (index_task_name format)
        return [
            f"{s.index:02d}_flows-{s.task_name}" for s in sessions
        ]
    except Exception:
        return []


def _validate_session(
    session_arg: str,
    flows_dir: str,
    errors: list[str],
) -> None:
    """Validate that *session_arg* names a known session in *flows_dir*.

    Appends an error message to *errors* if the session is not found.

    Args:
        session_arg: The ``--session`` argument value (e.g.
            ``"01_flows-analyze_codebase"``).
        flows_dir: Absolute path to the flows directory.
        errors: List of error messages to append to.
    """
    session_names = _discover_session_names(flows_dir)
    if session_arg not in session_names:
        avail = ", ".join(session_names) if session_names else "(none)"
        errors.append(
            f"Error: session '{session_arg}' not found in "
            f"'{flows_dir}'. Available sessions: {avail}"
        )


def main() -> None:
    """Entry point for the LLM Flow Viewer TUI application.

    Parses command-line arguments and launches the Textual app.
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    if getattr(args, "version", False):
        from llm_flow_viewer import __version__

        print(f"llm-flow-viewer {__version__}")
        sys.exit(0)

    # Collect all validation errors before reporting, so the user sees
    # every problem with their arguments in one invocation.
    errors: list[str] = []

    flows_dir = os.path.abspath(args.flows_dir)

    # Validate --flows-dir
    if not os.path.isdir(flows_dir):
        errors.append(f"Error: flows directory not found: {flows_dir}")

    # Validate --session (only if --flows-dir is valid, because session
    # discovery depends on the directory contents)
    session_arg = getattr(args, "session", None)
    if session_arg is not None and not errors:
        _validate_session(session_arg, flows_dir, errors)

    # Report all errors before exit
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        sys.exit(1)

    app = LLMFlowViewerApp(flows_dir=flows_dir, session=session_arg)
    app.run()


if __name__ == "__main__":
    main()
