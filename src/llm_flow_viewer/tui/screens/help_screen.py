"""Help screen showing keyboard shortcuts and navigation guide.

Displays a full-screen overlay listing all available keyboard shortcuts
for both Browse and Dashboard views.  Accessible via the ``?`` key from
any screen.  Dismissed via Escape or pressing ``?`` again.

Help content is displayed in a scrollable vertical layout, using Rich
Table renderables for clean tabular formatting of key bindings.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Label, Static
from rich.table import Table
from rich.text import Text
from rich import box
from rich.style import Style


_STYLE_HEADER = Style(bold=True, color="#00aa00")
_STYLE_KEY = Style(bold=True, color="#ffcc00")
_STYLE_SECTION = Style(bold=True, color="#0088ff")


def _key(key_str: str) -> Text:
    """Return a styled key representation."""
    return Text(key_str, style=_STYLE_KEY)


def _make_help_table(title: str, bindings: list[tuple[str, str]]) -> Table:
    """Build a Rich Table for a section of keyboard shortcuts.

    Args:
        title: The section title (e.g. "Navigation").
        bindings: List of ``(key, action)`` tuples.

    Returns:
        A Rich Table renderable suitable for display in a Static widget.
    """
    table = Table(
        show_header=True,
        header_style=_STYLE_HEADER,
        box=box.SIMPLE,
        padding=(0, 2),
        expand=True,
    )
    table.add_column("Key", justify="left", no_wrap=True, ratio=1)
    table.add_column("Action", justify="left", ratio=3)

    for key_str, action in bindings:
        table.add_row(_key(key_str), action)

    return table


# ---------------------------------------------------------------------------
# Browse view keyboard shortcuts
# ---------------------------------------------------------------------------

_BROWSE_BINDINGS = [
    ("Tab / Shift+Tab", "Cycle focus: Sidebar → Tree → Detail Panel"),
    ("↑ / ↓", "Navigate up/down in focused panel"),
    ("Enter", "Select a node (updates detail panel)"),
    ("Space", "Expand / collapse a tree node"),
    ("Shift+Space", "Expand / collapse all nodes recursively"),
    ("→ / ←", "Expand / collapse tree node (alternative)"),
    ("PageUp / PageDown", "Scroll detail panel page-wise"),
]

# ---------------------------------------------------------------------------
# Dashboard view keyboard shortcuts
# ---------------------------------------------------------------------------

_DASHBOARD_BINDINGS = [
    ("Tab / Shift+Tab", "Cycle focus between dashboard widgets"),
    ("↑ / ↓", "Navigate rows in DataTable"),
    ("Enter", "Focus a session (narrow charts to that session)"),
    ("Backspace", "Show all sessions (clear session focus)"),
    ("1-7", "Focus session 01-07 directly"),
    ("0", "Show all sessions"),
    ("c", "Toggle comparison mode"),
    ("Enter (on bar)", "Drill into a session's detailed view"),
    ("Escape", "Exit drill-down / comparison / back to Browse"),
]

# ---------------------------------------------------------------------------
# Global keyboard shortcuts
# ---------------------------------------------------------------------------

_GLOBAL_BINDINGS = [
    ("b", "Switch to Browse view"),
    ("d", "Switch to Dashboard view"),
    ("?", "Show this help overlay"),
    ("Escape", "Dismiss help overlay / go back"),
    ("q", "Quit the application"),
]

# ---------------------------------------------------------------------------
# Helpful hints
# ---------------------------------------------------------------------------

_HELP_HINTS = [
    "💡 Tip: The footer at the bottom shows context-sensitive shortcuts for the currently focused widget.",
    "💡 Tip: Browse view shows a three-panel layout: Sessions (left) | Call Tree (center) | Details (right).",
    "💡 Tip: Dashboard shows aggregate metrics across all sessions. Focus a session to see its details.",
    "💡 Tip: Comparison mode (press 'c' in Dashboard) lets you compare 2+ sessions side-by-side.",
]


class HelpScreen(ModalScreen):
    """Modal overlay showing all keyboard shortcuts and navigation guide.

    Accessible from any screen by pressing ``?``.
    Dismissed by pressing Escape or ``?`` again.

    The overlay is rendered as a full-screen scrollable modal with
    sections for Global, Browse, and Dashboard shortcuts, plus
    helpful usage hints.
    """

    BINDINGS = [
        Binding("escape", "dismiss_help", "Close", show=True),
        Binding("?", "dismiss_help", "Close", show=True),
    ]

    CSS = """
    HelpScreen {
        align: center middle;
        background: rgba(0, 0, 0, 0.85);
    }

    #help-container {
        width: 80%;
        max-width: 80;
        height: 80%;
        min-height: 12;
        border: solid $accent;
        background: $surface;
        padding: 1 2;
        overflow-y: auto;
    }

    #help-title {
        width: 100%;
        height: 3;
        content-align: center middle;
        text-style: bold;
        color: $accent;
        padding: 0 1;
        margin: 0 0 1 0;
    }

    #help-content {
        width: 100%;
        height: auto;
        overflow-y: auto;
    }

    #help-hints {
        width: 100%;
        height: auto;
        margin: 0 0 1 0;
    }

    #help-footer {
        width: 100%;
        height: 1;
        content-align: center middle;
        color: $text-muted;
        text-style: italic;
        padding: 0 1;
        margin: 1 0 0 0;
    }

    .help-section-label {
        width: 100%;
        height: 1;
        padding: 0 0 0 1;
        margin: 1 0 0 0;
    }
    """

    def compose(self) -> ComposeResult:
        """Create child widgets for the help overlay."""
        with VerticalScroll(id="help-container"):
            yield Label(
                "Keyboard Shortcuts & Navigation Guide",
                id="help-title",
            )

            with VerticalScroll(id="help-content"):
                # Global shortcuts
                yield Label(
                    Text("  [bold]Global Shortcuts[/]  ", style=_STYLE_SECTION),
                    classes="help-section-label",
                )
                yield Static(
                    _make_help_table("Global", _GLOBAL_BINDINGS),
                )

                # Browse view shortcuts
                yield Label(
                    Text("  [bold]Browse View Shortcuts[/]  ", style=_STYLE_SECTION),
                    classes="help-section-label",
                )
                yield Static(
                    _make_help_table("Browse", _BROWSE_BINDINGS),
                )

                # Dashboard view shortcuts
                yield Label(
                    Text("  [bold]Dashboard View Shortcuts[/]  ", style=_STYLE_SECTION),
                    classes="help-section-label",
                )
                yield Static(
                    _make_help_table("Dashboard", _DASHBOARD_BINDINGS),
                )

                # Helpful hints
                yield Label(
                    Text("  [bold]Tips[/]  ", style=_STYLE_SECTION),
                    classes="help-section-label",
                )
                for hint in _HELP_HINTS:
                    yield Label(Text(hint), classes="help-section-label")

            yield Label(
                "  Press [bold $accent]Escape[/] or [bold $accent]?[/] to close this help screen  ",
                id="help-footer",
            )

    def action_dismiss_help(self) -> None:
        """Dismiss the help overlay, returning to the previous screen."""
        self.app.pop_screen()
