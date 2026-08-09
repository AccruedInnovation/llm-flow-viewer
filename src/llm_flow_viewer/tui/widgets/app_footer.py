"""Custom footer widget showing context-sensitive keyboard shortcut hints."""

from __future__ import annotations

from textual.reactive import reactive
from textual.widgets import Static


_HINT_MAP: dict[str | None, str] = {
    "sidebar": "Enter:Load Session  Down/Up:Navigate  ?:Help  b:Browse  d:Dash  q:Quit",
    "call-tree": "Enter:Select  Space:Expand  Shift+Space:All  Down/Up:Navigate  Tab:Next Panel  ?:Help  b:Browse  d:Dash  q:Quit",
    "detail-panel": "Down/Up:Scroll  ?:Help  b:Browse  d:Dash  q:Quit",
}

_DEFAULT_HINTS = "Tab:Next Panel  ?:Help  b:Browse  d:Dash  q:Quit"


class AppFooter(Static):
    """A footer that displays context-sensitive keyboard shortcut hints
    based on the currently focused widget.

    The hint text updates automatically when the focused widget changes
    by watching the screen's ``focused`` reactive attribute.

    A persistent status message can be set via :meth:`set_status`,
    which is displayed alongside the keyboard hints.  The status is
    shown on the left side, followed by the keyboard hints on the right.
    """

    status_message = reactive("", layout=True)
    """A persistent status message shown on the left side of the footer."""

    DEFAULT_CSS = """
    AppFooter {
        dock: bottom;
        height: 1;
        padding: 0 1;
        color: $text-muted;
        background: $panel;
    }
    """

    def on_mount(self) -> None:
        """Set up the watcher on the screen's focused attribute."""
        self.screen.watch(self.screen, "focused", self._on_screen_focus_changed)
        self._update_hints()

    def _on_screen_focus_changed(self, focused: object) -> None:
        """React to focus changes on the screen.

        Args:
            focused: The newly focused widget (or ``None``).
        """
        self._update_hints()

    def watch_status_message(self, old_value: str, new_value: str) -> None:
        """Re-render the footer when the status message changes."""
        self._update_hints()

    def set_status(self, message: str) -> None:
        """Set a persistent status message shown in the footer.

        The message is displayed on the left side of the footer,
        followed by the keyboard shortcut hints on the right.
        Pass an empty string to clear the status message.

        Args:
            message: The status message to display (e.g.
                ``"Ready — 7 sessions available"``).
        """
        self.status_message = message

    def clear_status(self) -> None:
        """Clear the status message, showing only keyboard hints."""
        self.status_message = ""

    def _update_hints(self) -> None:
        """Update the displayed hint text based on the currently focused widget."""
        focused = self.screen.focused
        widget_id = getattr(focused, "id", None)
        hint = _HINT_MAP.get(widget_id, _DEFAULT_HINTS)

        if self.status_message:
            self.update(f"{self.status_message}  |  {hint}")
        else:
            self.update(hint)
