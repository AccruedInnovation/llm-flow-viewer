"""Detail panel widget for displaying selected node content."""

from __future__ import annotations

import json

from rich.syntax import Syntax
from textual.binding import Binding
from textual.widgets import RichLog


class DetailPanel(RichLog):
    """A content panel that displays the full details of the currently
    selected tree node.

    When no node is selected, the panel shows a placeholder message.
    When a node is selected, its associated data is rendered — either as
    syntax-highlighted JSON for structured data or as formatted plain
    text for content blocks.

    The panel is scrollable for long content.  The border title reflects
    the type of node currently being displayed (e.g. ``"Request Details"``,
    ``"Tool Call"``, ``"Details"``).
    """

    BINDINGS = [
        Binding("up", "scroll_up", "Up", show=False),
        Binding("down", "scroll_down", "Down", show=False),
        Binding("page_up", "page_up", "Page Up", show=False),
        Binding("page_down", "page_down", "Page Down", show=False),
    ]

    PLACEHOLDER_TEXT = "Select a node to view details"

    DEFAULT_TITLE = "Details"

    # Allow the panel to receive keyboard focus for scrolling and focus cycling
    can_focus = True

    # Dark theme for JSON syntax highlighting
    _JSON_THEME = "monokai"

    def __init__(self, **kwargs):
        kwargs.setdefault("wrap", True)
        kwargs.setdefault("min_width", 0)
        super().__init__(**kwargs)
        self._last_displayed_content: str | None = None
        self._last_displayed_title: str | None = None

    @property
    def content(self) -> str:
        """Return the currently displayed content string.

        Provides backward compatibility with ``Static``-based tests that
        accessed ``.content`` to verify the panel text.
        """
        return self._last_displayed_content or self.PLACEHOLDER_TEXT

    def on_mount(self) -> None:
        """Perform initial setup after mount."""
        self.styles.overflow_y = "auto"
        self.styles.overflow_x = "hidden"
        self.styles.padding = (0, 1)
        self.border_title = self.DEFAULT_TITLE
        self.clear()
        self.write(self.PLACEHOLDER_TEXT, scroll_end=False)

    def set_title(self, title: str) -> None:
        """Set the border title to reflect the displayed content type.

        Args:
            title: The title string to display in the panel border.
        """
        self.border_title = title

    def show_placeholder(self) -> None:
        """Reset the panel to the default placeholder message."""
        self.set_title(self.DEFAULT_TITLE)
        self.clear()
        self.write(self.PLACEHOLDER_TEXT, scroll_end=False)
        self._last_displayed_content = self.PLACEHOLDER_TEXT
        self._last_displayed_title = self.DEFAULT_TITLE
        self.scroll_home(animate=False)

    def show_content(self, content: str, title: str = "") -> None:
        """Display the given content in the panel.

        Automatically detects JSON content and applies syntax highlighting.
        The *title* is forwarded to the appropriate display method.

        Args:
            content: The text or JSON content to display.
            title: Optional border title reflecting the node type.
        """
        if self._is_json_content(content):
            self.show_json(content, title=title)
        else:
            self.show_text(content, title=title)

    def show_json(self, json_str: str, title: str = "JSON") -> None:
        """Display content as syntax-highlighted JSON.

        Args:
            json_str: A JSON string to display with highlighting.
            title: Optional border title for the panel.
        """
        if title:
            self.set_title(title)
        self.clear()
        try:
            syntax = Syntax(
                json_str,
                "json",
                theme=self._JSON_THEME,
                word_wrap=True,
                line_numbers=False,
            )
            self.write(syntax, scroll_end=False)
        except Exception:
            # Fallback to plain text if syntax highlighting fails
            self.write(json_str, scroll_end=False)
        self._last_displayed_content = json_str
        self._last_displayed_title = title

    def show_text(self, text: str, title: str = "") -> None:
        """Display content as plain text with preserved whitespace.

        Args:
            text: The plain text content to display.
            title: Optional border title for the panel.
        """
        if title:
            self.set_title(title)
        self.clear()
        self.write(text, scroll_end=False)
        self._last_displayed_content = text
        self._last_displayed_title = title

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_json_content(content: str) -> bool:
        """Check if a string is valid JSON content.

        Args:
            content: The string to check.

        Returns:
            ``True`` if the string is parseable as JSON.
        """
        stripped = content.strip()
        if not stripped:
            return False
        # Quick check for JSON-like start
        if stripped[0] not in ("{", "["):
            return False
        try:
            json.loads(stripped)
            return True
        except (json.JSONDecodeError, ValueError):
            return False
