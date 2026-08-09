"""Session list sidebar widget."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from textual.binding import Binding
from textual.widgets import ListView, ListItem, Label
from textual.message import Message

logger = logging.getLogger(__name__)

SESSION_FILE_PATTERN = re.compile(r"^(\d+)_flows-(.+)$")


@dataclass
class SessionInfo:
    """Metadata about a discovered session."""

    index: int
    task_name: str
    file_path: str


def discover_sessions(
    flows_dir: str,
    recursive: bool = True,
    progress_callback=None,
) -> List[SessionInfo]:
    """Discover session flow files in the given directory.

    Scans *flows_dir* (and optionally its subdirectories) for files
    matching the pattern ``{index}_flows-{name}``
    (e.g., ``01_flows-analyze_codebase``).
    Files that do not match the numeric pattern are also discovered and
    assigned synthetic indices based on file modification time (newest
    first), starting from ``max(numeric_index) + 1`` (or 1000 if no
    numeric sessions exist).

    Args:
        flows_dir: Absolute path to the directory containing flow files.
        recursive: If ``True`` (default), also scan subdirectories
            recursively.
        progress_callback: Optional callable ``(current, total, message)``
            invoked during scanning to report progress.

    Returns:
        A list of :class:`SessionInfo` sorted by index, or an empty list if
        the directory does not exist or contains no matching files.
    """
    sessions: List[SessionInfo] = []
    # Collect non-matching entries for synthetic index assignment
    non_numeric_entries: List[os.DirEntry] = []
    if not os.path.isdir(flows_dir):
        logger.warning("Flows directory does not exist: %s", flows_dir)
        return sessions

    _do_discover(flows_dir, sessions, non_numeric_entries, recursive, progress_callback)

    # Sort non-numeric entries by mtime (newest first), tiebreak by filename
    non_numeric_entries.sort(
        key=lambda e: (e.stat().st_mtime, e.name),
        reverse=True,
    )

    # Determine starting index for synthetic sessions
    if sessions:
        numeric_max = max(s.index for s in sessions)
        synthetic_start = numeric_max + 1
    else:
        synthetic_start = 1000

    # Assign synthetic indices based on modification time (newest first)
    for i, entry in enumerate(non_numeric_entries):
        task_name = _derive_task_name(entry.name)
        synthetic_index = synthetic_start + i
        sessions.append(SessionInfo(
            index=synthetic_index,
            task_name=task_name,
            file_path=entry.path,
        ))

    sessions.sort(key=lambda s: s.index)
    return sessions


def _derive_task_name(filename: str) -> str:
    """Derive a human-readable task name from a flow filename.

    For files containing the ``_flows-`` separator (e.g.
    ``my_session_flows-debug``), returns the part after the separator.
    For files without this pattern, returns the filename as-is.

    Args:
        filename: The base name of the flow file.

    Returns:
        A task name string.
    """
    # Check for the _flows- separator (used in both numeric and non-numeric files)
    sep = "_flows-"
    idx = filename.find(sep)
    if idx != -1:
        return filename[idx + len(sep):]
    return filename


def _do_discover(
    directory: str,
    sessions: List[SessionInfo],
    non_numeric_entries: List[os.DirEntry],
    recursive: bool,
    progress_callback=None,
    _depth: int = 0,
) -> None:
    """Recursively scan *directory* for session flow files.

    Matching files (those fitting the ``NN_flows-*`` pattern) are added to
    *sessions*.  Non-matching flow files (non-parquet, non-zip) are added to
    *non_numeric_entries* for later synthetic index assignment.

    Args:
        directory: The directory to scan.
        sessions: Accumulator list for discovered numeric sessions.
        non_numeric_entries: Accumulator list for non-matching file entries.
        recursive: Whether to recurse into subdirectories.
        progress_callback: Optional progress callback.
        _depth: Current recursion depth (used to avoid infinite loops).
    """
    if _depth > 50:
        # Safety limit to prevent infinite recursion on circular symlinks
        return

    try:
        entries = list(os.scandir(directory))
    except PermissionError:
        logger.warning("Permission denied scanning directory: %s", directory)
        return

    for entry in entries:
        try:
            if entry.is_dir(follow_symlinks=False):
                if recursive:
                    _do_discover(
                        entry.path,
                        sessions,
                        non_numeric_entries,
                        recursive,
                        progress_callback,
                        _depth + 1,
                    )
            elif entry.is_file(follow_symlinks=False):
                name = entry.name
                # Skip parquet cache files and other non-flow files
                if name == ".dashboard_metrics.json":
                    continue
                if not name.endswith((".parquet", ".zip")):
                    match = SESSION_FILE_PATTERN.match(name)
                    if match:
                        index = int(match.group(1))
                        task_name = match.group(2)
                        sessions.append(SessionInfo(
                            index=index,
                            task_name=task_name,
                            file_path=entry.path,
                        ))
                    else:
                        # Non-matching flow file — collect for synthetic index
                        non_numeric_entries.append(entry)
        except OSError:
            # Skip entries that can't be accessed
            continue

    if progress_callback:
        progress_callback(len(sessions), 0, f"Scanned {directory}")


class SessionList(ListView):
    """Widget displaying the list of discovered sessions.

    Composes a :class:`~textual.widgets.ListView` populated with
    session entries.  The user can navigate with Up/Down and press
    Enter to select a session (which posts a
    :class:`SessionSelected` message for the parent screen to handle).

    Long session names are truncated with an ellipsis (``...``) when
    the sidebar is narrower than the label.  A tooltip on each item
    shows the full session label.
    """

    DEFAULT_CSS = """
    SessionList {
        overflow: hidden;
    }

    SessionList > ListItem {
        overflow: hidden;
        min-height: 1;
        max-height: 1;
    }

    SessionList > ListItem > Label {
        overflow: hidden;
        min-height: 1;
        max-height: 1;
        text-style: none;
    }
    """

    BINDINGS = [
        Binding("enter", "select_cursor", "Load Session", show=True),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
    ]

    class SessionSelected(Message):
        """Posted when the user selects a session from the list.

        Attributes:
            session: The selected session info.
        """

        def __init__(self, session: SessionInfo) -> None:
            super().__init__()
            self.session = session

    def __init__(self, flows_dir: str = "./flows", **kwargs):
        super().__init__(**kwargs)
        self._flows_dir = flows_dir
        self._sessions: List[SessionInfo] = []
        # Track per-session errors by task_name
        self._session_errors: Dict[str, str] = {}
        # Track per-session call counts for annotation
        self._session_call_counts: Dict[str, int] = {}

    def on_mount(self) -> None:
        """Discover sessions and populate the list."""
        self._sessions = discover_sessions(self._flows_dir)
        self._populate()
        if self._sessions:
            self.index = 0  # Highlight first item

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle the ListView.Selected message and post our own.

        When the user presses Enter (or clicks) on a session item,
        this handler posts a :class:`SessionSelected` message with
        the corresponding :class:`SessionInfo`.

        Args:
            event: The ``ListView.Selected`` message from the parent.
        """
        # Stop the event so other handlers don't process it
        event.stop()
        selected = self.selected_session
        if selected is not None:
            self.post_message(self.SessionSelected(selected))

    # ------------------------------------------------------------------
    # Session state tracking
    # ------------------------------------------------------------------

    def mark_session_error(self, task_name: str, error_message: str) -> None:
        """Mark a session as having a parse error.

        The session label will be updated with an error indicator (⚠).

        Args:
            task_name: The task name of the session (e.g. ``"analyze_codebase"``).
            error_message: The error description.
        """
        self._session_errors[task_name] = error_message
        self._populate()

    def mark_session_call_count(self, task_name: str, call_count: int) -> None:
        """Record the number of API calls for a session.

        Used to show ``(0 calls)`` annotation in the session list.

        Args:
            task_name: The task name of the session.
            call_count: The number of API calls in the session.
        """
        self._session_call_counts[task_name] = call_count
        self._populate()

    def clear_session_state(self, task_name: str) -> None:
        """Clear error/call-count state for a session.

        Args:
            task_name: The task name of the session.
        """
        self._session_errors.pop(task_name, None)
        self._session_call_counts.pop(task_name, None)
        # Only repopulate if the session still exists
        for s in self._sessions:
            if s.task_name == task_name:
                self._populate()
                break

    def has_error(self, task_name: str) -> bool:
        """Check whether a session has a recorded error.

        Args:
            task_name: The task name of the session.

        Returns:
            True if the session has an error.
        """
        return task_name in self._session_errors

    def get_error_message(self, task_name: str) -> Optional[str]:
        """Get the error message for a session, if any.

        Args:
            task_name: The task name of the session.

        Returns:
            The error message, or ``None`` if no error.
        """
        return self._session_errors.get(task_name)

    def _populate(self) -> None:
        """Clear and repopulate the ListView with session entries."""
        # Save current index before clear
        prev_index = self.index
        self.clear()
        # Compute available width from content region
        avail = self.content_region.width
        max_label_len = max(avail - 1, 8) if avail > 0 else 40
        for session in self._sessions:
            full_label = self._session_label(session)
            # Add error indicator
            if session.task_name in self._session_errors:
                full_label = f"\u26a0 {full_label}"
            # Add call-count annotation for zero-call sessions
            call_count = self._session_call_counts.get(session.task_name)
            if call_count is not None and call_count == 0:
                full_label = f"{full_label} (0 calls)"

            display_label = self._truncate_label(full_label, max_label_len)
            label = Label(display_label)

            # Style error items with red color
            if session.task_name in self._session_errors:
                label.styles.color = "red"

            item = ListItem(label)
            # Set tooltip to show the full label when truncated
            tooltip_text = full_label
            if session.task_name in self._session_errors:
                tooltip_text += f"\nError: {self._session_errors[session.task_name]}"
            item.tooltip = tooltip_text
            self.append(item)

        # Restore previous index if valid
        if prev_index is not None and 0 <= prev_index < len(self._sessions):
            self.index = prev_index
        elif self._sessions:
            self.index = 0

    @staticmethod
    def _session_label(session: SessionInfo) -> str:
        """Format a human-readable label for a session entry.

        Args:
            session: The session info.

        Returns:
            A formatted label string, e.g. ``"01 | analyze_codebase"``.
        """
        return f"{session.index:02d} | {session.task_name}"

    @staticmethod
    def _truncate_label(label: str, max_length: int = 40) -> str:
        """Truncate a label to *max_length* characters, appending ``...``.

        Args:
            label: The full label string.
            max_length: Maximum allowed length before truncation.

        Returns:
            The truncated label (with ``...``) if longer than *max_length*,
            otherwise the original label.
        """
        if len(label) > max_length:
            return label[: max_length - 3] + "..."
        return label

    @property
    def sessions(self) -> List[SessionInfo]:
        """The list of discovered sessions."""
        return list(self._sessions)

    @property
    def selected_session(self) -> SessionInfo | None:
        """The session at the currently highlighted index, or *None*."""
        idx = self.index
        if idx is not None and self._sessions and 0 <= idx < len(self._sessions):
            return self._sessions[idx]
        return None
