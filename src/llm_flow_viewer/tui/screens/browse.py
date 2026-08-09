"""Browse screen with three-panel layout."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import List

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import Screen
from textual.worker import Worker
from textual.widgets import Header, Tree

from llm_flow_viewer.parser.models import LLMCall
from llm_flow_viewer.parser.session import flow_file_to_session
from llm_flow_viewer.tui.widgets.app_footer import AppFooter
from llm_flow_viewer.tui.widgets.call_tree import CallTree, CallTreeNodeData
from llm_flow_viewer.tui.widgets.detail_panel import DetailPanel
from llm_flow_viewer.tui.widgets.session_list import SessionList, SessionInfo

logger = logging.getLogger(__name__)


class BrowseScreen(Screen):
    """The main Browse view showing a three-panel layout.

    Layout (left-to-right):
      - **Session sidebar** (left, ≈20% width)
      - **Call tree** (center, ≈45% width)
      - **Detail panel** (right, ≈35% width)

    A :class:`~textual.widgets.Header` is docked at the top and an
    :class:`AppFooter` at the bottom.
    """

    BINDINGS = [
        Binding("tab", "focus_next", "Next Panel", show=True),
        Binding("shift+tab", "focus_previous", "Previous Panel", show=False),
    ]

    CSS = """
    BrowseScreen {
        /* Use a horizontal layout for the three main panels */
    }

    #browse-container {
        layout: horizontal;
        height: 1fr;
    }

    #sidebar {
        width: 20%;
        min-width: 14;
        border: solid $primary;
        border-title-color: $primary;
    }

    #sidebar:focus-within {
        border: solid $accent;
    }

    #call-tree {
        width: 45%;
        min-width: 20;
        border: solid $primary;
        border-title-color: $primary;
    }

    #call-tree:focus-within {
        border: solid $accent;
    }

    #detail-panel {
        width: 35%;
        min-width: 16;
        height: 1fr;
        border: solid $primary;
        border-title-color: $primary;
    }

    #detail-panel:focus-within {
        border: solid $accent;
    }

    Header {
        dock: top;
    }
    """

    def __init__(
        self,
        flows_dir: str = "./flows",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._flows_dir = flows_dir
        # Track background workers so they can be cancelled during
        # shutdown (VAL-GEN-002, VAL-GEN-005)
        self._workers: list[Worker] = []

        # Streaming load state machine
        self._streaming_worker: Worker | None = None
        """Current streaming worker, if any."""
        self._current_load_id: int = -1
        """Monotonically increasing load identifier.

        Each call to :meth:`_load_session` increments this value.
        Stale callbacks from cancelled loads check this to avoid
        updating the UI with data from a superseded session.
        """
        self._next_load_id: int = 0
        """Next load identifier to assign."""

    def compose(self) -> ComposeResult:
        """Create child widgets for the browse screen."""
        # Use the app's flows_dir if available, otherwise fall back to instance value
        app_flows_dir = getattr(self.app, "flows_dir", self._flows_dir)
        yield Header(show_clock=True)
        with Horizontal(id="browse-container"):
            yield SessionList(id="sidebar", flows_dir=app_flows_dir)
            yield CallTree(id="call-tree")
            yield DetailPanel(id="detail-panel")
        yield AppFooter()

    def on_mount(self) -> None:
        """Set up the screen after mounting."""
        self.query_one("#sidebar", SessionList).focus()
        self._update_border_titles()
        self._update_footer_status()
        # If a session was pre-selected via ``--session``, auto-load it
        # after the session list has had a chance to populate.
        self.set_timer(0.1, self._auto_select_preselected_session)

    def on_unmount(self) -> None:
        """Clean up when the screen is removed.

        Cancels any in-progress streaming worker first, then all
        tracked background workers so the app can shut down promptly
        when the user presses ``q`` (VAL-GEN-002, VAL-GEN-005).
        """
        self._cancel_streaming()
        for worker in list(self._workers):
            if not worker.is_finished and not worker.is_cancelled:
                try:
                    worker.cancel()
                except Exception:
                    logger.debug(
                        "Exception cancelling worker '%s'", worker.name,
                    )
        self._workers.clear()

    def _auto_select_preselected_session(self) -> None:
        """Auto-select a session passed via the ``--session`` CLI flag.

        Checks the app's ``preselected_session`` attribute and,
        if set, finds the matching :class:`SessionInfo` and triggers
        the same loading path as if the user had pressed Enter on the
        session list item.
        """
        preselected = getattr(self.app, "preselected_session", None)
        if not preselected:
            return

        session_list = self.query_one("#sidebar", SessionList)
        for idx, session in enumerate(session_list.sessions):
            session_full_name = f"{session.index:02d}_flows-{session.task_name}"
            if session_full_name == preselected:
                # Highlight the session in the list
                session_list.index = idx
                # Trigger loading (same code path as Enter key)
                self._load_session(session)
                break

    def _update_border_titles(self) -> None:
        """Set border titles for each panel."""
        self.query_one("#sidebar", SessionList).border_title = "Sessions"
        self.query_one("#call-tree", CallTree).border_title = "Call Tree"
        self.query_one("#detail-panel", DetailPanel).border_title = "Details"

    def _update_footer_status(self) -> None:
        """Update the footer with a status message showing session count.

        If sessions were discovered, shows a "Ready — N sessions available"
        message.  If no sessions were found, shows guidance text and updates
        the tree to show an actionable empty-state message.
        """
        try:
            session_list = self.query_one("#sidebar", SessionList)
            footer = self.query_one(AppFooter)
            tree = self.query_one("#call-tree", CallTree)
            count = len(session_list.sessions)
            if count > 0:
                footer.set_status(f"Ready — {count} session{'s' if count != 1 else ''} available")
            else:
                footer.set_status("No flow files found")
                tree.show_placeholder(
                    "No flow files found.\n\n"
                    "Place mitmproxy flow files in the flows/ directory,\n"
                    "or use --flows-dir to specify a custom path."
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Session selection handling
    # ------------------------------------------------------------------

    def on_session_list_session_selected(
        self, event: SessionList.SessionSelected
    ) -> None:
        """Handle user selecting a session from the sidebar.

        Shows a loading indicator in the call tree, then starts a
        streaming background task to parse the session file and
        progressively populate the tree.

        If a previous session was still loading, the prior streaming
        worker is cancelled before starting the new one.

        Args:
            event: The ``SessionSelected`` message from the sidebar.
        """
        event.stop()
        session_info: SessionInfo = event.session

        call_tree = self.query_one("#call-tree", CallTree)
        call_tree.show_loading(session_info.task_name)

        # Detail panel placeholder
        detail_panel = self.query_one("#detail-panel", DetailPanel)
        detail_panel.show_placeholder()

        # Start streaming background loading
        self._load_session(session_info)

    def _load_session(self, session_info: SessionInfo) -> None:
        """Launch a streaming background task to parse the session.

        Uses ``flow_file_to_session`` with a ``progress_callback`` that
        fires for each batch of parsed LLMCalls.  Each batch is appended
        to the CallTree incrementally via :meth:`CallTree.append_calls`,
        keeping the UI responsive during large file parsing.

        Cancels any prior streaming worker before starting (session
        switch support).  Uses a monotonically-increasing load ID to
        guard against stale callbacks from cancelled workers.

        Args:
            session_info: The session to load.
        """
        # Cancel any existing streaming worker
        self._cancel_streaming()

        # Assign a new load ID for this load
        load_id = self._next_load_id
        self._next_load_id += 1
        self._current_load_id = load_id

        call_tree = self.query_one("#call-tree", CallTree)
        call_tree.border_subtitle = "Parsing..."

        # ------------------------------------------------------------------
        # Streaming callback — runs in the worker thread
        # ------------------------------------------------------------------
        def _on_batch(total: int, batch: List[LLMCall]) -> None:
            """Called from the executor thread for each batch.

            Checks the load ID to guard against stale callbacks from
            cancelled loads, then schedules a UI update on the main
            thread via :meth:`~textual.app.App.call_from_thread`.
            """
            # Guard against stale callbacks
            if self._current_load_id != load_id:
                return

            self.app.call_from_thread(self._apply_parse_batch, load_id, total, batch)

        # ------------------------------------------------------------------
        # Worker coroutine — runs in the TUI event loop
        # ------------------------------------------------------------------
        async def _do_streaming_load() -> None:
            loop = asyncio.get_running_loop()
            try:
                session = await loop.run_in_executor(
                    None,
                    lambda: flow_file_to_session(
                        session_info.file_path,
                        session_info.index,
                        session_info.task_name,
                        progress_callback=_on_batch,
                    ),
                )
                # Only finalise if we haven't been superseded
                if self._current_load_id == load_id:
                    self._on_session_loaded(session)
            except asyncio.CancelledError:
                logger.debug(
                    "Streaming load cancelled for %s", session_info.task_name,
                )
            except Exception as exc:
                if self._current_load_id != load_id:
                    # Stale error from cancelled load — ignore
                    return
                logger.error(
                    "Failed to load session %s: %s",
                    session_info.task_name,
                    exc,
                )
                self._on_session_error(session_info.task_name, str(exc))
            finally:
                if self._current_load_id == load_id:
                    self._hide_progress()

        worker = self.run_worker(
            _do_streaming_load(),
            name=f"stream-load-{session_info.task_name}",
        )
        self._streaming_worker = worker
        self._workers.append(worker)

    def _apply_parse_batch(
        self,
        load_id: int,
        total: int,
        batch: List[LLMCall],
    ) -> None:
        """Apply a batch of parsed calls to the tree (main thread).

        Called from :meth:`_load_session`'s streaming callback via
        :meth:`~textual.app.App.call_from_thread`.  Guards against
        stale callbacks and avoids redundant updates when the load has
        been superseded.

        Args:
            load_id: The load identifier this batch belongs to.
            total: Total calls parsed so far (across all batches).
            batch: The latest batch of parsed :class:`LLMCall` objects.
        """
        if self._current_load_id != load_id:
            return

        call_tree = self.query_one("#call-tree", CallTree)
        # Update progress indicator
        call_tree.border_subtitle = f"Parsed {total} calls..."
        # Append calls incrementally
        call_tree.append_calls(batch)

    def _hide_progress(self) -> None:
        """Remove the streaming progress indicator from the call tree."""
        try:
            self.query_one("#call-tree", CallTree).border_subtitle = ""
        except Exception:
            pass

    def _cancel_streaming(self) -> None:
        """Cancel any in-progress streaming worker.

        Sets the load-ID guard so stale callbacks are ignored, cancels
        the worker if running, and resets the progress indicator.
        Safe to call multiple times or when no stream is active.
        """
        # Invalidate any pending callbacks for the current load
        self._current_load_id = -1

        if self._streaming_worker is not None:
            if not self._streaming_worker.is_finished and not self._streaming_worker.is_cancelled:
                try:
                    self._streaming_worker.cancel()
                except Exception:
                    logger.debug("Exception cancelling streaming worker")
            self._streaming_worker = None

        # Reset progress
        self._hide_progress()

    def _on_session_loaded(self, session) -> None:
        """Handle successful session load — populate the tree.

        Also records the loaded session index on the app so the
        Dashboard screen can identify which session is active.

        Args:
            session: The parsed :class:`Session` object.
        """
        call_tree = self.query_one("#call-tree", CallTree)
        call_tree.populate(session)

        # Track selected session on the app for cross-view state
        self.app.selected_session_index = session.index  # type: ignore[union-attr]

        # Record call count for session list annotation
        session_list = self.query_one("#sidebar", SessionList)
        call_count = len(session.calls)
        session_list.mark_session_call_count(session.task_name, call_count)

    def _on_session_error(self, task_name: str, error: str) -> None:
        """Handle session loading error — append error node to tree.

        Preserves any partially-loaded call nodes already in the tree
        and appends an error indicator node to the root (VAL-STREAM-007).

        Args:
            task_name: The name of the session that failed to load.
            error: The error message.
        """
        call_tree = self.query_one("#call-tree", CallTree)
        call_tree.add_error_node(task_name, error)

        # Update footer with error status
        try:
            footer = self.query_one(AppFooter)
            footer.set_status(f"Error loading {task_name}: {error[:60]}")
        except Exception:
            pass

        # Mark the session as errored in the sidebar
        try:
            session_list = self.query_one("#sidebar", SessionList)
            session_list.mark_session_error(task_name, error)
        except Exception:
            logger.exception("Failed to mark session error in sidebar")

    # ------------------------------------------------------------------
    # Tree node selection handling
    # ------------------------------------------------------------------

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle user selecting a tree node with Enter.

        Updates the detail panel with the selected node's content,
        applying syntax highlighting for JSON content or plain text
        for text content.  Also sets the panel border title to reflect
        the selected node type.

        Args:
            event: The ``NodeSelected`` message from the CallTree.
        """
        event.stop()
        node_data = event.node.data

        if node_data is None or not isinstance(node_data, CallTreeNodeData):
            return

        # Track whether this is a new (different) node for scroll management.
        # When a different node is selected, scroll resets to top.
        # When the same node is re-selected, scroll position is preserved.
        is_new_node = node_data is not getattr(self, "_last_detail_node_data", None)
        self._last_detail_node_data = node_data

        detail_panel = self.query_one("#detail-panel", DetailPanel)
        content = self._get_node_content(node_data)
        title = self._get_node_title(node_data)

        if content:
            detail_panel.show_content(content, title=title)
            if is_new_node:
                detail_panel.scroll_home(animate=False)
        else:
            detail_panel.show_placeholder()

    def load_session_by_index(self, session_index: int) -> None:
        """Programmatically load a session by its numeric index.

        Finds the session in the sidebar's session list and triggers
        the same loading path as if the user had selected it manually.

        This method is called from DashboardScreen when a session is
        focused in the dashboard and the user returns to Browse (``b``),
        enabling cross-view session focus propagation (VAL-CROSS-006).

        Args:
            session_index: The numeric session index to load.
        """
        session_list = self.query_one("#sidebar", SessionList)
        for idx, s_info in enumerate(session_list.sessions):
            if s_info.index == session_index:
                # Highlight the session in the list
                session_list.index = idx
                # Trigger loading (same code path as Enter key)
                self._load_session(s_info)
                break

    @staticmethod
    def _get_node_title(data: CallTreeNodeData) -> str:
        """Return a human-readable border title for the selected node.

        Args:
            data: The :class:`CallTreeNodeData` for the selected node.

        Returns:
            A short title string (e.g. ``"Request Details"``, ``"Tool Call"``).
        """
        node_type = data.node_type
        section_type = data.section_type

        # Section nodes use section_type
        if node_type == "section" and section_type:
            return {
                "request_details": "Request Details",
                "response_details": "Response Details",
                "tool_calls": "Tool Calls & Results",
                "timing": "Timing",
                "token_usage": "Token Usage",
            }.get(section_type, "Section")

        # Field nodes use field_key
        if node_type == "field":
            return data.field_key or "Field"

        # Message nodes include role
        if node_type == "message":
            role = data.message_role.capitalize() if data.message_role else "Message"
            return f"Message ({role})"

        # Content blocks use type
        if node_type == "content_block":
            ctype = data.content_block_type.capitalize() if data.content_block_type else "Content"
            return f"Content ({ctype})"

        # Tool-related nodes
        if node_type == "tool_call_node":
            return f"Tool Call: {data.tool_name}" if data.tool_name else "Tool Call"
        if node_type == "tool_call_input":
            return "Input Parameters"
        if node_type == "tool_result_node":
            return f"Tool Result: {data.tool_name}" if data.tool_name else "Tool Result"
        if node_type == "tool_input_schema":
            return "Input Schema"
        if node_type == "tool":
            return f"Tool: {data.tool_name}" if data.tool_name else "Tool Definition"

        # Response nodes
        if node_type == "response_text":
            return "Text Output"
        if node_type == "response_thinking":
            return "Thinking"
        if node_type == "response_sse_event":
            return "SSE Event"
        if node_type == "response_raw_sse_header":
            return "Raw SSE Events"

        # System prompt nodes
        if node_type == "system_text":
            return "System Prompt"
        if node_type == "system_message":
            return "System Prompt"
        if node_type == "system_header":
            return "System Prompts"

        # Header nodes
        if node_type == "messages_header":
            return "Messages"
        if node_type == "tools_header":
            return "Tools"

        # Structured data nodes
        if node_type == "raw_request":
            return "Raw Request (JSON)"
        if node_type == "output_config":
            return "Output Config"

        # Top-level nodes
        if node_type == "call":
            return f"Call #{data.call_index + 1}"
        if node_type == "session":
            return "Session"

        # Error / Loading nodes
        if node_type == "error":
            return "Flow Read Error"
        if node_type == "loading":
            return "Loading"

        # Fallback
        return "Details"

    @staticmethod
    def _get_node_content(data: CallTreeNodeData) -> str:
        """Extract display content from a node's data.

        Args:
            data: The :class:`CallTreeNodeData` for the selected node.

        Returns:
            A string suitable for display in the detail panel.
        """
        node_type = data.node_type

        # Priority: use full_content if available
        if data.full_content:
            return data.full_content

        # For field nodes, show the field value
        if node_type == "field":
            return data.field_value or data.summary

        # For call nodes, show a summary
        if node_type == "call" and data.call is not None:
            call = data.call
            parts = [f"Call #{data.call_index + 1}"]
            if call.request and call.request.model:
                parts.append(f"Model: {call.request.model}")
            if call.request and call.request.max_tokens:
                parts.append(f"Max Tokens: {call.request.max_tokens}")
            if call.request_id:
                parts.append(f"Request ID: {call.request_id}")
            return "\n".join(parts)

        # For section nodes, show overview
        if node_type == "section":
            return data.summary

        # For message nodes, show the full message as JSON if available
        if node_type == "message":
            # full_content is now set to the JSON of the full message
            return data.full_content or data.content_preview or ""

        # For content blocks, return the preview
        if node_type == "content_block":
            return data.content_preview or data.full_content or ""

        # For raw request, output_config, tool_call_input: these have full_content
        if node_type in ("raw_request", "output_config", "tool_input_schema", "tool_call_input"):
            return data.full_content

        # For tool call nodes, show tool name + ID + input params as JSON
        if node_type == "tool_call_node":
            if data.full_content:
                return data.full_content
            parts = [f"Tool: {data.tool_name} ({data.tool_call_id})"]
            return "\n".join(parts)

        # For tool result nodes, show the result content (with error details
        # if applicable, already built into full_content)
        if node_type == "tool_result_node":
            if data.full_content:
                return data.full_content
            return data.summary

        # Error / Loading nodes
        if node_type == "error":
            return data.full_content or data.summary
        if node_type == "loading":
            return data.summary

        # Fallback
        return data.summary
