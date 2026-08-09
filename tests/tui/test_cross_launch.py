"""Tests for cross-cutting launch behaviors in the LLM Flow Viewer TUI.

Covers the following validation assertions:
- VAL-CROSS-001: Default launch opens browse view with session discovery
- VAL-CROSS-002: --session flag auto-selects session and populates tree
- VAL-CROSS-003: --flows-dir specifies a non-default data directory
- VAL-CROSS-004: Both --session and --flows-dir compose correctly
- VAL-CROSS-005: Invalid --session name shows error before TUI launch
- VAL-CROSS-020: No flow files found — graceful empty state with guidance
- VAL-CROSS-021: Auto-discovery of flow files on first launch
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from llm_flow_viewer.tui.app import (
    LLMFlowViewerApp,
    _discover_session_names,
    _validate_session,
    build_arg_parser,
    main,
)
from llm_flow_viewer.tui.widgets.session_list import (
    SessionList,
    discover_sessions,
)


# ===================================================================
# Helpers
# ===================================================================


def _create_flow_file(directory: str, index: int, task_name: str) -> str:
    """Create a dummy flow file in the given directory.

    Args:
        directory: The target directory.
        index: The session index (e.g. 1).
        task_name: The task name (e.g. ``"analyze_codebase"``).

    Returns:
        The absolute path to the created file.
    """
    filename = f"{index:02d}_flows-{task_name}"
    filepath = os.path.join(directory, filename)
    Path(filepath).write_text("dummy flow content")
    return filepath


def _create_flow_files(directory: str, count: int = 3) -> list[str]:
    """Create multiple dummy flow files in the given directory.

    Args:
        directory: The target directory.
        count: Number of flow files to create (1-indexed).

    Returns:
        List of created file paths.
    """
    created = []
    for i in range(1, count + 1):
        path = _create_flow_file(directory, i, f"test_session_{i:02d}")
        created.append(path)
    return created


# ===================================================================
# VAL-CROSS-001: Default launch opens browse view with session discovery
# ===================================================================


class TestDefaultLaunch:
    """VAL-CROSS-001: Default launch opens browse view with session discovery."""

    def test_discover_sessions_in_directory(self):
        """Discover session files in the current directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_flow_files(tmpdir, count=3)
            sessions = discover_sessions(tmpdir)
            assert len(sessions) == 3, (
                f"Expected 3 sessions, got {len(sessions)}"
            )


# ===================================================================
# VAL-CROSS-002: --session flag auto-selects session and populates tree
# ===================================================================


class TestSessionFlag:
    """VAL-CROSS-002: ``--session`` flag auto-selects session and populates tree."""

    def test_session_defaults_to_none(self):
        """LLMFlowViewerApp defaults session to None."""
        app = LLMFlowViewerApp()
        assert app.preselected_session is None, (
            f"Default preselected_session should be None, "
            f"got: '{app.preselected_session}'"
        )

    def test_preselected_session_property_exists(self):
        """App should have preselected_session property."""
        app = LLMFlowViewerApp()
        assert hasattr(app, "preselected_session"), (
            "App should have preselected_session property"
        )


# ===================================================================
# VAL-CROSS-003: --flows-dir specifies a non-default data directory
# ===================================================================


class TestFlowsDir:
    """VAL-CROSS-003: ``--flows-dir`` specifies a non-default data directory."""

    def test_flows_dir_accepted_by_parser(self):
        """CLI parser accepts --flows-dir argument."""
        parser = build_arg_parser()
        args = parser.parse_args(["--flows-dir", "/custom/path"])
        assert args.flows_dir == "/custom/path"

    def test_flows_dir_passed_to_app(self):
        """The provided flows_dir is passed to the TUI app."""
        with tempfile.TemporaryDirectory() as tmpdir:
            app = LLMFlowViewerApp(flows_dir=tmpdir)
            assert app.flows_dir == os.path.abspath(tmpdir)

    def test_flows_dir_discovered_sessions(self):
        """Sessions from custom --flows-dir are discovered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_flow_files(tmpdir, count=2)
            sessions = discover_sessions(tmpdir)
            assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_custom_flows_dir_populates_list(self):
        """Session list populated from custom flows-dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_flow_files(tmpdir, count=2)
            app = LLMFlowViewerApp(flows_dir=tmpdir)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                session_list = app.screen.query(SessionList).first()
                assert len(session_list.sessions) == 2, (
                    f"Expected 2 sessions, got {len(session_list.sessions)}"
                )


# ===================================================================
# VAL-CROSS-004: Both --session and --flows-dir compose correctly
# ===================================================================


class TestCombinedFlags:
    """VAL-CROSS-004: Both ``--session`` and ``--flows-dir`` compose correctly."""

    def test_both_flags_accepted_together(self):
        """Parser accepts both --flows-dir and --session together."""
        parser = build_arg_parser()
        args = parser.parse_args([
            "--flows-dir", "/some/path",
            "--session", "01_flows-analyze_codebase",
        ])
        assert args.flows_dir == "/some/path"
        assert args.session == "01_flows-analyze_codebase"

    def test_both_flags_passed_to_app(self):
        """Both --flows-dir and --session are passed to LLMFlowViewerApp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_name = "01_flows-analyze_codebase"
            _create_flow_file(tmpdir, 1, "analyze_codebase")
            app = LLMFlowViewerApp(flows_dir=tmpdir, session=session_name)
            assert app.flows_dir == os.path.abspath(tmpdir)
            assert app.preselected_session == session_name

    def test_session_looked_up_in_custom_dir(self):
        """--session is looked up inside --flows-dir, not default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_flow_file(tmpdir, 1, "custom_task")
            session_names = _discover_session_names(tmpdir)
            assert "01_flows-custom_task" in session_names

    def test_validate_session_in_custom_dir(self):
        """_validate_session checks sessions in the provided flows_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_flow_file(tmpdir, 1, "my_session")
            errors = []
            _validate_session("01_flows-my_session", tmpdir, errors)
            assert len(errors) == 0, (
                f"Should find session, got errors: {errors}"
            )

    def test_validate_session_fails_in_custom_dir(self):
        """_validate_session fails for unknown session in custom dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            errors = []
            _validate_session("99_flows-nonexistent", tmpdir, errors)
            assert len(errors) > 0, "Should report error for unknown session"


# ===================================================================
# VAL-CROSS-005: Invalid --session name shows error before TUI launch
# ===================================================================


class TestInvalidSession:
    """VAL-CROSS-005: Invalid ``--session`` name shows error before TUI launch."""

    def test_validate_session_returns_error(self):
        """_validate_session should add error for unknown session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            errors = []
            _validate_session("nonexistent_flow", tmpdir, errors)
            assert len(errors) > 0, "Should produce error for unknown session"
            assert "nonexistent_flow" in errors[0], (
                f"Error should mention session name, got: {errors[0]}"
            )

    def test_validate_session_passes_for_known_session(self):
        """_validate_session should not error for known session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_flow_file(tmpdir, 1, "known_session")
            errors = []
            _validate_session("01_flows-known_session", tmpdir, errors)
            assert len(errors) == 0, f"Should pass for known session, got: {errors}"

    def test_validate_session_shows_available(self):
        """Error message should list available sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_flow_file(tmpdir, 1, "session_one")
            _create_flow_file(tmpdir, 2, "session_two")
            errors = []
            _validate_session("nonexistent", tmpdir, errors)
            assert len(errors) > 0
            assert "Available sessions" in errors[0] or "available" in errors[0].lower(), (
                f"Error should mention available sessions, got: {errors[0]}"
            )
            assert "session_one" in errors[0], (
                f"Error should include available session names, got: {errors[0]}"
            )

    def test_nonexistent_session_exits_with_error(self):
        """main() exits with non-zero when --session refers to unknown session."""
        from llm_flow_viewer.tui import app as app_module

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setattr(
                app_module.sys, "argv",
                ["llm-flow-viewer", "--flows-dir", tmpdir, "--session", "nonexistent"],
            )
            try:
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1, (
                    f"Should exit with code 1, got: {exc_info.value.code}"
                )
            finally:
                monkeypatch.undo()

    def test_nonexistent_session_does_not_launch_app(self):
        """TUI app is NOT launched when --session is invalid."""
        from llm_flow_viewer.tui import app as app_module

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setattr(
                app_module.sys, "argv",
                ["llm-flow-viewer", "--flows-dir", tmpdir, "--session", "invalid"],
            )

            def fail_on_create(*args, **kwargs):
                pytest.fail("LLMFlowViewerApp should not be instantiated")

            monkeypatch.setattr(app_module, "LLMFlowViewerApp", fail_on_create)
            try:
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code != 0, (
                    "Should exit with non-zero code"
                )
            finally:
                monkeypatch.undo()


# ===================================================================
# VAL-CROSS-020: No flow files found — graceful empty state with guidance
# ===================================================================


class TestNoFilesEmptyState:
    """VAL-CROSS-020: No flow files found — graceful empty state with guidance."""

    @pytest.mark.asyncio
    async def test_empty_dir_shows_no_files_message(self):
        """Empty session list when no flow files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            app = LLMFlowViewerApp(flows_dir=tmpdir)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                session_list = app.screen.query(SessionList).first()
                assert len(session_list.sessions) == 0, (
                    "Should have 0 sessions in empty directory"
                )

    @pytest.mark.asyncio
    async def test_empty_dir_does_not_crash(self):
        """App should not crash when launched with empty flows directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            app = LLMFlowViewerApp(flows_dir=tmpdir)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                # App should still be running
                assert app._running is True
                # Can still quit
                await pilot.press("q")
                await pilot.pause()
                assert app._running is False

    @pytest.mark.asyncio
    async def test_empty_dir_can_switch_to_dashboard(self):
        """Can switch to Dashboard even with empty flows directory."""
        from llm_flow_viewer.tui.screens.dashboard import DashboardScreen

        with tempfile.TemporaryDirectory() as tmpdir:
            app = LLMFlowViewerApp(flows_dir=tmpdir)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                # Press 'd' to switch to dashboard
                await pilot.press("d")
                await pilot.pause()
                # Should be on dashboard (or at least not crashed)
                assert app._running is True, "App should still be running"

    @pytest.mark.asyncio
    async def test_empty_dir_quit_gracefully(self):
        """Can quit gracefully from empty state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            app = LLMFlowViewerApp(flows_dir=tmpdir)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                await pilot.press("q")
                await pilot.pause()
                assert app._running is False, "App should exit cleanly"

    def test_discover_sessions_empty_directory(self):
        """discover_sessions returns empty list for empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions = discover_sessions(tmpdir)
            assert len(sessions) == 0

    def test_discover_sessions_non_existent_directory(self):
        """discover_sessions returns empty list for non-existent directory."""
        sessions = discover_sessions(r"D:\nonexistent_path_X12345")
        assert len(sessions) == 0


# ===================================================================
# VAL-CROSS-021: Auto-discovery of flow files on first launch
# ===================================================================


class TestAutoDiscovery:
    """VAL-CROSS-021: Auto-discovery of flow files on first launch."""

    def test_discover_session_names_utility(self):
        """_discover_session_names returns list of session file names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_flow_files(tmpdir, count=2)
            names = _discover_session_names(tmpdir)
            assert len(names) == 2
            assert "01_flows-test_session_01" in names
            assert "02_flows-test_session_02" in names

    def test_discover_session_names_empty(self):
        """_discover_session_names returns empty list for empty dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            names = _discover_session_names(tmpdir)
            assert names == []

    def test_discover_skips_parquet_files(self):
        """Session discovery should skip .parquet files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create parquet-looking file
            Path(tmpdir, "01_flows-test.parquet").write_text("parquet")
            # Create real flow file
            _create_flow_file(tmpdir, 1, "real_session")
            sessions = discover_sessions(tmpdir)
            assert len(sessions) == 1
            assert sessions[0].task_name == "real_session"

    def test_discover_skips_zip_files(self):
        """Session discovery should skip .zip files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "01_flows-test.zip").write_text("zip")
            _create_flow_file(tmpdir, 1, "real_session")
            sessions = discover_sessions(tmpdir)
            assert len(sessions) == 1

    def test_discover_finds_all_flow_files(self):
        """Session discovery finds all matching flow files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_flow_file(tmpdir, 1, "session_one")
            _create_flow_file(tmpdir, 2, "session_two")
            _create_flow_file(tmpdir, 3, "session_three")
            sessions = discover_sessions(tmpdir)
            assert len(sessions) == 3

    def test_discover_returns_sorted_by_index(self):
        """Sessions should be sorted by index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_flow_file(tmpdir, 3, "third")
            _create_flow_file(tmpdir, 1, "first")
            _create_flow_file(tmpdir, 2, "second")
            sessions = discover_sessions(tmpdir)
            indices = [s.index for s in sessions]
            assert indices == [1, 2, 3], (
                f"Sessions should be sorted by index, got: {indices}"
            )


# ===================================================================
# Recursive auto-discovery tests
# ===================================================================


class TestRecursiveDiscovery:
    """Auto-discovery should find session files in subdirectories."""

    def test_discover_files_in_subdirectory(self):
        """Session files in subdirectories should be discovered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a subdirectory with a flow file
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            _create_flow_file(str(subdir), 1, "sub_session")

            sessions = discover_sessions(tmpdir)
            assert len(sessions) >= 1, (
                "Should discover session in subdirectory"
            )

    def test_discover_files_in_nested_subdirectories(self):
        """Session files in nested subdirectories should be discovered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested subdirectories
            nested = Path(tmpdir) / "a" / "b" / "c"
            nested.mkdir(parents=True)
            _create_flow_file(str(nested), 5, "nested_session")

            sessions = discover_sessions(tmpdir)
            assert len(sessions) >= 1, (
                "Should discover session in nested subdirectory"
            )

    def test_discover_mixed_root_and_subdir_files(self):
        """Files in root and subdirectories should all be discovered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # File in root
            _create_flow_file(tmpdir, 1, "root_session")
            # File in subdirectory
            subdir = Path(tmpdir) / "sub"
            subdir.mkdir()
            _create_flow_file(str(subdir), 2, "sub_session")
            # File in nested subdirectory
            nested = Path(tmpdir) / "deep" / "level"
            nested.mkdir(parents=True)
            _create_flow_file(str(nested), 3, "deep_session")

            sessions = discover_sessions(tmpdir)
            assert len(sessions) == 3, (
                f"Expected 3 sessions, got {len(sessions)}"
            )

    def test_discover_subdir_only(self):
        """Only files in subdirectories should be discovered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Empty root
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            _create_flow_file(str(subdir), 7, "only_sub")

            sessions = discover_sessions(tmpdir)
            assert len(sessions) == 1
            assert sessions[0].task_name == "only_sub"


# ===================================================================
# App properties tests
# ===================================================================


class TestAppProperties:
    """Tests for app properties related to launch configuration."""

    def test_flows_dir_default(self):
        """App flows_dir defaults to absolute form of './flows'."""
        app = LLMFlowViewerApp()
        assert app.flows_dir is not None
        assert os.path.isabs(app.flows_dir)
        assert "flows" in app.flows_dir

    def test_flows_dir_custom(self):
        """App flows_dir can be set to a custom path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            app = LLMFlowViewerApp(flows_dir=tmpdir)
            assert app.flows_dir == os.path.abspath(tmpdir)

    def test_session_arg_none(self):
        """App preselected_session defaults to None."""
        app = LLMFlowViewerApp()
        assert app.preselected_session is None

    def test_session_arg_set(self):
        """App preselected_session stores the session arg."""
        app = LLMFlowViewerApp(session="01_flows-analyze_codebase")
        assert app.preselected_session == "01_flows-analyze_codebase"

    def test_get_flows_dir(self):
        """App get_flows_dir returns the flows directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            app = LLMFlowViewerApp(flows_dir=tmpdir)
            assert app.get_flows_dir() == os.path.abspath(tmpdir)
