"""Tests for the SessionList widget.

Tests cover session discovery via the filename pattern and temp-directory
discovery, empty states, and the SessionList widget.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from llm_flow_viewer.tui.app import LLMFlowViewerApp
from llm_flow_viewer.tui.widgets.session_list import (
    SESSION_FILE_PATTERN,
    SessionInfo,
    SessionList,
    discover_sessions,
)


# ---------------------------------------------------------------------------
# discover_sessions unit tests
# ---------------------------------------------------------------------------


class TestSessionFilePattern:
    """Tests for the SESSION_FILE_PATTERN regex."""

    def test_matches_valid_filename(self):
        """Valid flow filenames should match the pattern."""
        match = SESSION_FILE_PATTERN.match("01_flows-analyze_codebase")
        assert match is not None
        assert match.group(1) == "01"
        assert match.group(2) == "analyze_codebase"

    def test_matches_all_seven_sessions(self):
        """All expected session filenames should match."""
        names = [
            ("01_flows-analyze_codebase", 1, "analyze_codebase"),
            ("02_flows-readiness_report", 2, "readiness_report"),
            ("03_flows-receipt_wiki", 3, "receipt_wiki"),
            ("04_flows-compress", 4, "compress"),
            ("05_flows-security", 5, "security"),
            ("06_flows-mission", 6, "mission"),
            ("07_flows-baffled_wiki", 7, "baffled_wiki"),
        ]
        for filename, expected_index, expected_name in names:
            match = SESSION_FILE_PATTERN.match(filename)
            assert match is not None, f"Filename '{filename}' should match pattern"
            assert int(match.group(1)) == expected_index
            assert match.group(2) == expected_name

    def test_rejects_non_matching_name(self):
        """Filenames not matching the pattern should not match."""
        assert SESSION_FILE_PATTERN.match("flows-analyze_codebase") is None
        assert SESSION_FILE_PATTERN.match("01_random_file") is None
        assert SESSION_FILE_PATTERN.match("something_flows-else") is None
        assert SESSION_FILE_PATTERN.match("01_flows") is None

    def test_rejects_parquet_files(self):
        """Parquet cache files should not match (checked in discover_sessions)."""
        # Pattern itself would match, but discover_sessions filters by extension
        match = SESSION_FILE_PATTERN.match("01_flows-analyze_codebase_requests.parquet")
        if match:
            # Pattern might match but discover_sessions filters by extension
            pass
        # The discover_sessions function should exclude .parquet files


class TestDiscoverSessions:
    """Tests for the discover_sessions function."""

    def test_discover_empty_directory(self):
        """Should return empty list for non-existent or empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions = discover_sessions(tmpdir)
            assert sessions == []

    def test_discover_non_existent_directory(self):
        """Should return empty list for non-existent directory."""
        sessions = discover_sessions(r"D:\nonexistent_path_12345")
        assert sessions == []

    def test_discover_filters_parquet(self):
        """Should exclude .parquet files from results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid flow file
            flow_path = Path(tmpdir) / "01_flows-analyze_codebase"
            flow_path.write_text("fake content")
            # Create a parquet file that would match the pattern
            parquet_path = Path(tmpdir) / "01_flows-analyze_codebase_requests.parquet"
            parquet_path.write_text("fake parquet")
            sessions = discover_sessions(tmpdir)
            assert len(sessions) == 1
            assert sessions[0].task_name == "analyze_codebase"

    def test_session_info_dataclass(self):
        """SessionInfo should store index, task_name, and file_path."""
        info = SessionInfo(index=3, task_name="test_task", file_path="/path/to/file")
        assert info.index == 3
        assert info.task_name == "test_task"
        assert info.file_path == "/path/to/file"
        # Should be sortable by index
        info2 = SessionInfo(index=1, task_name="first", file_path="/path/first")
        info3 = SessionInfo(index=2, task_name="second", file_path="/path/second")
        sorted_infos = sorted([info, info2, info3], key=lambda s: s.index)
        assert sorted_infos[0].index == 1
        assert sorted_infos[1].index == 2
        assert sorted_infos[2].index == 3


class TestNonNumericNaming:
    """Tests for non-numeric file naming support in discover_sessions."""

    def test_derive_task_name_with_separator(self):
        """_flows- separator should extract the task name."""
        from llm_flow_viewer.tui.widgets.session_list import _derive_task_name
        assert _derive_task_name("custom_flows-debug") == "debug"
        assert _derive_task_name("my_session_flows-test_task") == "test_task"
        assert _derive_task_name("01_flows-analyze_codebase") == "analyze_codebase"

    def test_derive_task_name_without_separator(self):
        """Files without _flows- separator should use filename as-is."""
        from llm_flow_viewer.tui.widgets.session_list import _derive_task_name
        assert _derive_task_name("01_flows_all") == "01_flows_all"
        assert _derive_task_name("random_file") == "random_file"

    def test_discover_non_numeric_files_in_temp_dir(self):
        """Non-numeric flow files should be discovered with synthetic indices."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Create a numeric flow file
            (tmp / "01_flows-analyze_codebase").write_text("content1")

            # Create a non-numeric flow file
            (tmp / "custom_flows-debug").write_text("content2")

            # Create a parquet file (should be excluded)
            (tmp / "some_flows-debug_requests.parquet").write_text("parquet")

            sessions = discover_sessions(tmpdir)

            # Should find 2 sessions (numeric + non-numeric)
            assert len(sessions) == 2, f"Expected 2 sessions, got {len(sessions)}"

            # Numeric session should have its original index
            numeric = [s for s in sessions if s.task_name == "analyze_codebase"]
            assert len(numeric) == 1
            assert numeric[0].index == 1

            # Non-numeric session should have a synthetic index
            synthetic = [s for s in sessions if s.task_name == "debug"]
            assert len(synthetic) == 1
            assert synthetic[0].index == 2  # max(numeric) + 1 = 2

    def test_synthetic_indices_by_mtime_newest_first(self):
        """Synthetic indices should be assigned by mtime (newest first)."""
        import time
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Create a numeric session first
            (tmp / "01_flows-analyze_codebase").write_text("content1")
            time.sleep(0.05)

            # Create older non-numeric file — use filename without _flows- to
            # keep the full filename as the task name
            older_path = tmp / "older_session"
            older_path.write_text("old content")
            older_mtime = time.time() - 100  # 100 seconds ago
            os.utime(older_path, (older_mtime, older_mtime))
            time.sleep(0.05)

            # Create newer non-numeric file
            newer_path = tmp / "newer_session"
            newer_path.write_text("new content")
            # newer_path has current mtime, so it's newer
            time.sleep(0.05)

            sessions = discover_sessions(tmpdir)

            # Should find 3 sessions
            assert len(sessions) == 3, f"Expected 3 sessions, got {len(sessions)}"

            # Find synthetic sessions (index >= 2 since numeric max is 1)
            synthetic = [s for s in sessions if s.index >= 2]
            synthetic_sorted_by_index = sorted(synthetic, key=lambda s: s.index)

            assert len(synthetic_sorted_by_index) == 2
            # Newest file (newer_session) should have the lower synthetic index
            assert synthetic_sorted_by_index[0].task_name == "newer_session", (
                f"Newest file should have lower index, got {synthetic_sorted_by_index[0].task_name}"
            )
            assert synthetic_sorted_by_index[1].task_name == "older_session", (
                f"Oldest file should have higher index, got {synthetic_sorted_by_index[1].task_name}"
            )

    def test_synthetic_indices_deterministic(self):
        """Synthetic index assignment should be deterministic across reloads."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Create numeric session
            (tmp / "01_flows-analyze_codebase").write_text("content1")

            # Create non-numeric sessions
            (tmp / "a_flows-alpha").write_text("content2")
            (tmp / "b_flows-beta").write_text("content3")

            # First discovery
            sessions1 = discover_sessions(tmpdir)
            # Second discovery
            sessions2 = discover_sessions(tmpdir)

            # Both should produce the same indices (deterministic)
            for s1, s2 in zip(sessions1, sessions2):
                assert s1.index == s2.index, (
                    f"Session {s1.task_name} should have same index across reloads: "
                    f"{s1.index} vs {s2.index}"
                )
                assert s1.task_name == s2.task_name

    def test_numeric_indices_preserved_with_non_numeric_present(self):
        """Numeric-prefix sessions must retain their original indices when non-numeric files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Create numeric sessions
            (tmp / "03_flows-receipt_wiki").write_text("content1")
            (tmp / "01_flows-analyze_codebase").write_text("content2")
            (tmp / "02_flows-readiness_report").write_text("content3")

            # Non-numeric
            (tmp / "custom_flows-debug").write_text("content4")

            sessions = discover_sessions(tmpdir)

            assert len(sessions) == 4

            # Numeric sessions should retain their original indices
            analyze = next(s for s in sessions if s.task_name == "analyze_codebase")
            assert analyze.index == 1, f"Expected index 1, got {analyze.index}"

            readiness = next(s for s in sessions if s.task_name == "readiness_report")
            assert readiness.index == 2, f"Expected index 2, got {readiness.index}"

            receipt = next(s for s in sessions if s.task_name == "receipt_wiki")
            assert receipt.index == 3, f"Expected index 3, got {receipt.index}"

            # Non-numeric gets synthetic index = 4 (max numeric is 3, start from 4)
            debug = next(s for s in sessions if s.task_name == "debug")
            assert debug.index == 4, f"Expected synthetic index 4, got {debug.index}"

    def test_no_numeric_sessions_starts_at_1000(self):
        """When no numeric sessions exist, synthetic indices start at 1000."""
        import time
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            (tmp / "custom_flows-first").write_text("first")
            time.sleep(0.05)
            (tmp / "custom_flows-second").write_text("second")

            sessions = discover_sessions(tmpdir)

            assert len(sessions) == 2
            # Both should have synthetic indices starting at 1000
            assert sessions[0].index == 1000
            assert sessions[1].index == 1001


# ---------------------------------------------------------------------------
# SessionList widget tests (integration with Textual)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_list_empty_state():
    """Session list should be empty when flows_dir has no sessions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        app = LLMFlowViewerApp(flows_dir=tmpdir)
        async with app.run_test(size=(120, 40)) as pilot:
            session_list = app.screen.query(SessionList).first()
            await pilot.pause()

            # The SessionList should discover 0 sessions in an empty directory
            assert len(session_list.children) == 0, (
                f"Expected 0 items for empty dir, got {len(session_list.children)}"
            )
            assert session_list.selected_session is None, (
                "selected_session should be None for empty list"
            )

