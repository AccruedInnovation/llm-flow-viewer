"""Tests for CLI entry point.

Covers VAL-CLI-001, VAL-CLI-002, VAL-CLI-009, and VAL-CLI-010.

The CLI entry must:
- Accept zero arguments and default to the flows directory.
- Accept --flows-dir with a valid path.
- Support ``python -m llm_flow_viewer`` module invocation.
- Be installable as a console script entry point ``llm-flow-viewer``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from unittest.mock import patch

import pytest

from llm_flow_viewer.tui.app import LLMFlowViewerApp, build_arg_parser, main


# ─── VAL-CLI-001: No arguments ──────────────────────────────────────────────

class TestNoArgumentLaunch:
    """VAL-CLI-001: Run with no arguments — TUI launches with default flows directory."""

    def test_arg_parser_accepts_zero_arguments(self):
        """CLI parser accepts zero arguments (all are optional)."""
        parser = build_arg_parser()
        args = parser.parse_args([])
        assert args.flows_dir == "./flows", (
            f"Default flows_dir should be './flows', got: {args.flows_dir}"
        )

    def test_default_flows_dir_is_relative_path(self):
        """The default flows_dir is a relative path resolved against CWD."""
        parser = build_arg_parser()
        args = parser.parse_args([])
        # Default is a relative path; it's resolved to absolute at launch time
        assert not os.path.isabs(args.flows_dir), (
            "Default flows_dir should be a relative path, "
            f"got absolute: {args.flows_dir}"
        )

    def test_llmflowviewer_app_accepts_no_flows_dir(self):
        """LLMFlowViewerApp can be instantiated without flows_dir kwarg,
        defaulting to './flows'."""
        app = LLMFlowViewerApp()
        assert app.flows_dir.endswith("flows"), (
            f"Default flows_dir should end with 'flows', got: {app.flows_dir}"
        )
        assert os.path.isabs(app.flows_dir), (
            f"App should resolve flows_dir to absolute path, got: {app.flows_dir}"
        )

    def test_llmflowviewer_app_default_dir_is_absolute(self):
        """Default flows_dir is resolved to an absolute path."""
        app = LLMFlowViewerApp()
        assert os.path.isabs(app.flows_dir), (
            f"flows_dir should be absolute, got: {app.flows_dir}"
        )
        assert "flows" in app.flows_dir, (
            f"flows_dir should contain 'flows', got: {app.flows_dir}"
        )


# ─── VAL-CLI-002: --flows-dir ───────────────────────────────────────────────

class TestFlowsDirArgument:
    """VAL-CLI-002: ``--flows-dir`` pointing to valid directory — loads flows."""

    def test_flows_dir_accepted_by_parser(self):
        """CLI parser accepts --flows-dir argument."""
        test_path = "/some/valid/path"
        parser = build_arg_parser()
        args = parser.parse_args(["--flows-dir", test_path])
        assert args.flows_dir == test_path, (
            f"flows_dir should be '{test_path}', got: '{args.flows_dir}'"
        )

    def test_flows_dir_accepted_as_single_arg(self):
        """CLI parser accepts --flows-dir=PATH syntax."""
        test_path = "/some/valid/path"
        parser = build_arg_parser()
        args = parser.parse_args([f"--flows-dir={test_path}"])
        assert args.flows_dir == test_path, (
            f"flows_dir should be '{test_path}', got: '{args.flows_dir}'"
        )

    def test_flows_dir_passed_to_app(self):
        """The provided flows_dir is passed to the TUI app."""
        with tempfile.TemporaryDirectory() as tmpdir:
            app = LLMFlowViewerApp(flows_dir=tmpdir)
            assert app.flows_dir == os.path.abspath(tmpdir), (
                f"App flows_dir should be '{os.path.abspath(tmpdir)}', "
                f"got: '{app.flows_dir}'"
            )

    def test_flows_dir_resolved_to_absolute(self):
        """Relative --flows-dir path is resolved to absolute."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Change to temp dir and use a relative path
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                rel_path = "my_flows"
                os.makedirs(rel_path, exist_ok=True)
                app = LLMFlowViewerApp(flows_dir=rel_path)
                expected_abs = os.path.abspath(os.path.join(tmpdir, rel_path))
                assert app.flows_dir == expected_abs, (
                    f"flows_dir should resolve to '{expected_abs}', "
                    f"got: '{app.flows_dir}'"
                )
            finally:
                os.chdir(original_cwd)

    def test_main_accepts_flows_dir(self):
        """main() parses --flows-dir and passes it to the app."""
        # We can't easily run main() in tests without launching the full TUI,
        # but we can verify the arg parser correctly captures the value.
        parser = build_arg_parser()
        with tempfile.TemporaryDirectory() as tmpdir:
            args = parser.parse_args(["--flows-dir", tmpdir])
            assert args.flows_dir == tmpdir


# ─── VAL-CLI-009: python -m module invocation ───────────────────────────────

class TestModuleInvocation:
    """VAL-CLI-009: ``python -m llm_flow_viewer`` entry point works."""

    def test_main_py_exists(self):
        """__main__.py exists for python -m invocation."""
        import importlib
        import llm_flow_viewer
        # Just verify the __main__.py module file exists
        main_spec = importlib.util.find_spec("llm_flow_viewer.__main__")
        assert main_spec is not None, (
            "llm_flow_viewer.__main__ module spec should exist"
        )
        assert main_spec.origin is not None, (
            "llm_flow_viewer.__main__ should have a file origin"
        )
        assert main_spec.origin.endswith("__main__.py"), (
            f"Expected __main__.py, got: {main_spec.origin}"
        )

    def test_main_py_imports_main_function(self):
        """__main__.py imports and calls main() from app module."""
        import ast
        import os

        # Find the __main__.py file
        main_py_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "src",
            "llm_flow_viewer",
            "__main__.py",
        )
        assert os.path.exists(main_py_path), (
            f"__main__.py not found at {main_py_path}"
        )

        with open(main_py_path) as f:
            tree = ast.parse(f.read())

        # Check that main() is called
        has_main_call = any(
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "main"
            for node in tree.body
        )
        assert has_main_call, (
            "__main__.py should call main()"
        )

        # Check that main is imported from app
        has_import = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "llm_flow_viewer.tui.app"
            and any(alias.name == "main" for alias in node.names)
            for node in tree.body
        )
        assert has_import, (
            "__main__.py should import main from llm_flow_viewer.tui.app"
        )

    def test_module_invocation_help_output(self):
        """``python -m llm_flow_viewer --help`` shows correct output."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "llm_flow_viewer",
                "--help",
            ],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        )
        assert result.returncode == 0, (
            f"python -m llm_flow_viewer --help exited with code "
            f"{result.returncode}"
        )
        assert "llm-flow-viewer" in result.stdout, (
            f"Help output should contain 'llm-flow-viewer', "
            f"got: {result.stdout[:200]}"
        )
        assert "--flows-dir" in result.stdout, (
            f"Help output should contain '--flows-dir', "
            f"got: {result.stdout[:200]}"
        )

    def test_module_invocation_version(self):
        """``python -m llm_flow_viewer --version`` shows version."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "llm_flow_viewer",
                "--version",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"python -m llm_flow_viewer --version exited with code "
            f"{result.returncode}"
        )
        version_output = result.stdout.strip()
        assert version_output.startswith("llm-flow-viewer"), (
            f"Version should start with 'llm-flow-viewer', "
            f"got: '{version_output}'"
        )
        assert "0.1.0" in version_output, (
            f"Version should contain '0.1.0', got: '{version_output}'"
        )

    def test_module_invocation_same_as_console_script_help(self):
        """``python -m llm_flow_viewer --help`` output matches console script."""
        module_result = subprocess.run(
            [sys.executable, "-m", "llm_flow_viewer", "--help"],
            capture_output=True,
            text=True,
        )
        script_result = subprocess.run(
            [sys.executable, "-m", "llm_flow_viewer", "--help"],
            capture_output=True,
            text=True,
        )
        assert module_result.stdout == script_result.stdout, (
            "Module invocation and console script should produce identical "
            "help output"
        )


# ─── VAL-CLI-010: Console script entry point ────────────────────────────────

class TestConsoleScript:
    """VAL-CLI-010: ``llm-flow-viewer`` console script entry point works."""

    def test_entry_point_defined_in_pyproject(self):
        """The console script entry point is defined in pyproject.toml."""
        import tomllib

        pyproject_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "pyproject.toml",
        )
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        scripts = data.get("project", {}).get("scripts", {})
        assert "llm-flow-viewer" in scripts, (
            "pyproject.toml should define 'llm-flow-viewer' under "
            "[project.scripts]"
        )
        assert scripts["llm-flow-viewer"] == "llm_flow_viewer.tui.app:main", (
            f"llm-flow-viewer should point to "
            f"'llm_flow_viewer.tui.app:main', "
            f"got: '{scripts['llm-flow-viewer']}'"
        )

    def test_entry_point_resolves_correctly(self):
        """The entry point function can be imported."""
        from llm_flow_viewer.tui.app import main as entry_main
        assert callable(entry_main), "Entry point 'main' should be callable"

    def test_console_script_help_output(self):
        """The console script --help produces correct output."""
        # Try to find the console script executable
        venv_scripts = os.path.join(
            os.path.dirname(sys.executable),
        )
        script_path = os.path.join(venv_scripts, "llm-flow-viewer.exe")
        alt_script_path = os.path.join(venv_scripts, "llm-flow-viewer")

        script_exe = None
        if os.path.exists(script_path):
            script_exe = script_path
        elif os.path.exists(alt_script_path):
            script_exe = alt_script_path

        if script_exe:
            result = subprocess.run(
                [script_exe, "--help"],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, (
                f"Console script --help exited with code {result.returncode}"
            )
            assert "--flows-dir" in result.stdout, (
                f"Help output should contain '--flows-dir', "
                f"got: {result.stdout[:200]}"
            )
        else:
            # If the script doesn't exist as a standalone exe, verify
            # that pip show lists it
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", "-f", "llm-flow-viewer"],
                capture_output=True,
                text=True,
            )
            assert "llm-flow-viewer" in result.stdout, (
                "pip show -f llm-flow-viewer should list "
                "llm-flow-viewer entry point"
            )

    def test_script_installed_in_venv_bin(self):
        """The console script is installed in the virtual environment's Scripts directory."""
        # Check both system Python dir and user Scripts dir (for pip --user installs)
        candidates = [
            os.path.join(os.path.dirname(sys.executable)),
            os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "Python",
                         f"Python{sys.version_info.major}{sys.version_info.minor}", "Scripts"),
        ]
        for scripts_dir in candidates:
            script_path = os.path.join(scripts_dir, "llm-flow-viewer.exe")
            alt_script_path = os.path.join(scripts_dir, "llm-flow-viewer")
            if os.path.exists(script_path) or os.path.exists(alt_script_path):
                return
        assert False, (
            f"Console script not found. Checked: {candidates}"
        )


# ─── Integration: app instantiation with flows_dir ──────────────────────────

class TestAppInstantiation:
    """Verify the LLMFlowViewerApp can be instantiated with various flows_dir values."""

    def test_app_instantiation_with_no_args(self):
        """App can be created with default flows_dir (no argument)."""
        app = LLMFlowViewerApp()
        assert app.flows_dir is not None
        assert len(app.flows_dir) > 0

    def test_app_instantiation_with_valid_flows_dir(self):
        """App can be created with a specific flows_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            app = LLMFlowViewerApp(flows_dir=tmpdir)
            assert os.path.isabs(app.flows_dir)
            assert os.path.normpath(app.flows_dir) == os.path.normpath(
                os.path.abspath(tmpdir)
            )

    def test_app_flows_dir_is_absolute(self):
        """flows_dir property returns an absolute path."""
        app = LLMFlowViewerApp(flows_dir="relative/path")
        assert os.path.isabs(app.flows_dir), (
            f"flows_dir should be absolute, got: {app.flows_dir}"
        )


# ─── VAL-CLI-001: Default flows dir behavior ────────────────────────────────

class TestDefaultFlowsDirBehavior:
    """Additional tests for the default flows directory."""

    def test_main_validates_flows_dir_exists(self, monkeypatch):
        """main() exits with error when default flows dir does not exist."""
        from llm_flow_viewer.tui import app as app_module
        import argparse

        # Temporarily change to a directory without ./flows
        with tempfile.TemporaryDirectory() as tmpdir:
            original_abspath = os.path.abspath

            def mock_abspath(path):
                if path == "./flows":
                    return os.path.join(tmpdir, "flows")
                return original_abspath(path)

            monkeypatch.setattr(os.path, "abspath", mock_abspath)
            monkeypatch.setattr(app_module.sys, "argv", ["llm-flow-viewer"])

            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1, (
                "Should exit with code 1 when flows dir does not exist"
            )

    def test_build_arg_parser_help_includes_flows_dir(self):
        """--help output mentions --flows-dir."""
        parser = build_arg_parser()
        help_text = parser.format_help()
        assert "--flows-dir" in help_text, (
            "Help should mention --flows-dir"
        )
        assert "FLOWS_DIR" in help_text, (
            "Help should mention FLOWS_DIR metavar"
        )

    def test_build_arg_parser_help_includes_version(self):
        """--help output mentions --version."""
        parser = build_arg_parser()
        help_text = parser.format_help()
        assert "--version" in help_text, (
            "Help should mention --version"
        )

    def test_build_arg_parser_help_includes_help(self):
        """--help output mentions -h/--help."""
        parser = build_arg_parser()
        help_text = parser.format_help()
        assert "-h" in help_text, "Help should mention -h"
        assert "--help" in help_text, "Help should mention --help"


# ─── version check ──────────────────────────────────────────────────────────

class TestVersion:
    """Verify --version flag works correctly."""

    def test_version_flag_with_args(self, monkeypatch):
        """--version flag works and exits with code 0."""
        from llm_flow_viewer import __version__ as pkg_version
        from llm_flow_viewer.tui import app as app_module

        monkeypatch.setattr(
            app_module.sys, "argv", ["llm-flow-viewer", "--version"]
        )
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0, (
            "--version should exit with code 0"
        )

    def test_version_matches_pyproject(self):
        """Version number matches pyproject.toml."""
        import tomllib

        pyproject_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "pyproject.toml",
        )
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        pyproject_version = data["project"]["version"]

        from llm_flow_viewer import __version__
        assert __version__ == pyproject_version, (
            f"Package version '{__version__}' should match "
            f"pyproject.toml version '{pyproject_version}'"
        )


# ─── VAL-CLI-003: --flows-dir to invalid/missing directory ──────────────────

class TestFlowsDirInvalid:
    """VAL-CLI-003: ``--flows-dir`` pointing to invalid/missing directory — error and exit."""

    def test_nonexistent_flows_dir_exits_with_error(self):
        """CLI exits with non-zero when --flows-dir points to nonexistent path."""
        from llm_flow_viewer.tui import app as app_module

        nonexistent = os.path.join(tempfile.gettempdir(), "__nonexistent_flows__")

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            app_module.sys, "argv",
            ["llm-flow-viewer", "--flows-dir", nonexistent],
        )
        try:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1, (
                f"Should exit with code 1, got: {exc_info.value.code}"
            )
        finally:
            monkeypatch.undo()

    def test_nonexistent_flows_dir_error_message(self, capsys):
        """CLI prints a clear error message for nonexistent --flows-dir."""
        from llm_flow_viewer.tui import app as app_module

        nonexistent = os.path.join(tempfile.gettempdir(), "__nonexistent_flows__")

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            app_module.sys, "argv",
            ["llm-flow-viewer", "--flows-dir", nonexistent],
        )
        try:
            with pytest.raises(SystemExit):
                main()
            captured = capsys.readouterr()
            assert captured.err, "Error output should not be empty"
            assert "Error" in captured.err, (
                f"Error message should contain 'Error', got: {captured.err}"
            )
            assert nonexistent in captured.err, (
                f"Error message should include the path, got: {captured.err}"
            )
            # Must not contain raw stack traces or traceback indicators
            assert "Traceback" not in captured.err, (
                f"Error output should not contain stack traces, got: {captured.err}"
            )
        finally:
            monkeypatch.undo()

    def test_nonexistent_flows_dir_stdout_empty(self, capsys):
        """Error for nonexistent --flows-dir goes to stderr, not stdout."""
        from llm_flow_viewer.tui import app as app_module

        nonexistent = os.path.join(tempfile.gettempdir(), "__nonexistent_flows__")

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            app_module.sys, "argv",
            ["llm-flow-viewer", "--flows-dir", nonexistent],
        )
        try:
            with pytest.raises(SystemExit):
                main()
            captured = capsys.readouterr()
            assert captured.out == "", (
                "No output should go to stdout for directory-not-found error"
            )
        finally:
            monkeypatch.undo()

    def test_nonexistent_flows_dir_resolved_to_absolute_in_error(self, capsys):
        """Error message includes the resolved absolute path, not the relative one."""
        from llm_flow_viewer.tui import app as app_module

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setattr(
                app_module.sys, "argv",
                ["llm-flow-viewer", "--flows-dir", "nonexistent_rel"],
            )
            # Also patch os.getcwd so the relative path resolves inside tmpdir
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                with pytest.raises(SystemExit):
                    main()
                captured = capsys.readouterr()
                expected_abs = os.path.abspath(os.path.join(tmpdir, "nonexistent_rel"))
                assert expected_abs in captured.err, (
                    f"Error should contain resolved absolute path '{expected_abs}', "
                    f"got: {captured.err}"
                )
            finally:
                os.chdir(original_cwd)
                monkeypatch.undo()

    def test_nonexistent_flows_dir_exits_without_launching_app(self):
        """TUI app is not launched when flows dir does not exist."""
        from llm_flow_viewer.tui import app as app_module

        nonexistent = os.path.join(tempfile.gettempdir(), "__nonexistent_flows__")

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            app_module.sys, "argv",
            ["llm-flow-viewer", "--flows-dir", nonexistent],
        )
        # We can verify the app is not launched by checking that main()
        # exits before calling LLMFlowViewerApp
        original_app = app_module.LLMFlowViewerApp

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

    def test_nonexistent_flows_dir_no_traceback(self, capsys):
        """No Python traceback appears in error output for invalid --flows-dir."""
        from llm_flow_viewer.tui import app as app_module

        nonexistent = os.path.join(tempfile.gettempdir(), "__nonexistent_flows__")

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            app_module.sys, "argv",
            ["llm-flow-viewer", "--flows-dir", nonexistent],
        )
        try:
            with pytest.raises(SystemExit):
                main()
            captured = capsys.readouterr()
            assert "Traceback" not in captured.err, (
                "Error output must not contain Python traceback"
            )
            assert "File \"" not in captured.err, (
                "Error output must not contain file paths from tracebacks"
            )
        finally:
            monkeypatch.undo()


# ─── VAL-CLI-006: --session for non-existent session ────────────────────────

class TestSessionInvalid:
    """VAL-CLI-006: ``--session`` for non-existent session — error and exit."""

    def test_nonexistent_session_exits_with_error(self):
        """CLI exits with non-zero when --session refers to unknown session."""
        from llm_flow_viewer.tui import app as app_module

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setattr(
                app_module.sys, "argv",
                ["llm-flow-viewer", "--flows-dir", tmpdir, "--session", "nonexistent_session"],
            )
            try:
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1, (
                    f"Should exit with code 1, got: {exc_info.value.code}"
                )
            finally:
                monkeypatch.undo()

    def test_nonexistent_session_error_message(self, capsys):
        """CLI prints a clear error message for unknown --session."""
        from llm_flow_viewer.tui import app as app_module

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setattr(
                app_module.sys, "argv",
                ["llm-flow-viewer", "--flows-dir", tmpdir, "--session", "nonexistent_session"],
            )
            try:
                with pytest.raises(SystemExit):
                    main()
                captured = capsys.readouterr()
                assert captured.err, "Error output should not be empty"
                assert "Error" in captured.err, (
                    f"Error message should contain 'Error', got: {captured.err}"
                )
                assert "nonexistent_session" in captured.err, (
                    f"Error message should include the session name, "
                    f"got: {captured.err}"
                )
                assert "Available sessions" in captured.err or "available" in captured.err.lower(), (
                    f"Error message should mention available sessions, "
                    f"got: {captured.err}"
                )
                assert "Traceback" not in captured.err, (
                    "Error output should not contain stack traces"
                )
            finally:
                monkeypatch.undo()

    def test_nonexistent_session_stdout_empty(self, capsys):
        """Error for unknown --session goes to stderr, not stdout."""
        from llm_flow_viewer.tui import app as app_module

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setattr(
                app_module.sys, "argv",
                ["llm-flow-viewer", "--flows-dir", tmpdir, "--session", "nonexistent"],
            )
            try:
                with pytest.raises(SystemExit):
                    main()
                captured = capsys.readouterr()
                assert captured.out == "", (
                    "No output should go to stdout for session-not-found error"
                )
            finally:
                monkeypatch.undo()

    def test_nonexistent_session_no_traceback(self, capsys):
        """No Python traceback appears in error output for invalid --session."""
        from llm_flow_viewer.tui import app as app_module

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setattr(
                app_module.sys, "argv",
                ["llm-flow-viewer", "--flows-dir", tmpdir, "--session", "does_not_exist"],
            )
            try:
                with pytest.raises(SystemExit):
                    main()
                captured = capsys.readouterr()
                assert "Traceback" not in captured.err, (
                    "Error output must not contain Python traceback"
                )
                assert "File \"" not in captured.err, (
                    "Error output must not contain file paths from tracebacks"
                )
            finally:
                monkeypatch.undo()

    def test_nonexistent_session_with_known_sessions_shows_available(self, capsys):
        """Error for unknown session lists available sessions when sessions exist."""
        from llm_flow_viewer.tui import app as app_module

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some fake session files in the temp dir
            for name in ["01_flows-analyze_codebase", "02_flows-readiness_report"]:
                file_path = os.path.join(tmpdir, name)
                with open(file_path, "w") as f:
                    f.write("dummy content")

            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setattr(
                app_module.sys, "argv",
                ["llm-flow-viewer", "--flows-dir", tmpdir, "--session", "nonexistent"],
            )
            try:
                with pytest.raises(SystemExit):
                    main()
                captured = capsys.readouterr()
                assert "Available sessions" in captured.err, (
                    f"Error should list available sessions, got: {captured.err}"
                )
                assert "01_flows-analyze_codebase" in captured.err, (
                    f"Error should include first known session, got: {captured.err}"
                )
                assert "02_flows-readiness_report" in captured.err, (
                    f"Error should include second known session, got: {captured.err}"
                )
            finally:
                monkeypatch.undo()

    def test_nonexistent_session_exits_without_launching_app(self):
        """TUI app is not launched when --session is invalid."""
        from llm_flow_viewer.tui import app as app_module

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setattr(
                app_module.sys, "argv",
                ["llm-flow-viewer", "--flows-dir", tmpdir, "--session", "invalid"],
            )
            original_app = app_module.LLMFlowViewerApp

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


# ─── VAL-CLI-012: Invalid argument combination ──────────────────────────────

class TestInvalidArgumentCombination:
    """VAL-CLI-012: Invalid argument combination — both errors reported."""

    def test_both_errors_reported(self, capsys):
        """Both invalid --flows-dir and --session errors are reported before exit."""
        from llm_flow_viewer.tui import app as app_module

        nonexistent_dir = os.path.join(tempfile.gettempdir(), "__nonexistent_dir__")
        nonexistent_sess = "bogus_session"

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            app_module.sys, "argv",
            ["llm-flow-viewer", "--flows-dir", nonexistent_dir, "--session", nonexistent_sess],
        )
        try:
            with pytest.raises(SystemExit) as exc_info:
                main()
            captured = capsys.readouterr()
            # Should report the flows-dir error
            assert "flows directory not found" in captured.err or nonexistent_dir in captured.err, (
                f"Error should mention flows-dir problem, got: {captured.err}"
            )
            # Exit code should be non-zero
            assert exc_info.value.code == 1, (
                f"Should exit with code 1, got: {exc_info.value.code}"
            )
            # No traceback
            assert "Traceback" not in captured.err, (
                "Error output must not contain Python traceback"
            )
        finally:
            monkeypatch.undo()

    def test_both_errors_no_traceback(self, capsys):
        """No stack trace appears when both --flows-dir and --session are invalid."""
        from llm_flow_viewer.tui import app as app_module

        nonexistent_dir = os.path.join(tempfile.gettempdir(), "__nonexistent_dir__")

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            app_module.sys, "argv",
            ["llm-flow-viewer", "--flows-dir", nonexistent_dir, "--session", "bad"],
        )
        try:
            with pytest.raises(SystemExit):
                main()
            captured = capsys.readouterr()
            assert "Traceback" not in captured.err, (
                "Error output must not contain Python traceback"
            )
            assert "File \"" not in captured.err, (
                "Error output must not contain file paths from tracebacks"
            )
        finally:
            monkeypatch.undo()

    def test_both_errors_nonzero_exit(self):
        """Exit code is non-zero when both --flows-dir and --session are invalid."""
        from llm_flow_viewer.tui import app as app_module

        nonexistent_dir = os.path.join(tempfile.gettempdir(), "__nonexistent_dir__")

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            app_module.sys, "argv",
            ["llm-flow-viewer", "--flows-dir", nonexistent_dir, "--session", "bad"],
        )
        try:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code != 0, (
                "Should exit with non-zero code for invalid arguments"
            )
        finally:
            monkeypatch.undo()

    def test_both_errors_stderr_only(self, capsys):
        """Combined errors go to stderr, not stdout."""
        from llm_flow_viewer.tui import app as app_module

        nonexistent_dir = os.path.join(tempfile.gettempdir(), "__nonexistent_dir__")

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            app_module.sys, "argv",
            ["llm-flow-viewer", "--flows-dir", nonexistent_dir, "--session", "bad"],
        )
        try:
            with pytest.raises(SystemExit):
                main()
            captured = capsys.readouterr()
            assert captured.out == "", (
                "No output should go to stdout for combined errors"
            )
        finally:
            monkeypatch.undo()

    def test_valid_flows_dir_with_invalid_session(self, capsys):
        """When --flows-dir is valid but --session is invalid, session error is reported."""
        from llm_flow_viewer.tui import app as app_module

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setattr(
                app_module.sys, "argv",
                ["llm-flow-viewer", "--flows-dir", tmpdir, "--session", "invalid"],
            )
            try:
                with pytest.raises(SystemExit) as exc_info:
                    main()
                captured = capsys.readouterr()
                assert "Error" in captured.err, (
                    f"Error should contain 'Error', got: {captured.err}"
                )
                assert "session" in captured.err.lower(), (
                    f"Error message should mention 'session', got: {captured.err}"
                )
                assert exc_info.value.code == 1, (
                    f"Should exit with code 1, got: {exc_info.value.code}"
                )
            finally:
                monkeypatch.undo()


# ─── --session argument parsing ─────────────────────────────────────────────

class TestSessionArgument:
    """Verify the ``--session`` argument parsing."""

    def test_session_accepted_by_parser(self):
        """CLI parser accepts --session argument."""
        parser = build_arg_parser()
        args = parser.parse_args(["--session", "01_flows-analyze_codebase"])
        assert args.session == "01_flows-analyze_codebase", (
            f"session should be '01_flows-analyze_codebase', "
            f"got: '{args.session}'"
        )

    def test_session_defaults_to_none(self):
        """CLI parser defaults --session to None."""
        parser = build_arg_parser()
        args = parser.parse_args([])
        assert args.session is None, (
            f"Default session should be None, got: '{args.session}'"
        )

    def test_help_mentions_session(self):
        """--help output mentions --session."""
        parser = build_arg_parser()
        help_text = parser.format_help()
        assert "--session" in help_text, (
            "Help should mention --session"
        )


# ─── Non-zero exit codes for all error cases ────────────────────────────────

class TestErrorExitCodes:
    """VAL-CLI-015: Exit codes — non-zero for all error types."""

    def test_nonexistent_flows_dir_exit_code_1(self):
        """Invalid --flows-dir exits with code 1."""
        from llm_flow_viewer.tui import app as app_module

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            app_module.sys, "argv",
            ["llm-flow-viewer", "--flows-dir", os.path.join("C:", os.sep, "nonexistent_path_test_12345")],
        )
        try:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1, (
                f"Invalid --flows-dir should exit with code 1, "
                f"got: {exc_info.value.code}"
            )
        finally:
            monkeypatch.undo()

    def test_nonexistent_session_exit_code_1(self):
        """Invalid --session exits with code 1."""
        from llm_flow_viewer.tui import app as app_module

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setattr(
                app_module.sys, "argv",
                ["llm-flow-viewer", "--flows-dir", tmpdir, "--session", "bad"],
            )
            try:
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1, (
                    f"Invalid --session should exit with code 1, "
                    f"got: {exc_info.value.code}"
                )
            finally:
                monkeypatch.undo()

    def test_combined_errors_exit_code_1(self):
        """Combined invalid args exit with code 1."""
        from llm_flow_viewer.tui import app as app_module

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            app_module.sys, "argv",
            ["llm-flow-viewer", "--flows-dir", "C:\\nonexistent\\dir", "--session", "bad"],
        )
        try:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code != 0, (
                "Combined invalid args should exit with non-zero code"
            )
        finally:
            monkeypatch.undo()


# ─── VAL-CLI-004: Empty flows directory ─────────────────────────────────────

class TestEmptyFlowsDir:
    """VAL-CLI-004: ``--flows-dir`` pointing to empty directory — TUI launches with empty state."""

    def test_empty_flows_dir_launches_app(self):
        """App is instantiated when --flows-dir points to an empty directory (no crash)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Directory exists but is empty
            app = LLMFlowViewerApp(flows_dir=tmpdir)
            assert os.path.isabs(app.flows_dir)
            assert os.path.isdir(app.flows_dir)

    def test_empty_flows_dir_no_error_from_main(self):
        """main() does not exit with error when --flows-dir points to an empty directory."""
        from llm_flow_viewer.tui import app as app_module

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setattr(
                app_module.sys, "argv",
                ["llm-flow-viewer", "--flows-dir", tmpdir],
            )
            # Override run() via subclass to prevent TUI launch
            original_app_class = app_module.LLMFlowViewerApp

            class NoRunApp(original_app_class):
                def run(self):
                    pass

            monkeypatch.setattr(app_module, "LLMFlowViewerApp", NoRunApp)
            try:
                # No SystemExit should be raised for valid directory
                main()
            except SystemExit as e:
                assert e.code == 0, (
                    f"Empty flows dir should exit with code 0, got: {e.code}"
                )
            finally:
                monkeypatch.undo()

    def test_discover_sessions_empty_directory(self):
        """discover_sessions returns empty list for empty directory."""
        from llm_flow_viewer.tui.widgets.session_list import discover_sessions

        with tempfile.TemporaryDirectory() as tmpdir:
            sessions = discover_sessions(tmpdir)
            assert len(sessions) == 0, (
                f"Empty directory should have 0 sessions, got: {len(sessions)}"
            )


# ─── VAL-CLI-005: Valid --session ──────────────────────────────────────────

class TestValidSession:
    """VAL-CLI-005: ``--session`` to load a specific session — TUI opens to that session."""

    def test_session_arg_passed_to_app(self):
        """The --session argument is passed to LLMFlowViewerApp."""
        app = LLMFlowViewerApp(session="01_flows-analyze_codebase")
        assert app.preselected_session == "01_flows-analyze_codebase", (
            f"preselected_session should be '01_flows-analyze_codebase', "
            f"got: '{app.preselected_session}'"
        )

    def test_session_defaults_to_none_in_app(self):
        """LLMFlowViewerApp defaults session to None."""
        app = LLMFlowViewerApp()
        assert app.preselected_session is None, (
            f"Default preselected_session should be None, got: '{app.preselected_session}'"
        )

    def test_main_passes_session_to_app(self):
        """main() passes --session argument to LLMFlowViewerApp."""
        from llm_flow_viewer.tui import app as app_module

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake session file
            session_name = "01_flows-analyze_codebase"
            file_path = os.path.join(tmpdir, session_name)
            with open(file_path, "w") as f:
                f.write("dummy content")

            captured_kwargs = {}

            # Capture the original run method to restore later
            original_run = app_module.LLMFlowViewerApp.run
            original_app_class = app_module.LLMFlowViewerApp

            # Create a wrapper class that captures constructor kwargs
            class CapturingApp(original_app_class):
                def __init__(self, *args, **kwargs):
                    captured_kwargs.update(kwargs)
                    super().__init__(*args, **kwargs)
                def run(self):
                    pass  # Prevent actual TUI launch

            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setattr(app_module, "LLMFlowViewerApp", CapturingApp)
            monkeypatch.setattr(
                app_module.sys, "argv",
                ["llm-flow-viewer", "--flows-dir", tmpdir, "--session", session_name],
            )
            try:
                main()
            except SystemExit as e:
                # If it exits, it should be code 0
                assert e.code == 0, f"Should exit with code 0, got: {e.code}"
            finally:
                monkeypatch.undo()

            # Verify session was passed to app
            assert "session" in captured_kwargs, (
                "session kwarg should be passed to LLMFlowViewerApp"
            )
            assert captured_kwargs["session"] == session_name, (
                f"session kwarg should be '{session_name}', "
                f"got: '{captured_kwargs.get('session')}'"
            )

    def test_valid_session_exit_code_0(self):
        """With valid --flows-dir and --session, exit code should be 0."""
        from llm_flow_viewer.tui import app as app_module

        with tempfile.TemporaryDirectory() as tmpdir:
            session_name = "01_flows-analyze_codebase"
            file_path = os.path.join(tmpdir, session_name)
            with open(file_path, "w") as f:
                f.write("dummy content")

            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setattr(
                app_module.sys, "argv",
                ["llm-flow-viewer", "--flows-dir", tmpdir, "--session", session_name],
            )
            # Override run() via subclass to prevent TUI launch
            original_app_class = app_module.LLMFlowViewerApp

            class NoRunApp(original_app_class):
                def run(self):
                    pass

            monkeypatch.setattr(app_module, "LLMFlowViewerApp", NoRunApp)
            try:
                # main() should NOT raise SystemExit for valid args
                main()
            except SystemExit as e:
                assert e.code == 0, (
                    f"Valid args should exit with code 0, got: {e.code}"
                )
            finally:
                monkeypatch.undo()


# ─── VAL-CLI-007: --help ───────────────────────────────────────────────────

class TestHelp:
    """VAL-CLI-007: ``--help`` — display help text and exit."""

    def test_help_exit_code_0(self):
        """``--help`` exits with code 0."""
        from llm_flow_viewer.tui import app as app_module

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            app_module.sys, "argv",
            ["llm-flow-viewer", "--help"],
        )
        try:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0, (
                f"--help should exit with code 0, got: {exc_info.value.code}"
            )
        finally:
            monkeypatch.undo()

    def test_help_prints_usage(self, capsys):
        """``--help`` prints usage information to stdout."""
        from llm_flow_viewer.tui import app as app_module

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            app_module.sys, "argv",
            ["llm-flow-viewer", "--help"],
        )
        try:
            with pytest.raises(SystemExit):
                main()
            captured = capsys.readouterr()
            assert captured.out, "Help output should not be empty"
            assert "usage:" in captured.out.lower(), (
                f"Help should contain 'usage:', got: {captured.out[:200]}"
            )
            assert "--flows-dir" in captured.out, (
                f"Help should mention --flows-dir, got: {captured.out[:200]}"
            )
            assert "--session" in captured.out, (
                f"Help should mention --session, got: {captured.out[:200]}"
            )
            assert "--version" in captured.out, (
                f"Help should mention --version, got: {captured.out[:200]}"
            )
        finally:
            monkeypatch.undo()

    def test_help_goes_to_stdout(self, capsys):
        """``--help`` output goes to stdout, not stderr."""
        from llm_flow_viewer.tui import app as app_module

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            app_module.sys, "argv",
            ["llm-flow-viewer", "--help"],
        )
        try:
            with pytest.raises(SystemExit):
                main()
            captured = capsys.readouterr()
            assert captured.out, "Help should be on stdout"
            assert captured.err == "", (
                "Help should not write to stderr"
            )
        finally:
            monkeypatch.undo()

    def test_help_with_h_flag(self, capsys):
        """``-h`` flag works the same as --help."""
        from llm_flow_viewer.tui import app as app_module

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            app_module.sys, "argv",
            ["llm-flow-viewer", "-h"],
        )
        try:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0, (
                f"-h should exit with code 0, got: {exc_info.value.code}"
            )
            captured = capsys.readouterr()
            assert "--flows-dir" in captured.out, (
                f"-h output should mention --flows-dir, got: {captured.out[:200]}"
            )
        finally:
            monkeypatch.undo()

    def test_help_does_not_launch_app(self):
        """TUI app is NOT launched when --help is given."""
        from llm_flow_viewer.tui import app as app_module

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            app_module.sys, "argv",
            ["llm-flow-viewer", "--help"],
        )

        def fail_on_create(*args, **kwargs):
            pytest.fail("LLMFlowViewerApp should not be instantiated with --help")

        monkeypatch.setattr(app_module, "LLMFlowViewerApp", fail_on_create)
        try:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0, (
                "--help should exit with code 0"
            )
        finally:
            monkeypatch.undo()


# ─── VAL-CLI-008: --version ────────────────────────────────────────────────

class TestVersionExitCode:
    """VAL-CLI-008: ``--version`` — display version and exit."""

    def test_version_exit_code_0(self):
        """``--version`` exits with code 0."""
        from llm_flow_viewer.tui import app as app_module

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            app_module.sys, "argv",
            ["llm-flow-viewer", "--version"],
        )
        try:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0, (
                f"--version should exit with code 0, got: {exc_info.value.code}"
            )
        finally:
            monkeypatch.undo()

    def test_version_prints_version_string(self, capsys):
        """``--version`` prints 'llm-flow-viewer X.Y.Z' to stdout."""
        from llm_flow_viewer.tui import app as app_module

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            app_module.sys, "argv",
            ["llm-flow-viewer", "--version"],
        )
        try:
            with pytest.raises(SystemExit):
                main()
            captured = capsys.readouterr()
            assert captured.out, "Version output should not be empty"
            assert captured.out.strip().startswith("llm-flow-viewer"), (
                f"Version should start with 'llm-flow-viewer', "
                f"got: '{captured.out.strip()}'"
            )
        finally:
            monkeypatch.undo()

    def test_version_goes_to_stdout(self, capsys):
        """``--version`` output goes to stdout, not stderr."""
        from llm_flow_viewer.tui import app as app_module

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            app_module.sys, "argv",
            ["llm-flow-viewer", "--version"],
        )
        try:
            with pytest.raises(SystemExit):
                main()
            captured = capsys.readouterr()
            assert captured.out, "Version should be on stdout"
            assert captured.err == "", (
                "Version should not write to stderr"
            )
        finally:
            monkeypatch.undo()

    def test_version_does_not_launch_app(self):
        """TUI app is NOT launched when --version is given."""
        from llm_flow_viewer.tui import app as app_module

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            app_module.sys, "argv",
            ["llm-flow-viewer", "--version"],
        )

        def fail_on_create(*args, **kwargs):
            pytest.fail("LLMFlowViewerApp should not be instantiated with --version")

        monkeypatch.setattr(app_module, "LLMFlowViewerApp", fail_on_create)
        try:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0, (
                "--version should exit with code 0"
            )
        finally:
            monkeypatch.undo()


# ─── VAL-CLI-011: Combined --flows-dir and --session ────────────────────────

class TestCombinedFlags:
    """VAL-CLI-011: Combined ``--flows-dir`` and ``--session`` flags."""

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
            file_path = os.path.join(tmpdir, session_name)
            with open(file_path, "w") as f:
                f.write("dummy content")

            app = LLMFlowViewerApp(flows_dir=tmpdir, session=session_name)
            assert app.flows_dir == os.path.abspath(tmpdir)
            assert app.preselected_session == session_name

    def test_combined_flags_no_conflict(self):
        """Combined --flows-dir and --session do not conflict."""
        from llm_flow_viewer.tui import app as app_module

        with tempfile.TemporaryDirectory() as tmpdir:
            session_name = "01_flows-analyze_codebase"
            file_path = os.path.join(tmpdir, session_name)
            with open(file_path, "w") as f:
                f.write("dummy content")

            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setattr(
                app_module.sys, "argv",
                ["llm-flow-viewer", "--flows-dir", tmpdir, "--session", session_name],
            )
            # Override run() via subclass to prevent TUI launch
            original_app_class = app_module.LLMFlowViewerApp

            class NoRunApp(original_app_class):
                def run(self):
                    pass

            monkeypatch.setattr(app_module, "LLMFlowViewerApp", NoRunApp)
            try:
                main()
            except SystemExit as e:
                assert e.code == 0, (
                    f"Combined valid flags should exit with code 0, got: {e.code}"
                )
            finally:
                monkeypatch.undo()


# ─── VAL-CLI-013: Relative path resolution ─────────────────────────────────

class TestRelativePath:
    """VAL-CLI-013: ``--flows-dir`` with relative path — resolved correctly."""

    def test_relative_path_resolved_to_absolute(self):
        """Relative --flows-dir path is resolved to absolute by LLMFlowViewerApp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                os.makedirs("my_flows", exist_ok=True)
                app = LLMFlowViewerApp(flows_dir="my_flows")
                expected = os.path.abspath(os.path.join(tmpdir, "my_flows"))
                assert app.flows_dir == expected, (
                    f"flows_dir should resolve to '{expected}', got: '{app.flows_dir}'"
                )
            finally:
                os.chdir(original_cwd)

    def test_relative_path_with_dot_prefix(self):
        """Relative --flows-dir './rel/path' resolves correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                os.makedirs("rel/path", exist_ok=True)
                app = LLMFlowViewerApp(flows_dir="./rel/path")
                expected = os.path.abspath(os.path.join(tmpdir, "rel", "path"))
                assert app.flows_dir == expected, (
                    f"flows_dir should resolve to '{expected}', got: '{app.flows_dir}'"
                )
            finally:
                os.chdir(original_cwd)

    def test_relative_path_with_parent_refs(self):
        """Relative --flows-dir '../other/project/flows' resolves correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                # Create nested structure: tmpdir/sub/proj/
                os.makedirs("sub/proj", exist_ok=True)
                os.makedirs("other/flows", exist_ok=True)
                os.chdir(os.path.join(tmpdir, "sub", "proj"))
                app = LLMFlowViewerApp(flows_dir="../../other/flows")
                expected = os.path.abspath(os.path.join(tmpdir, "other", "flows"))
                assert app.flows_dir == expected, (
                    f"flows_dir with parent refs should resolve to '{expected}', "
                    f"got: '{app.flows_dir}'"
                )
            finally:
                os.chdir(original_cwd)


# ─── VAL-CLI-014: Trailing slash handling ──────────────────────────────────

class TestTrailingSlash:
    """VAL-CLI-014: ``--flows-dir`` with trailing slash — handled."""

    def test_trailing_forward_slash_normalized(self):
        """Trailing forward slash in --flows-dir is normalized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path_with_slash = tmpdir + "/"
            app = LLMFlowViewerApp(flows_dir=path_with_slash)
            # The path should be normalized (no trailing slash, but still valid)
            assert os.path.isabs(app.flows_dir)
            assert os.path.isdir(app.flows_dir), (
                f"Normalized path should still be a valid directory: {app.flows_dir}"
            )
            # After normalization the path should match the original tmpdir
            assert os.path.normpath(app.flows_dir) == os.path.normpath(tmpdir), (
                f"Normalized path '{app.flows_dir}' should match '{tmpdir}'"
            )

    def test_trailing_backslash_normalized(self):
        """Trailing backslash in --flows-dir is normalized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path_with_backslash = tmpdir + os.sep
            app = LLMFlowViewerApp(flows_dir=path_with_backslash)
            assert os.path.isabs(app.flows_dir)
            assert os.path.isdir(app.flows_dir), (
                f"Normalized path should still be a valid directory: {app.flows_dir}"
            )
            assert os.path.normpath(app.flows_dir) == os.path.normpath(tmpdir), (
                f"Normalized path '{app.flows_dir}' should match '{tmpdir}'"
            )

    def test_trailing_slash_directory_check_passes(self):
        """Directory validation passes with trailing slash."""
        from llm_flow_viewer.tui import app as app_module

        with tempfile.TemporaryDirectory() as tmpdir:
            path_with_slash = tmpdir + "/"
            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setattr(
                app_module.sys, "argv",
                ["llm-flow-viewer", "--flows-dir", path_with_slash],
            )
            # Override run() via subclass to prevent TUI launch
            original_app_class = app_module.LLMFlowViewerApp

            class NoRunApp(original_app_class):
                def run(self):
                    pass

            monkeypatch.setattr(app_module, "LLMFlowViewerApp", NoRunApp)
            try:
                main()
            except SystemExit as e:
                assert e.code == 0, (
                    f"Trailing slash should not cause error, got: {e.code}"
                )
            else:
                pass
            finally:
                monkeypatch.undo()

    def test_multiple_trailing_slashes_normalized(self):
        """Multiple trailing slashes are normalized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path_with_slashes = tmpdir + "///"
            app = LLMFlowViewerApp(flows_dir=path_with_slashes)
            assert os.path.isdir(app.flows_dir), (
                f"Path with multiple slashes should resolve to valid directory: "
                f"{app.flows_dir}"
            )
            assert os.path.normpath(app.flows_dir) == os.path.normpath(tmpdir), (
                f"Normalized path '{app.flows_dir}' should match '{tmpdir}'"
            )

    def test_trailing_slash_with_relative_path(self):
        """Trailing slash with relative path is handled correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                os.makedirs("flows_dir", exist_ok=True)
                app = LLMFlowViewerApp(flows_dir="./flows_dir/")
                expected = os.path.abspath(os.path.join(tmpdir, "flows_dir"))
                assert app.flows_dir == expected, (
                    f"Relative path with trailing slash should resolve to "
                    f"'{expected}', got: '{app.flows_dir}'"
                )
            finally:
                os.chdir(original_cwd)


# ─── VAL-CLI-015: Comprehensive exit codes ─────────────────────────────────

class TestComprehensiveExitCodes:
    """VAL-CLI-015: Exit codes — 0 on success, non-zero on error."""

    def test_help_exit_code_0(self):
        """``--help`` exits with code 0."""
        from llm_flow_viewer.tui import app as app_module

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            app_module.sys, "argv",
            ["llm-flow-viewer", "--help"],
        )
        try:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0, (
                f"--help should exit with 0, got: {exc_info.value.code}"
            )
        finally:
            monkeypatch.undo()

    def test_version_exit_code_0(self):
        """``--version`` exits with code 0."""
        from llm_flow_viewer.tui import app as app_module

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            app_module.sys, "argv",
            ["llm-flow-viewer", "--version"],
        )
        try:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0, (
                f"--version should exit with 0, got: {exc_info.value.code}"
            )
        finally:
            monkeypatch.undo()

    def test_app_launch_exit_code_0(self):
        """Successful app launch exits with code 0 (when TUI quits normally)."""
        from llm_flow_viewer.tui import app as app_module

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setattr(
                app_module.sys, "argv",
                ["llm-flow-viewer", "--flows-dir", tmpdir],
            )
            # Override run() via subclass to prevent TUI launch
            original_app_class = app_module.LLMFlowViewerApp

            class NoRunApp(original_app_class):
                def run(self):
                    pass

            monkeypatch.setattr(app_module, "LLMFlowViewerApp", NoRunApp)
            try:
                main()
            except SystemExit as e:
                assert e.code == 0, (
                    f"Successful launch should exit with 0, got: {e.code}"
                )
            finally:
                monkeypatch.undo()

    def test_invalid_flows_dir_exit_code_1(self):
        """Invalid --flows-dir exits with code 1."""
        from llm_flow_viewer.tui import app as app_module

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            app_module.sys, "argv",
            ["llm-flow-viewer", "--flows-dir", "C:\\__nonexistent_test_dir__"],
        )
        try:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1, (
                f"Invalid --flows-dir should exit with 1, got: {exc_info.value.code}"
            )
        finally:
            monkeypatch.undo()

    def test_invalid_session_exit_code_1(self):
        """Invalid --session exits with code 1."""
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
                    f"Invalid --session should exit with 1, got: {exc_info.value.code}"
                )
            finally:
                monkeypatch.undo()

    def test_combined_invalid_args_exit_code_1(self):
        """Combined invalid args exit with code 1."""
        from llm_flow_viewer.tui import app as app_module

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            app_module.sys, "argv",
            ["llm-flow-viewer", "--flows-dir", "C:\\__nonexistent__", "--session", "bad"],
        )
        try:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1, (
                f"Combined invalid args should exit with 1, got: {exc_info.value.code}"
            )
        finally:
            monkeypatch.undo()

    def test_missing_flows_dir_default_exit_code_1(self):
        """Default ./flows dir not found exits with code 1."""
        from llm_flow_viewer.tui import app as app_module

        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            monkeypatch = pytest.MonkeyPatch()
            try:
                os.chdir(tmpdir)
                monkeypatch.setattr(
                    app_module.sys, "argv",
                    ["llm-flow-viewer"],
                )
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1, (
                    f"Missing default ./flows should exit with 1, "
                    f"got: {exc_info.value.code}"
                )
            finally:
                os.chdir(original_cwd)
                monkeypatch.undo()

    def test_unknown_flag_exit_code_2(self):
        """Unknown CLI flag exits with code 2 (argparse default)."""
        from llm_flow_viewer.tui import app as app_module

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            app_module.sys, "argv",
            ["llm-flow-viewer", "--unknown-flag"],
        )
        try:
            with pytest.raises(SystemExit) as exc_info:
                main()
            # argparse uses code 2 for unknown arguments
            assert exc_info.value.code == 2, (
                f"Unknown flag should exit with 2, got: {exc_info.value.code}"
            )
        finally:
            monkeypatch.undo()
