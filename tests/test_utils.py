import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import subprocess

from every_python.utils import (
    BuildInfo,
    get_llvm_version_for_commit,
    check_llvm_available,
    _check_tool_available,
    _check_tool_version,
    _get_homebrew_llvm_tool,
)


class TestBuildInfo:
    """Test BuildInfo dataclass."""

    def test_non_jit_build(self):
        """Test non-JIT build info."""
        info = BuildInfo(commit="abc123def456", jit_enabled=False)
        assert info.suffix == ""
        assert info.directory_name == "abc123def456"

    def test_jit_build(self):
        """Test JIT build info."""
        info = BuildInfo(commit="abc123def456", jit_enabled=True)
        assert info.suffix == "-jit"
        assert info.directory_name == "abc123def456-jit"

    def test_get_path(self, tmp_path):
        """Test getting build path."""
        builds_dir = tmp_path / "builds"
        info = BuildInfo(commit="abc123", jit_enabled=True)
        path = info.get_path(builds_dir)
        assert path == builds_dir / "abc123-jit"

    def test_from_directory_name_non_jit(self):
        """Test parsing non-JIT directory name."""
        info = BuildInfo.from_directory_name("abc123def456")
        assert info.commit == "abc123def456"
        assert info.jit_enabled is False

    def test_from_directory_name_jit(self):
        """Test parsing JIT directory name."""
        info = BuildInfo.from_directory_name("abc123def456-jit")
        assert info.commit == "abc123def456"
        assert info.jit_enabled is True

    def test_from_directory(self, tmp_path):
        """Test parsing from directory path."""
        build_dir = tmp_path / "abc123-jit"
        build_dir.mkdir(parents=True)
        info = BuildInfo.from_directory(build_dir)
        assert info.commit == "abc123"
        assert info.jit_enabled is True


class TestGetLLVMVersion:
    """Test LLVM version detection from commit."""

    @patch("subprocess.run")
    def test_get_llvm_version_success(self, mock_run, tmp_path):
        """Test successfully getting LLVM version from commit."""
        repo_dir = tmp_path / "repo"

        mock_run.return_value = Mock(
            returncode=0,
            stdout="_LLVM_VERSION = 20\n",
            stderr="",
        )

        version = get_llvm_version_for_commit("abc123d", repo_dir)

        assert version == "20"
        mock_run.assert_called_once()
        assert "Tools/jit/_llvm.py" in str(mock_run.call_args)

    @patch("subprocess.run")
    def test_get_llvm_version_with_quotes(self, mock_run, tmp_path):
        """Test getting LLVM version with quotes."""
        repo_dir = tmp_path / "repo"

        mock_run.return_value = Mock(
            returncode=0,
            stdout='_LLVM_VERSION = "20"\n',
            stderr="",
        )

        version = get_llvm_version_for_commit("abc123d", repo_dir)

        assert version == "20"

    @patch("subprocess.run")
    def test_get_llvm_version_file_not_found(self, mock_run, tmp_path):
        """Test when JIT file doesn't exist in commit."""
        repo_dir = tmp_path / "repo"

        mock_run.side_effect = subprocess.CalledProcessError(
            1, "git show", stderr="fatal: path not in tree"
        )

        version = get_llvm_version_for_commit("abc123d", repo_dir)

        assert version is None

    @patch("subprocess.run")
    def test_get_llvm_version_no_match(self, mock_run, tmp_path):
        """Test when LLVM_VERSION pattern not found."""
        repo_dir = tmp_path / "repo"

        mock_run.return_value = Mock(
            returncode=0,
            stdout="# Some other content\n",
            stderr="",
        )

        version = get_llvm_version_for_commit("abc123d", repo_dir)

        assert version is None


class TestCheckLLVMAvailable:
    """Test LLVM availability checking."""

    @patch("every_python.utils._check_tool_available")
    def test_all_tools_available(self, mock_check_tool):
        """Test when all required tools are available."""
        mock_check_tool.return_value = True

        result = check_llvm_available("20")

        assert result is True
        # Should check both clang and llvm-readobj
        assert mock_check_tool.call_count == 2
        calls = [call[0][0] for call in mock_check_tool.call_args_list]
        assert "clang" in calls
        assert "llvm-readobj" in calls

    @patch("every_python.utils._check_tool_available")
    def test_missing_tool(self, mock_check_tool):
        """Test when a tool is missing."""

        def side_effect(tool, version):
            return tool == "clang"  # Only clang available

        mock_check_tool.side_effect = side_effect

        result = check_llvm_available("20")

        assert result is False


class TestCheckToolAvailable:
    """Test individual tool availability checking."""

    @patch("every_python.utils._check_tool_version")
    def test_versioned_tool_available(self, mock_check_version):
        """Test finding versioned tool (e.g., clang-20)."""
        mock_check_version.return_value = True

        result = _check_tool_available("clang", "20")

        assert result is True
        # Should check clang-20 first
        assert mock_check_version.call_args_list[0][0][0] == "clang-20"

    @patch("every_python.utils._check_tool_version")
    def test_unversioned_tool_available(self, mock_check_version):
        """Test finding unversioned tool in PATH."""

        def side_effect(tool, version):
            return tool == "clang"  # Only unversioned works

        mock_check_version.side_effect = side_effect

        result = _check_tool_available("clang", "20")

        assert result is True

    @patch("platform.system")
    @patch("every_python.utils._get_homebrew_llvm_tool")
    @patch("every_python.utils._check_tool_version")
    def test_homebrew_tool_available(
        self, mock_check_version, mock_homebrew_tool, mock_platform
    ):
        """Test finding tool via Homebrew on macOS."""
        mock_platform.return_value = "Darwin"  # Simulate macOS
        mock_check_version.return_value = False  # Not in PATH
        mock_homebrew_tool.return_value = "/opt/homebrew/opt/llvm@20/bin/clang"

        # Make the homebrew path check succeed
        def version_check_side_effect(tool, version):
            return "/opt/homebrew" in tool

        mock_check_version.side_effect = version_check_side_effect

        result = _check_tool_available("clang", "20")

        assert result is True
        mock_homebrew_tool.assert_called_once_with("clang", "20")

    @patch("every_python.utils._get_homebrew_llvm_tool")
    @patch("every_python.utils._check_tool_version")
    def test_tool_not_available(self, mock_check_version, mock_homebrew_tool):
        """Test when tool is not available anywhere."""
        mock_check_version.return_value = False
        mock_homebrew_tool.return_value = None

        result = _check_tool_available("clang", "20")

        assert result is False


class TestCheckToolVersion:
    """Test tool version checking."""

    @patch("subprocess.run")
    def test_correct_version(self, mock_run):
        """Test tool with correct version."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="clang version 20.1.8\n",
            stderr="",
        )

        result = _check_tool_version("clang-20", "20")

        assert result is True

    @patch("subprocess.run")
    def test_wrong_version(self, mock_run):
        """Test tool with wrong version."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="clang version 19.1.0\n",
            stderr="",
        )

        result = _check_tool_version("clang-20", "20")

        assert result is False

    @patch("subprocess.run")
    def test_tool_not_found(self, mock_run):
        """Test when tool is not found."""
        mock_run.side_effect = FileNotFoundError()

        result = _check_tool_version("clang-20", "20")

        assert result is False

    @patch("subprocess.run")
    def test_command_fails(self, mock_run):
        """Test when tool command fails."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "clang --version")

        result = _check_tool_version("clang-20", "20")

        assert result is False

    @patch("subprocess.run")
    def test_timeout(self, mock_run):
        """Test when tool command times out."""
        mock_run.side_effect = subprocess.TimeoutExpired("clang --version", 5)

        result = _check_tool_version("clang-20", "20")

        assert result is False


class TestGetHomebrewLLVMTool:
    """Test Homebrew LLVM tool detection."""

    @patch("subprocess.run")
    def test_versioned_formula(self, mock_run):
        """Test finding versioned Homebrew formula (llvm@20)."""

        def run_side_effect(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args", [])
            if "brew" in cmd and "--prefix" in cmd and "llvm@20" in cmd:
                return Mock(
                    returncode=0,
                    stdout="/opt/homebrew/opt/llvm@20\n",
                    stderr="",
                )
            elif "--version" in cmd:
                return Mock(returncode=0, stdout="", stderr="")
            return Mock(returncode=1, stdout="", stderr="")

        mock_run.side_effect = run_side_effect

        result = _get_homebrew_llvm_tool("clang", "20")

        assert result == "/opt/homebrew/opt/llvm@20/bin/clang"

    @patch("subprocess.run")
    def test_default_formula(self, mock_run):
        """Test falling back to default llvm formula."""

        def run_side_effect(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args", [])
            if "brew" in cmd and "--prefix" in cmd and "llvm@20" in cmd:
                # Versioned formula not found
                raise subprocess.CalledProcessError(1, cmd)
            elif "brew" in cmd and "--prefix" in cmd and cmd[-1] == "llvm":
                return Mock(
                    returncode=0,
                    stdout="/opt/homebrew/opt/llvm\n",
                    stderr="",
                )
            elif "--version" in cmd:
                return Mock(returncode=0, stdout="", stderr="")
            return Mock(returncode=1, stdout="", stderr="")

        mock_run.side_effect = run_side_effect

        result = _get_homebrew_llvm_tool("clang", "20")

        assert result == "/opt/homebrew/opt/llvm/bin/clang"

    @patch("subprocess.run")
    def test_not_found(self, mock_run):
        """Test when Homebrew LLVM is not installed."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "brew")

        result = _get_homebrew_llvm_tool("clang", "20")

        assert result is None

    @patch("subprocess.run")
    def test_tool_missing_in_prefix(self, mock_run):
        """Test when brew prefix exists but tool is missing."""

        def run_side_effect(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args", [])
            if "brew" in cmd and "--prefix" in cmd:
                return Mock(
                    returncode=0,
                    stdout="/opt/homebrew/opt/llvm\n",
                    stderr="",
                )
            elif "--version" in cmd:
                # Tool doesn't exist
                return Mock(returncode=1, stdout="", stderr="")
            return Mock(returncode=1, stdout="", stderr="")

        mock_run.side_effect = run_side_effect

        result = _get_homebrew_llvm_tool("clang", "20")

        assert result is None

    @patch("subprocess.run")
    def test_timeout(self, mock_run):
        """Test handling of timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("brew --prefix", 5)

        result = _get_homebrew_llvm_tool("clang", "20")

        assert result is None
