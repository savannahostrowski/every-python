from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import typer
from typer.testing import CliRunner

from every_python.main import BuildOptions, _get_ccache_env, app

runner = CliRunner()


@patch("every_python.main.build_python")
@patch("every_python.main._resolve_ref")
def test_install_with_ccache(
    mock_resolve: Mock, mock_build: Mock, tmp_path: Path
) -> None:
    mock_resolve.return_value = "abc123def456"
    mock_build.return_value = tmp_path / "abc123def456"

    result = runner.invoke(app, ["install", "main", "--ccache"])

    assert result.exit_code == 0
    mock_build.assert_called_once_with("abc123def456", BuildOptions(ccache=True))


@patch("every_python.main.shutil.which", return_value="/opt/homebrew/bin/ccache")
@patch("every_python.main.platform.system", return_value="Darwin")
def test_uses_clang_on_macos(_mock_platform: Mock, _mock_which: Mock) -> None:
    with patch.dict("os.environ", {}, clear=True):
        env = _get_ccache_env()

    assert env["CC"] == "/opt/homebrew/bin/ccache clang"


@patch("every_python.main.shutil.which", return_value="/usr/bin/ccache")
@patch("every_python.main.platform.system", return_value="Linux")
def test_preserves_configured_compiler(_mock_platform: Mock, _mock_which: Mock) -> None:
    with patch.dict("os.environ", {"CC": "gcc-15"}, clear=True):
        env = _get_ccache_env()

    assert env["CC"] == "/usr/bin/ccache gcc-15"


@patch("every_python.main.shutil.which", return_value="/usr/bin/ccache")
@patch("every_python.main.platform.system", return_value="Linux")
def test_does_not_wrap_ccache_twice(_mock_platform: Mock, _mock_which: Mock) -> None:
    with patch.dict("os.environ", {"CC": "ccache clang"}, clear=True):
        env = _get_ccache_env()

    assert env["CC"] == "ccache clang"


@patch("every_python.main.shutil.which", return_value=None)
@patch("every_python.main.platform.system", return_value="Linux")
def test_missing_ccache_exits(_mock_platform: Mock, _mock_which: Mock) -> None:
    with pytest.raises(typer.Exit):
        _get_ccache_env()


@patch("every_python.main.platform.system", return_value="Windows")
def test_windows_exits(_mock_platform: Mock) -> None:
    with pytest.raises(typer.Exit):
        _get_ccache_env()
