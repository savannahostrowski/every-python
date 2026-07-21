from pathlib import Path
from unittest.mock import Mock, call, patch

from rich.progress import TaskID
from typer.testing import CliRunner

from every_python.main import BuildOptions, _build_and_install_unix, app

runner = CliRunner()


@patch("every_python.main.build_python")
@patch("every_python.main._resolve_ref")
def test_install_passes_jobs(
    mock_resolve: Mock, mock_build: Mock, tmp_path: Path
) -> None:
    mock_resolve.return_value = "abc123def456"
    mock_build.return_value = tmp_path / "abc123def456"

    result = runner.invoke(app, ["install", "main", "--jobs", "6"])

    assert result.exit_code == 0
    mock_build.assert_called_once_with("abc123def456", BuildOptions(jobs=6))


def test_jobs_must_be_positive() -> None:
    result = runner.invoke(app, ["install", "main", "--jobs", "0"])

    assert result.exit_code == 2


@patch("every_python.main.REPO_DIR", Path("/tmp/cpython"))
def test_build_uses_requested_jobs_and_install_is_serial() -> None:
    command_runner = Mock()
    command_runner.run.return_value = Mock(success=True, stderr="")
    progress = Mock()
    build_env = {"CC": "ccache clang"}

    _build_and_install_unix(
        command_runner,
        verbose=False,
        progress=progress,
        task=TaskID(1),
        build_env=build_env,
        jobs=6,
    )

    assert command_runner.run.call_args_list == [
        call(
            ["make", "-j6"],
            cwd=Path("/tmp/cpython"),
            capture_output=True,
            env=build_env,
        ),
        call(
            ["make", "install"],
            cwd=Path("/tmp/cpython"),
            env=build_env,
        ),
    ]
