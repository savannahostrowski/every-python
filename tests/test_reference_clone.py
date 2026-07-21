from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from every_python.main import BuildOptions, _ensure_repo, app
from every_python.runner import set_runner

runner = CliRunner()


@pytest.fixture(autouse=True)
def reset_runner():
    yield
    set_runner(None)  # type: ignore


@patch("subprocess.run")
def test_clone_uses_reference_repository(mock_run: Mock, tmp_path: Path) -> None:
    mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
    repo_dir = tmp_path / "managed-cpython"
    reference_repo = tmp_path / "local-cpython"

    with patch("every_python.main.REPO_DIR", repo_dir):
        _ensure_repo(reference_repo=reference_repo)

    assert mock_run.call_args.args[0] == [
        "git",
        "clone",
        "--filter=blob:none",
        "--reference-if-able",
        str(reference_repo),
        "--dissociate",
        "https://github.com/python/cpython.git",
        str(repo_dir),
    ]


@patch("every_python.main.build_python")
@patch("every_python.main._resolve_ref")
def test_install_propagates_reference_repository(
    mock_resolve: Mock, mock_build: Mock, tmp_path: Path
) -> None:
    reference_repo = tmp_path / "cpython"
    mock_resolve.return_value = "abc123def456"
    mock_build.return_value = tmp_path / "abc123def456"

    result = runner.invoke(
        app,
        ["install", "main", "--reference-repo", str(reference_repo)],
    )

    assert result.exit_code == 0
    mock_resolve.assert_called_once_with("main", None, reference_repo)
    mock_build.assert_called_once_with(
        "abc123def456", BuildOptions(reference_repo=reference_repo)
    )


@patch("every_python.main.build_python")
@patch("every_python.main._resolve_ref")
def test_reference_repository_environment_variable(
    mock_resolve: Mock, mock_build: Mock, tmp_path: Path
) -> None:
    reference_repo = tmp_path / "cpython"
    mock_resolve.return_value = "abc123def456"
    mock_build.return_value = tmp_path / "abc123def456"

    result = runner.invoke(
        app,
        ["install", "main"],
        env={"EVERY_PYTHON_REFERENCE_REPO": str(reference_repo)},
    )

    assert result.exit_code == 0
    mock_resolve.assert_called_once_with("main", None, reference_repo)
    mock_build.assert_called_once_with(
        "abc123def456", BuildOptions(reference_repo=reference_repo)
    )
