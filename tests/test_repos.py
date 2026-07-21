from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from every_python.main import (
    BuildOptions,
    _build_repositories,
    _commit_link,
    _ensure_repo,
    _normalize_repo,
    _record_build_repository,
    _repo_dir,
    _repository_label,
    app,
)
from every_python.runner import set_runner

runner = CliRunner()


@pytest.fixture(autouse=True)
def reset_runner():
    yield
    set_runner(None)  # type: ignore


def test_normalize_github_shorthand() -> None:
    assert (
        _normalize_repo("LazyImportsCabal/cpython")
        == "https://github.com/LazyImportsCabal/cpython.git"
    )


def test_preserve_full_git_url() -> None:
    url = "ssh://git@github.com/LazyImportsCabal/cpython.git"
    assert _normalize_repo(url) == url


def test_default_repository_uses_legacy_directory() -> None:
    with patch("every_python.main.REPO_DIR", Path("/tmp/legacy-cpython")):
        assert _repo_dir(None) == Path("/tmp/legacy-cpython")


def test_record_build_repositories(tmp_path: Path) -> None:
    _record_build_repository(tmp_path, None)
    _record_build_repository(tmp_path, "LazyImportsCabal/cpython")
    _record_build_repository(tmp_path, "LazyImportsCabal/cpython")

    assert _build_repositories(tmp_path) == [
        "https://github.com/python/cpython.git",
        "https://github.com/LazyImportsCabal/cpython.git",
    ]


def test_repository_label_shortens_github_urls() -> None:
    assert (
        _repository_label("https://github.com/LazyImportsCabal/cpython.git")
        == "LazyImportsCabal"
    )


def test_commit_link_uses_github_repository() -> None:
    commit = "abc123def456"
    link = _commit_link(commit, "https://github.com/LazyImportsCabal/cpython.git")

    assert link.plain == "abc123d"
    assert link.style == (
        "bright_magenta underline link "
        "https://github.com/LazyImportsCabal/cpython/commit/abc123def456"
    )


def test_custom_repositories_get_separate_stable_directories(tmp_path: Path) -> None:
    with patch("every_python.main.REPOS_DIR", tmp_path):
        first = _repo_dir("one/cpython")
        second = _repo_dir("two/cpython")

    assert first.parent == tmp_path
    assert first.name.startswith("cpython-")
    assert first != second


@patch("subprocess.run")
def test_clone_custom_repository(mock_run: Mock, tmp_path: Path) -> None:
    mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

    with patch("every_python.main.REPOS_DIR", tmp_path):
        repo_dir = _ensure_repo("LazyImportsCabal/cpython")

    command = mock_run.call_args.args[0]
    assert command[:4] == [
        "git",
        "clone",
        "--filter=blob:none",
        "https://github.com/LazyImportsCabal/cpython.git",
    ]
    assert command[4] == str(repo_dir)


@patch("every_python.main.build_python")
@patch("every_python.main._resolve_ref")
def test_install_propagates_custom_repository(
    mock_resolve: Mock, mock_build: Mock, tmp_path: Path
) -> None:
    mock_resolve.return_value = "abc123def456"
    mock_build.return_value = tmp_path / "abc123def456"

    result = runner.invoke(
        app,
        ["install", "8639e50", "--repo", "LazyImportsCabal/cpython"],
    )

    assert result.exit_code == 0
    mock_resolve.assert_called_once_with("8639e50", "LazyImportsCabal/cpython")
    mock_build.assert_called_once_with(
        "abc123def456", BuildOptions(repo="LazyImportsCabal/cpython")
    )
