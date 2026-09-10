from pathlib import Path
from unittest.mock import Mock

import pytest
import typer
from rich.progress import TaskID
from typer.testing import CliRunner

from every_python.main import (
    BuildOptions,
    ConfigureCacheError,
    _resolve_configure_cache,
    _run_configure,
    app,
    build_python,
)
from every_python.runner import CommandResult

COMMIT = "abc123def456"
INSTALL = ["install", "main"]
RUN = ["run", "main", "python"]
BISECT = ["bisect", "--good", "good", "--bad", "bad", "--run", "exit 0"]


@pytest.fixture
def build_cli(mocker, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("EVERY_PYTHON_CONFIGURE_CACHE_FILE", raising=False)
    monkeypatch.delenv("EVERY_PYTHON_REFERENCE_REPO", raising=False)
    repo = tmp_path / "cpython"
    (repo / ".git").mkdir(parents=True)
    (repo / ".git" / "BISECT_LOG").touch()
    mocker.patch("every_python.main._ensure_repo", return_value=repo)
    mocker.patch("every_python.main._resolve_ref", return_value=COMMIT)
    mocker.patch("every_python.main.platform.system", return_value="Linux")
    mocker.patch("every_python.main.BUILDS_DIR", tmp_path / "builds")
    python = tmp_path / "python"
    python.touch()
    mocker.patch("every_python.main.python_binary_location", return_value=python)
    mocker.patch("every_python.main.os.execv")
    mocker.patch("every_python.main.subprocess.run", return_value=Mock(returncode=0))
    commands = mocker.patch("every_python.main.get_runner").return_value

    def git_result(args, *positional, **kwargs):
        stdout = COMMIT if args == ["rev-parse", "HEAD"] else "is the first bad commit"
        return CommandResult(0, stdout, "")

    commands.run_git.side_effect = git_result
    build = mocker.patch(
        "every_python.main.build_python", return_value=tmp_path / COMMIT
    )
    return build, commands, repo


@pytest.mark.parametrize("command", [INSTALL, RUN, BISECT])
@pytest.mark.parametrize(
    "args, env_cache, expected",
    [
        ([], "", None),
        (["--configure-cache", "local.cache"], "", "local.cache"),
        ([], "default.cache", "default.cache"),
        (["--configure-cache", "local.cache"], "default.cache", "local.cache"),
        (["--no-configure-cache"], "", None),
        (["--no-configure-cache"], "default.cache", None),
        (
            ["--no-configure-cache", "--configure-cache", "local.cache"],
            "default.cache",
            None,
        ),
        (
            ["--configure-cache", "local.cache", "--no-configure-cache"],
            "default.cache",
            None,
        ),
    ],
)
def test_cache_options(build_cli, command, args, env_cache, expected):
    build, _, _ = build_cli
    result = CliRunner().invoke(
        app, command + args, env={"EVERY_PYTHON_CONFIGURE_CACHE_FILE": env_cache}
    )
    assert result.exit_code == 0, result.output
    build.assert_called_once_with(
        COMMIT, BuildOptions(configure_cache=Path(expected) if expected else None)
    )


def test_cache_paths(mocker, monkeypatch, tmp_path):
    mocker.patch("every_python.main.platform.system", return_value="Linux")
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "cpython"
    assert (
        _resolve_configure_cache(Path("config cache"), repo)
        == tmp_path / "config cache"
    )
    assert (
        _resolve_configure_cache(Path("~/config.cache"), repo)
        == Path.home() / "config.cache"
    )


@pytest.mark.parametrize("command", [INSTALL, BISECT])
@pytest.mark.parametrize(
    "system, path",
    [("Linux", "cpython/config.cache"), ("Linux", "."), ("Windows", "config.cache")],
)
def test_invalid_cache_fails_before_cleanup(build_cli, mocker, command, system, path):
    _, commands, _ = build_cli
    mocker.patch("every_python.main.build_python", wraps=build_python)
    mocker.patch("every_python.main.platform.system", return_value=system)
    result = CliRunner().invoke(app, command + ["--configure-cache", path])
    assert result.exit_code == 1
    commands.run_git.assert_not_called()


def test_build_passes_cache_to_configure(build_cli, mocker, tmp_path):
    _, commands, repo = build_cli
    mocker.patch("every_python.main._record_build_repository")
    commands.run.return_value = CommandResult(0, "", "")

    build_python(
        COMMIT, BuildOptions(configure_cache=Path("config cache"), ccache=False)
    )

    configure = commands.run.call_args_list[0]
    assert configure.args[0][-1] == f"--cache-file={tmp_path / 'config cache'}"
    assert configure.kwargs["cwd"] == repo


@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize(
    "cache, error", [(None, typer.Exit), (Path("config.cache"), ConfigureCacheError)]
)
def test_configure_failure(cache, error, verbose, tmp_path, capsys):
    commands = Mock()
    commands.run.return_value = CommandResult(1, "", "incompatible settings")
    with pytest.raises(error) as exc:
        _run_configure(
            commands,
            tmp_path,
            tmp_path / "build",
            frozenset(),
            verbose,
            Mock(),
            TaskID(1),
            configure_cache=cache,
        )
    assert exc.value.exit_code == 1
    commands.run.assert_called_once()
    assert ("--no-configure-cache" in capsys.readouterr().out) == (cache is not None)


@pytest.mark.parametrize(
    "error, exit_code", [(ConfigureCacheError, 1), (typer.Exit, 0)]
)
def test_bisect_stops_for_cache_errors_and_skips_other_build_failures(
    build_cli, error, exit_code
):
    build, commands, repo = build_cli
    build.side_effect = [error(1), build.return_value]
    result = CliRunner().invoke(app, BISECT + ["--configure-cache", "config.cache"])
    assert result.exit_code == exit_code, result.output
    skipped = any(
        c.args[0] == ["bisect", "skip"] for c in commands.run_git.call_args_list
    )
    assert skipped == (error is typer.Exit)
    commands.run_git.assert_called_with(["bisect", "reset"], repo)
