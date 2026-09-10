from pathlib import Path
from unittest.mock import Mock, call

import pytest
import typer
from rich.progress import TaskID
from typer.testing import CliRunner

from every_python.main import (
    BuildOptions,
    ConfigureCacheError,
    _get_configure_args,
    _resolve_configure_cache,
    _run_configure,
    app,
    build_python,
)
from every_python.runner import CommandResult

runner = CliRunner()
COMMIT = "abc123def456"


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch):
    monkeypatch.delenv("EVERY_PYTHON_CONFIGURE_CACHE_FILE", raising=False)
    monkeypatch.delenv("EVERY_PYTHON_REFERENCE_REPO", raising=False)


@pytest.fixture
def cli_build(mocker, tmp_path):
    repo_dir = tmp_path / "cpython"
    (repo_dir / ".git").mkdir(parents=True)
    (repo_dir / ".git" / "BISECT_LOG").touch()
    mocker.patch("every_python.main._ensure_repo", return_value=repo_dir)
    mocker.patch("every_python.main._resolve_ref", return_value=COMMIT)
    mocker.patch("every_python.main.BUILDS_DIR", tmp_path / "builds")
    mocker.patch("every_python.main.platform.system", return_value="Linux")
    python_bin = tmp_path / "python"
    python_bin.touch()
    mocker.patch("every_python.main.python_binary_location", return_value=python_bin)
    mocker.patch("every_python.main.os.execv")
    mocker.patch("every_python.main.subprocess.run", return_value=Mock(returncode=0))
    command_runner = mocker.patch("every_python.main.get_runner").return_value
    command_runner.run_git.return_value = CommandResult(
        0, f"{COMMIT} is the first bad commit", ""
    )
    build = mocker.patch("every_python.main.build_python")
    build.return_value = tmp_path / "builds" / COMMIT
    return build, command_runner, repo_dir


def invoke_build(command, arguments, env=None):
    if command == "install":
        args = ["install", "main", *arguments]
    elif command == "run":
        args = ["run", "main", *arguments, "--", "python", "--version"]
    else:
        args = ["bisect", "--good", "good", "--bad", "bad", "--run", "exit 0"]
        args.extend(arguments)
    return runner.invoke(app, args, env=env)


@pytest.mark.parametrize("command", ["install", "run", "bisect"])
@pytest.mark.parametrize(
    "source",
    [
        "default",
        "cli",
        "env",
        "override",
        "devnull",
        "disable",
        "disable_only",
        "disable_before_cli",
        "disable_after_cli",
    ],
)
def test_commands_propagate_configure_cache(cli_build, tmp_path, command, source):
    build, _, _ = cli_build
    cache = tmp_path / "config cache"
    env_cache = tmp_path / "environment.cache"
    env = {}
    arguments = []
    expected = None
    if source in {
        "env",
        "override",
        "devnull",
        "disable",
        "disable_before_cli",
        "disable_after_cli",
    }:
        env["EVERY_PYTHON_CONFIGURE_CACHE_FILE"] = str(env_cache)
        expected = env_cache
    if source in {
        "cli",
        "override",
        "devnull",
        "disable_before_cli",
        "disable_after_cli",
    }:
        expected = Path("/dev/null") if source == "devnull" else cache
        arguments = ["--configure-cache", str(expected)]
    if source.startswith("disable"):
        expected = None
        # An invalid environment path must not prevent disabling the cache.
        env_cache.mkdir()
        if source == "disable_before_cli":
            arguments.insert(0, "--no-configure-cache")
        else:
            arguments.append("--no-configure-cache")

    result = invoke_build(command, arguments, env)

    assert result.exit_code == 0, result.output
    assert len(build.call_args_list) == 1
    assert build.call_args.args[1] == BuildOptions(configure_cache=expected)


def test_no_configure_cache_overrides_environment_on_windows(
    cli_build, mocker, tmp_path
):
    build, _, _ = cli_build
    mocker.patch("every_python.main.platform.system", return_value="Windows")

    result = invoke_build(
        "bisect",
        ["--no-configure-cache"],
        env={"EVERY_PYTHON_CONFIGURE_CACHE_FILE": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    assert build.call_args.args[1].configure_cache is None


@pytest.mark.parametrize("system", ["Linux", "Darwin"])
def test_configure_args_include_cache_as_one_argument(mocker, tmp_path, system):
    mocker.patch("every_python.main.platform.system", return_value=system)
    cache = tmp_path / "config cache;literal"
    args = _get_configure_args(tmp_path, frozenset({"jit", "pgo", "nogil"}), cache)
    assert args == [
        "./configure",
        "--prefix",
        str(tmp_path),
        "--with-pydebug",
        f"--cache-file={cache}",
        "--enable-experimental-jit",
        "--enable-optimizations",
        "--disable-gil",
    ]


def test_resolve_relative_and_home_paths(mocker, monkeypatch, tmp_path):
    mocker.patch("every_python.main.platform.system", return_value="Linux")
    monkeypatch.chdir(tmp_path)
    assert _resolve_configure_cache(Path("~/config.cache"), tmp_path / "cpython") == (
        Path.home() / "config.cache"
    )
    # A new cache file is allowed; configure will create it.
    assert _resolve_configure_cache(Path("config.cache"), tmp_path / "cpython") == (
        tmp_path / "config.cache"
    )


@pytest.mark.parametrize("system", ["Linux", "Darwin", "Windows"])
def test_no_cache_preserves_platform_defaults(mocker, tmp_path, system):
    mocker.patch("every_python.main.platform.system", return_value=system)
    assert _resolve_configure_cache(None, tmp_path) is None
    assert not any(
        arg.startswith("--cache-file=")
        for arg in _get_configure_args(tmp_path, frozenset())
    )


def test_devnull_disables_cache(mocker, tmp_path):
    mocker.patch("every_python.main.platform.system", return_value="Linux")
    mocker.patch("every_python.main.os.devnull", "/dev/null")
    assert _resolve_configure_cache(Path("/dev/null"), tmp_path) is None


@pytest.mark.parametrize("command", ["install", "bisect"])
@pytest.mark.parametrize("location", ["checkout", "directory", "windows"])
def test_invalid_cache_fails_before_checkout_or_cleanup(
    cli_build, mocker, tmp_path, command, location
):
    _, command_runner, repo_dir = cli_build
    mocker.patch("every_python.main.build_python", wraps=build_python)
    cache = tmp_path / "config.cache"
    if location == "checkout":
        cache = repo_dir / "config.cache"
        cache.write_text("preserve this cache")
    elif location == "directory":
        cache.mkdir()
    else:
        mocker.patch("every_python.main.platform.system", return_value="Windows")

    result = invoke_build(command, ["--configure-cache", str(cache)])

    assert result.exit_code == 1
    assert "cache" in result.output
    command_runner.run_git.assert_not_called()
    if location == "checkout":
        assert cache.read_text() == "preserve this cache"


def test_rejects_symlink_into_checkout(mocker, tmp_path):
    mocker.patch("every_python.main.platform.system", return_value="Linux")
    repo_dir = tmp_path / "cpython"
    repo_dir.mkdir()
    link = tmp_path / "cache-link"
    try:
        link.symlink_to(repo_dir / "config.cache")
    except OSError:
        pytest.skip("Symlinks are unavailable")
    with pytest.raises(ConfigureCacheError):
        _resolve_configure_cache(link, repo_dir)


def test_build_passes_absolute_cache_after_cleanup(mocker, monkeypatch, tmp_path):
    repo_dir = tmp_path / "cpython"
    repo_dir.mkdir()
    builds_dir = tmp_path / "builds"
    mocker.patch("every_python.main._ensure_repo", return_value=repo_dir)
    mocker.patch("every_python.main.BUILDS_DIR", builds_dir)
    mocker.patch("every_python.main.platform.system", return_value="Linux")
    mocker.patch("every_python.main._record_build_repository")
    python_bin = mocker.patch("every_python.main.python_binary_location").return_value
    python_bin.exists.return_value = True
    command_runner = mocker.patch("every_python.main.get_runner").return_value
    command_runner.run.return_value = CommandResult(0, "", "")
    command_runner.run_git.return_value = CommandResult(0, "", "")
    monkeypatch.chdir(tmp_path)

    result = build_python(
        COMMIT, BuildOptions(configure_cache=Path("config cache"), ccache=False, jobs=2)
    )

    assert result == builds_dir / COMMIT
    assert command_runner.method_calls == [
        call.run_git(["checkout", COMMIT], repo_dir),
        call.run_git(["clean", "-fdx"], repo_dir=repo_dir),
        call.run(
            [
                "./configure",
                "--prefix",
                str(result),
                "--with-pydebug",
                f"--cache-file={tmp_path / 'config cache'}",
            ],
            cwd=repo_dir,
            capture_output=True,
            env=None,
        ),
        call.run(["make", "-j2"], cwd=repo_dir, capture_output=True, env=None),
        call.run(["make", "install"], cwd=repo_dir, env=None),
    ]


@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize("use_cache", [False, True])
def test_configure_failure_reports_cache_without_retry(
    tmp_path, capsys, verbose, use_cache
):
    command_runner = Mock()
    command_runner.run.return_value = CommandResult(1, "", "incompatible settings")
    cache = tmp_path / "config.cache" if use_cache else None
    error = ConfigureCacheError if use_cache else typer.Exit

    with pytest.raises(error) as exc:
        _run_configure(
            command_runner,
            tmp_path,
            tmp_path / "build",
            frozenset(),
            verbose,
            Mock(),
            TaskID(1),
            configure_cache=cache,
        )

    assert exc.value.exit_code == 1
    command_runner.run.assert_called_once()
    output = capsys.readouterr().out
    assert ("--no-configure-cache" in output) == use_cache
    if not verbose:
        assert "incompatible settings" in output


def test_bisect_reuses_cache_for_each_commit(cli_build, tmp_path):
    build, command_runner, _ = cli_build
    commits = iter([COMMIT, "def456abc123"])
    good_calls = 0

    def git_result(args, *positional, **kwargs):
        nonlocal good_calls
        if args == ["rev-parse", "HEAD"]:
            return CommandResult(0, next(commits), "")
        if args[:2] == ["bisect", "good"]:
            good_calls += 1
            if good_calls == 3:
                return CommandResult(0, f"{COMMIT} is the first bad commit", "")
        return CommandResult(0, "", "")

    command_runner.run_git.side_effect = git_result
    cache = tmp_path / "config.cache"
    result = invoke_build("bisect", ["--configure-cache", str(cache)])

    assert result.exit_code == 0, result.output
    assert build.call_args_list == [
        call(COMMIT, BuildOptions(configure_cache=cache)),
        call("def456abc123", BuildOptions(configure_cache=cache)),
    ]


def test_bisect_aborts_on_cached_configure_failure_and_resets(cli_build, tmp_path):
    build, command_runner, repo_dir = cli_build
    build.side_effect = ConfigureCacheError(1)

    result = invoke_build(
        "bisect", ["--configure-cache", str(tmp_path / "config.cache")]
    )

    assert result.exit_code == 1
    build.assert_called_once()
    assert not any(
        c.args[0] == ["bisect", "skip"] for c in command_runner.run_git.call_args_list
    )
    command_runner.run_git.assert_called_with(["bisect", "reset"], repo_dir)


def test_bisect_still_skips_other_build_failures(cli_build, tmp_path):
    build, command_runner, repo_dir = cli_build
    build.side_effect = [typer.Exit(1), build.return_value]

    result = invoke_build(
        "bisect", ["--configure-cache", str(tmp_path / "config.cache")]
    )

    assert result.exit_code == 0, result.output
    assert build.call_count == 2
    command_runner.run_git.assert_any_call(["bisect", "skip"], repo_dir, check=True)
