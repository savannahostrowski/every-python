import multiprocessing
import os
import platform
import shlex
import shutil
import subprocess
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table
from typing_extensions import Annotated

from every_python import __version__
from every_python.output import create_progress, get_output, jit_indicator
from every_python.runner import CommandResult, CommandRunner, get_runner
from every_python.utils import (
    BUILD_FLAGS,
    BuildInfo,
    BuildVersion,
    check_llvm_available,
    get_llvm_version_for_commit,
    is_nogil_available_in_commit,
    python_binary_location,
)


def _flags_from_bools(jit: bool, pgo: bool, nogil: bool) -> frozenset[str]:
    flag_states: list[tuple[str, bool]] = [("jit", jit), ("pgo", pgo), ("nogil", nogil)]
    return frozenset(name for name, enabled in flag_states if enabled)


@dataclass(frozen=True)
class BuildOptions:
    """Options controlling a CPython build."""

    flags: frozenset[str] = frozenset()
    ccache: bool = False
    jobs: int | None = None
    verbose: bool = False


def _build_options(
    *,
    jit: bool,
    pgo: bool,
    nogil: bool,
    ccache: bool,
    jobs: int | None,
    verbose: bool = False,
) -> BuildOptions:
    """Create build options from CLI values."""
    return BuildOptions(
        flags=_flags_from_bools(jit, pgo, nogil),
        ccache=ccache,
        jobs=jobs,
        verbose=verbose,
    )


def _all_flag_combos() -> Iterator[frozenset[str]]:
    return (
        frozenset(c)
        for r in range(len(BUILD_FLAGS) + 1)
        for c in combinations(BUILD_FLAGS, r)
    )


app = typer.Typer()
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"every-python {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the every-python version and exit.",
        ),
    ] = False,
) -> None:
    """Build and run any commit of CPython."""


BASE_DIR = Path.home() / ".every-python"
REPO_DIR = BASE_DIR / "cpython"
BUILDS_DIR = BASE_DIR / "builds"
CPYTHON_REPO = "https://github.com/python/cpython.git"


def _ensure_repo() -> Path:
    """Ensure CPython repo exists as a blobless clone."""
    if not REPO_DIR.exists():
        runner: CommandRunner = get_runner()
        output = get_output()

        output.warning("First-time setup: cloning CPython repository...")
        output.info(
            "This will download ~200MB and only needs to happen once per version."
        )

        BASE_DIR.mkdir(parents=True, exist_ok=True)

        # Blobless clone to save space and time
        result: CommandResult = runner.run(
            ["git", "clone", "--filter=blob:none", CPYTHON_REPO, str(REPO_DIR)]
        )

        if not result.success:
            output.error(f"Failed to clone CPython: {result.stderr}")
            raise typer.Exit(1)

        output.success("Repository cloned successfully")

    return REPO_DIR


def _resolve_ref(ref: str) -> str:
    """Resolve a git ref (tag, branch, commit) to a full commit hash."""
    _ensure_repo()
    runner = get_runner()
    output = get_output()

    # Fetch latest refs
    runner.run_git(["fetch", "--tags", "--quiet"], REPO_DIR)

    # Try to resolve the ref
    result = runner.run_git(["rev-parse", "--verify", f"{ref}^{{commit}}"], REPO_DIR)

    if not result.success:
        # Try with origin/ prefix
        result = runner.run_git(
            ["rev-parse", "--verify", f"origin/{ref}^{{commit}}"], REPO_DIR
        )

        if not result.success:
            output.error(f"Could not resolve '{ref}' to a commit")
            output.info("Try: main, 3.13, v3.13.0, or a commit hash")
            raise typer.Exit(1)

    return result.stdout.strip()


def _show_llvm_install_instructions(llvm_version: str) -> None:
    """Show instructions to install LLVM on the current platform."""
    output = get_output()
    if platform.system() == "Darwin":
        output.info(f"Install with: brew install llvm@{llvm_version}")
    elif platform.system() == "Linux":
        output.info(
            f"Install with: apt install llvm-{llvm_version} clang-{llvm_version} lld-{llvm_version}"
        )
    else:
        output.info(
            f"Install LLVM {llvm_version} from https://github.com/llvm/llvm-project/releases"
        )


def _resolve_jit_availability(commit: str, repo_dir: Path) -> tuple[bool, str | None]:
    """Resolve whether JIT can be enabled for the given commit and which LLVM to use.

    Returns (enabled, llvm_version). llvm_version is None when JIT is disabled.
    """
    output = get_output()
    llvm_version = get_llvm_version_for_commit(commit, repo_dir)

    if not llvm_version:
        output.warning("Warning: JIT not available in this commit")
        if not typer.confirm("Continue building without JIT?", default=True):
            raise typer.Exit(0)
        return False, None

    if not check_llvm_available(llvm_version):
        output.warning(f"Warning: LLVM {llvm_version} not found")
        _show_llvm_install_instructions(llvm_version)
        if not typer.confirm("Continue building without JIT?", default=True):
            raise typer.Exit(0)
        return False, None

    return True, llvm_version


def _resolve_nogil_availability(commit: str, repo_dir: Path) -> bool:
    """Check if --disable-gil is supported at the given commit.

    Pre-3.13 commits silently accept the flag but produce a GIL-enabled build,
    so we refuse to mislabel the cache.
    """
    output = get_output()
    if not is_nogil_available_in_commit(commit, repo_dir):
        output.warning(
            "Warning: free-threading (--disable-gil) not available in this commit"
        )
        if not typer.confirm("Continue building with the GIL?", default=True):
            raise typer.Exit(0)
        return False
    return True


FLAG_TO_CONFIGURE_ARG_UNIX: dict[str, str] = {
    "jit": "--enable-experimental-jit",
    "pgo": "--enable-optimizations",
    "nogil": "--disable-gil",
}

FLAG_TO_CONFIGURE_ARG_WINDOWS: dict[str, str] = {
    "jit": "--experimental-jit",
    "pgo": "--pgo",
    "nogil": "--disable-gil",
}


def _get_configure_args(build_dir: Path, flags: frozenset[str]) -> list[str]:
    """Get platform-specific configure arguments."""
    if platform.system() == "Windows":
        plat = _windows_build_platform()
        args = ["cmd", "/c", "PCbuild\\build.bat", "-c", "Debug", "-p", plat]
        flag_map = FLAG_TO_CONFIGURE_ARG_WINDOWS
    else:
        args = ["./configure", "--prefix", str(build_dir), "--with-pydebug"]
        flag_map = FLAG_TO_CONFIGURE_ARG_UNIX

    # Iterate BUILD_FLAGS to keep argument order stable
    for flag in BUILD_FLAGS:
        if flag in flags:
            args.append(flag_map[flag])
    return args


def _windows_build_platform() -> str:
    """Map the host architecture to a PCbuild -p value."""
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "ARM64"
    if machine in ("x86", "i386", "i686"):
        return "Win32"
    return "x64"


def _windows_pcbuild_subdir(plat: str) -> str:
    """Map a PCbuild -p value to its output subdirectory."""
    return {"ARM64": "arm64", "Win32": "win32", "x64": "amd64"}[plat]


def _get_ccache_env() -> dict[str, str]:
    """Return a build environment that compiles through ccache."""
    output = get_output()
    if platform.system() == "Windows":
        output.error("--ccache is currently supported on macOS and Linux only")
        raise typer.Exit(1)

    ccache = shutil.which("ccache")
    if not ccache:
        output.error("ccache was not found in PATH")
        output.info(
            "Install it with: brew install ccache (macOS) or your Linux package manager"
        )
        raise typer.Exit(1)

    compiler = os.environ.get("CC") or (
        "clang" if platform.system() == "Darwin" else "cc"
    )
    compiler_command = shlex.split(compiler)
    if compiler_command and Path(compiler_command[0]).name == "ccache":
        cached_compiler = compiler
    else:
        cached_compiler = f"{ccache} {compiler}"
    return {**os.environ, "CC": cached_compiler}


def _run_clean_repo(
    runner: CommandRunner,
    verbose: bool,
    progress: Progress,
    task: TaskID,
) -> None:
    """Run the clean repo step."""
    output = get_output()
    args = ["clean", "-fdx"]

    if verbose:
        progress.stop()
        output.status(f"Running: git {' '.join(args)}")
    else:
        progress.update(task, description="Cleaning repo...")

    result = runner.run_git(args, repo_dir=REPO_DIR)

    if not result.success:
        if not verbose:
            progress.stop()
            output.error(f"Cleaning repo failed: {result.stderr}")
        else:
            output.error("Cleaning repo failed!")
        raise typer.Exit(1)


def _run_configure(
    runner: CommandRunner,
    build_dir: Path,
    flags: frozenset[str],
    verbose: bool,
    progress: Progress,
    task: TaskID,
    build_env: dict[str, str] | None = None,
) -> None:
    """Run the configure step."""
    output = get_output()
    configure_args = _get_configure_args(build_dir, flags)

    if verbose:
        progress.stop()
        output.status(f"Running: {' '.join(configure_args)}")
    else:
        progress.update(task, description="Configuring build...")

    result = runner.run(
        configure_args,
        cwd=REPO_DIR,
        capture_output=not verbose,
        env=build_env,
    )

    if not result.success:
        if not verbose:
            progress.stop()
            output.error(f"Configure failed: {result.stderr}")
        else:
            output.error("Configure failed")
        raise typer.Exit(1)


def _build_and_install_windows(
    build_dir: Path, verbose: bool, progress: Progress, task: TaskID
) -> None:
    """Build and install on Windows by copying PCbuild output."""
    output = get_output()
    progress.update(task, description="Copying build artifacts...")

    # Find build output directory matching the host architecture
    plat = _windows_build_platform()
    pcbuild_dir = REPO_DIR / "PCbuild" / _windows_pcbuild_subdir(plat)

    if not pcbuild_dir.exists():
        progress.stop()
        output.error(f"Build output not found at {pcbuild_dir}")
        raise typer.Exit(1)

    build_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        output.status(f"Copying from {pcbuild_dir} to {build_dir}")

    shutil.copytree(pcbuild_dir, build_dir, dirs_exist_ok=True)


def _build_and_install_unix(
    runner: CommandRunner,
    verbose: bool,
    progress: Progress,
    task: TaskID,
    build_env: dict[str, str] | None = None,
    jobs: int | None = None,
) -> None:
    """Build and install on Unix systems using make."""
    output = get_output()
    ncpu = jobs or multiprocessing.cpu_count()

    if verbose:
        output.status(f"Building with {ncpu} cores (this may take a few minutes)...")
        output.status(f"Running: make -j{ncpu}")
    else:
        progress.update(
            task,
            description=f"Building with {ncpu} cores (this may take a few minutes)...",
        )

    # Build
    make_result = runner.run(
        ["make", f"-j{ncpu}"],
        cwd=REPO_DIR,
        capture_output=not verbose,
        env=build_env,
    )

    if not make_result.success:
        if not verbose:
            progress.stop()
            output.error(f"Build failed: {make_result.stderr}")
        else:
            output.error("Build failed!")
        raise typer.Exit(1)

    # Install
    progress.update(task, description="Installing...")
    # CPython's install target is not parallel-safe: multiple rules can race to
    # create the same destination directory. Keep compilation parallel and
    # installation serial.
    install_result = runner.run(["make", "install"], cwd=REPO_DIR, env=build_env)

    if not install_result.success:
        progress.stop()
        output.error(f"Install failed: {install_result.stderr}")
        raise typer.Exit(1)


def build_python(
    commit: str,
    options: BuildOptions = BuildOptions(),
) -> Path:
    """Build Python at the given commit."""
    _ensure_repo()
    runner = get_runner()
    output = get_output()

    # Check JIT availability if requested
    llvm_version: str | None = None
    flags = set(options.flags)
    if "jit" in flags:
        jit_enabled, llvm_version = _resolve_jit_availability(commit, REPO_DIR)
        if not jit_enabled:
            flags.remove("jit")

    # Check free-threading availability if requested
    if "nogil" in flags and not _resolve_nogil_availability(commit, REPO_DIR):
        flags.remove("nogil")

    # Determine build directory based on final flags (after availability checks)
    build_info = BuildInfo(commit=commit, flags=frozenset(flags))
    build_dir = build_info.get_path(BUILDS_DIR)

    # Check if we have a complete cached build
    if build_dir.exists():
        python_bin = python_binary_location(BUILDS_DIR, build_info)
        if python_bin.exists():
            output.success(
                f"Build {commit[:7]}{build_info.suffix} already exists, skipping build"
            )
            return build_dir
        else:
            # Incomplete build - clean it up and rebuild
            output.warning(
                f"Incomplete build detected for {commit[:7]}{build_info.suffix}, cleaning and rebuilding..."
            )
            shutil.rmtree(build_dir)

    build_env = _get_ccache_env() if options.ccache else None
    if options.jobs is not None and platform.system() == "Windows":
        output.error(
            "--jobs is currently supported on macOS and Linux only; "
            "Windows builds are already parallel by default"
        )
        raise typer.Exit(1)

    if "jit" in flags:
        output.status(f"Building with JIT (LLVM {llvm_version})")
    if "pgo" in flags:
        output.status("Building with PGO")
    if "nogil" in flags:
        output.status("Building with GIL disabled (free-threaded)")
    if options.ccache:
        output.status("Building with ccache")

    with create_progress(console) as progress:
        # Checkout
        task = progress.add_task(f"Checking out {commit[:7]}...", total=None)
        result = runner.run_git(["checkout", commit], REPO_DIR)

        if not result.success:
            progress.stop()
            output.error(f"Failed to checkout {commit}: {result.stderr}")
            raise typer.Exit(1)

        # Clean repo
        _run_clean_repo(runner, options.verbose, progress, task)

        # Configure
        _run_configure(
            runner,
            build_dir,
            build_info.flags,
            options.verbose,
            progress,
            task,
            build_env,
        )

        # Build and install (platform-specific)
        if platform.system() == "Windows":
            _build_and_install_windows(build_dir, options.verbose, progress, task)
        else:
            _build_and_install_unix(
                runner,
                options.verbose,
                progress,
                task,
                build_env,
                options.jobs,
            )

        # Validate that the build produced a Python binary
        python_bin = python_binary_location(BUILDS_DIR, build_info)
        if not python_bin.exists():
            progress.stop()
            output.error(f"Build completed but Python binary not found at {python_bin}")
            raise typer.Exit(1)

        progress.update(task, description=f"[green]Built {commit[:7]}[/green]")

    return build_dir


@app.command()
def install(
    ref: Annotated[
        str,
        typer.Argument(help="Git ref to install (main, v3.13.0, commit hash, etc.)"),
    ],
    jit: Annotated[
        bool, typer.Option("--jit", help="Enable experimental JIT compiler")
    ] = False,
    pgo: Annotated[
        bool, typer.Option("--pgo", help="Enable optimized build (PGO + LTO)")
    ] = False,
    nogil: Annotated[
        bool, typer.Option("--nogil", help="Build with GIL disabled")
    ] = False,
    ccache: Annotated[
        bool, typer.Option("--ccache", help="Cache compilation results with ccache")
    ] = False,
    jobs: Annotated[
        int | None,
        typer.Option("--jobs", min=1, help="Number of parallel Unix build jobs"),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", help="Show build output")
    ] = False,
):
    """Build and install a specific CPython version."""
    output = get_output()
    try:
        commit = _resolve_ref(ref)
        output.info(f"Resolved '{ref}' to commit {commit[:7]}")

        options = _build_options(
            jit=jit,
            pgo=pgo,
            nogil=nogil,
            ccache=ccache,
            jobs=jobs,
            verbose=verbose,
        )
        build_dir = build_python(commit, options)

        output.success(f"\nSuccessfully built CPython {commit[:7]}")
        output.info(f"Location: {build_dir}")

        # Reflect what was actually enabled by inspecting the build directory name
        actual_flags = BuildInfo.from_directory(build_dir).flags

        run_example = f"every-python run {ref}"
        if "jit" in actual_flags:
            run_example += " --jit"
        if "pgo" in actual_flags:
            run_example += " --pgo"
        if "nogil" in actual_flags:
            run_example += " --nogil"

        run_example += " -- python --version"
        output.info(f"\nRun with: {run_example}")

    except subprocess.CalledProcessError as e:
        output.error(f"Command failed: {e}")
        raise typer.Exit(1)


@app.command()
def run(
    ref: Annotated[str, typer.Argument(help="Git ref to use")],
    command: Annotated[list[str], typer.Argument(help="Command to execute")],
    jit: Annotated[bool, typer.Option("--jit", help="Use JIT-enabled build")] = False,
    pgo: Annotated[
        bool, typer.Option("--pgo", help="Enable optimized build (PGO + LTO)")
    ] = False,
    nogil: Annotated[
        bool, typer.Option("--nogil", help="Use free-threaded build")
    ] = False,
    ccache: Annotated[
        bool,
        typer.Option("--ccache", help="Use ccache if an automatic build is needed"),
    ] = False,
    jobs: Annotated[
        int | None,
        typer.Option(
            "--jobs",
            min=1,
            help="Number of parallel Unix jobs if an automatic build is needed",
        ),
    ] = None,
):
    """Run a command with a specific Python version."""
    output = get_output()
    try:
        commit = _resolve_ref(ref)
        options = _build_options(
            jit=jit, pgo=pgo, nogil=nogil, ccache=ccache, jobs=jobs
        )
        build_info = BuildInfo(commit=commit, flags=options.flags)
        build_dir = build_info.get_path(BUILDS_DIR)

        if not build_dir.exists():
            output.warning(
                f"Build for {ref}{build_info.suffix} not found, building now..."
            )
            build_dir = build_python(commit, options)
            build_info = BuildInfo.from_directory(build_dir)

        python_bin = python_binary_location(BUILDS_DIR, build_info)

        if not python_bin.exists():
            output.error(f"Python binary not found at {python_bin}")
            raise typer.Exit(1)

        # If first argument is "python", replace it with the actual binary path
        # Otherwise, run python with the command as arguments
        if command and command[0] in ("python", "python3"):
            args = [str(python_bin)] + command[1:]
        else:
            args = [str(python_bin)] + command

        # Execute the command
        os.execv(str(python_bin), args)

    except subprocess.CalledProcessError as e:
        output.error(f"Command failed: {e}")
        raise typer.Exit(1)


@app.command()
def list_builds():
    """List locally built Python versions."""
    output = get_output()
    builds = list(BUILDS_DIR.iterdir()) if BUILDS_DIR.exists() else []
    if not builds:
        output.warning("No builds found.")
        output.info("Run every-python install main to build the latest version.")
        return

    # Starting an interpreter is relatively expensive, so probe cached builds
    # concurrently. ThreadPoolExecutor is appropriate here because the work is
    # spent waiting for subprocesses, not executing Python code.
    runner = get_runner()

    def get_build_version(build: Path) -> BuildVersion:
        build_info = BuildInfo.from_directory(build)
        python_bin = python_binary_location(BUILDS_DIR, build_info)

        if python_bin.exists():
            result = runner.run([str(python_bin), "--version"])
            version = result.stdout.strip() if result.success else "unknown"
        else:
            version = "unknown"

        return BuildVersion.from_build(build, version, build_info)

    with ThreadPoolExecutor() as executor:
        build_versions = list(executor.map(get_build_version, builds))

    # Parse versions once and sort
    build_versions.sort(
        key=lambda x: (
            x.major,
            x.minor,
            x.micro,
            x.suffix,
            "jit" not in x.build_info.flags,
        ),
        reverse=True,
    )

    table = Table(show_header=True, header_style="bold")
    table.add_column("Version", style="cyan")
    table.add_column("JIT", justify="center", width=4)
    table.add_column("Date", style="green")
    table.add_column("Commit", style="white", width=7)
    table.add_column("Message", style="dim", no_wrap=False)

    commits = [bv.build_info.commit for bv in build_versions]

    result = runner.run_git(
        ["log", "--format=%H|%at|%s", "--no-walk"] + commits, repo_dir=REPO_DIR
    )
    commit_info: dict[str, tuple[int, str]] = {}
    for line in result.stdout.strip().split("\n"):
        parts = line.split("|", 2)
        if len(parts) == 3:
            hash_val, timestamp_str, msg_val = parts
            commit_info[hash_val] = (int(timestamp_str), msg_val)

    for bv in build_versions:
        if bv.build_info.commit in commit_info:
            ts, msg = commit_info[bv.build_info.commit]
            timestamp = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
        else:
            timestamp = "unknown"
            msg = ""

        if bv.version_string != "unknown":
            jit_text = jit_indicator() if "jit" in bv.build_info.flags else ""
            table.add_row(
                bv.version_string.replace("Python ", ""),
                jit_text,
                timestamp,
                bv.build_info.commit[:7],
                msg,
            )
        else:
            table.add_row(
                "[red]incomplete[/red]",
                "",
                timestamp,
                bv.build_info.commit[:7],
                "",
            )

    console.print(table)


@app.command()
def clean(
    ref: Annotated[str | None, typer.Argument(help="Git ref to remove")] = None,
    all: Annotated[bool, typer.Option("--all", help="Remove all builds")] = False,
):
    """Remove built Python versions to free up space."""
    output = get_output()
    if all:
        if BUILDS_DIR.exists():
            shutil.rmtree(BUILDS_DIR)
            output.success("Removed all builds")
        else:
            output.warning("No builds to remove")
    elif ref:
        try:
            commit = _resolve_ref(ref)

            removed: list[str] = []
            for flags in _all_flag_combos():
                build_info = BuildInfo(commit=commit, flags=flags)
                build_dir = build_info.get_path(BUILDS_DIR)
                if build_dir.exists():
                    shutil.rmtree(build_dir)
                    label = (
                        "+".join(f.upper() for f in BUILD_FLAGS if f in flags) or "base"
                    )
                    removed.append(label)

            if removed:
                variants = ", ".join(removed)
                output.success(f"Removed {variants} build(s) for {commit[:7]}")
            else:
                output.warning(f"No builds found for {commit[:7]}")
        except typer.Exit:
            pass
    else:
        output.error("Specify a ref to remove or use --all")
        raise typer.Exit(1)


@app.command()
def bisect(
    good: Annotated[str, typer.Option("--good", help="Known good commit/ref")],
    bad: Annotated[str, typer.Option("--bad", help="Known bad commit/ref")],
    run: Annotated[
        str,
        typer.Option("--run", help="Command to run (exit 0 = good, non-zero = bad)"),
    ],
    jit: Annotated[
        bool, typer.Option("--jit", help="Enable experimental JIT compiler")
    ] = False,
    pgo: Annotated[
        bool, typer.Option("--pgo", help="Enable optimized build (PGO + LTO)")
    ] = False,
    nogil: Annotated[
        bool, typer.Option("--nogil", help="Build with GIL disabled")
    ] = False,
    ccache: Annotated[
        bool, typer.Option("--ccache", help="Cache compilation results with ccache")
    ] = False,
    jobs: Annotated[
        int | None,
        typer.Option("--jobs", min=1, help="Number of parallel Unix build jobs"),
    ] = None,
):
    """
    Use git bisect to find the commit that introduced a bug.

    The command should exit with code 0 if the commit is good,
    and non-zero if the commit is bad.

    Example:
        every-python bisect --good v3.13.0 --bad main --run "python test.py"
    """
    _ensure_repo()
    runner = get_runner()
    output = get_output()

    try:
        # Resolve refs to commits
        output.info(f"\nResolving good commit: {good}")
        good_commit = _resolve_ref(good)
        output.info(f"  → {good_commit[:7]}")

        output.info(f"Resolving bad commit: {bad}")
        bad_commit = _resolve_ref(bad)
        output.info(f"  → {bad_commit[:7]}")

        # Start bisect
        output.info("\n[bold]Starting git bisect...[/bold]")

        # Clean up any previous bisect state
        runner.run_git(["bisect", "reset"], REPO_DIR)

        # Reset any local changes in the repo
        runner.run_git(["reset", "--hard"], REPO_DIR)
        runner.run_git(["clean", "-fdx"], REPO_DIR)

        runner.run_git(["bisect", "start"], REPO_DIR, check=True)
        runner.run_git(["bisect", "bad", bad_commit], REPO_DIR, check=True)

        # Capture initial bisect output to show steps remaining
        initial_result = runner.run_git(
            ["bisect", "good", good_commit], REPO_DIR, check=True
        )

        # Extract and display steps remaining from git bisect output
        if "Bisecting:" in initial_result.stdout:
            # Example: "Bisecting: 3 revisions left to test after this (roughly 2 steps)"
            import re

            match = re.search(
                r"Bisecting: (\d+) revisions? left.*?\(roughly (\d+) steps?\)",
                initial_result.stdout,
            )
            if match:
                revisions = match.group(1)
                steps = match.group(2)
                output.info(
                    f"[dim]Bisecting: {revisions} revisions left to test (roughly {steps} steps)[/dim]"
                )

        def is_bisect_done() -> bool:
            """Check if bisect is complete by checking if BISECT_LOG exists."""
            bisect_log = REPO_DIR / ".git" / "BISECT_LOG"
            # Bisect is done when BISECT_LOG doesn't exist
            return not bisect_log.exists()

        iteration_count = 0
        max_iterations = 100  # Safety limit

        while not is_bisect_done() and iteration_count < max_iterations:
            iteration_count += 1
            # Get current commit being tested
            result = runner.run_git(["rev-parse", "HEAD"], REPO_DIR, check=True)
            current_commit = result.stdout.strip()

            output.status(
                f"\n[bold cyan]Testing commit {current_commit[:7]}...[/bold cyan]"
            )

            # Build this commit (build_python handles incomplete builds internally)
            try:
                options = _build_options(
                    jit=jit,
                    pgo=pgo,
                    nogil=nogil,
                    ccache=ccache,
                    jobs=jobs,
                )
                build_dir = build_python(current_commit, options)
                build_info = BuildInfo.from_directory(build_dir)
                python_bin = python_binary_location(BUILDS_DIR, build_info)
            except typer.Exit:
                # Build failed - skip this commit in bisect
                output.error("Build failed, skipping commit (exit 125)")
                runner.run_git(["bisect", "skip"], REPO_DIR, check=True)
                continue

            # Run the test command
            output.info(f"Running: {run}")
            # Note: using subprocess directly here since we need shell=True
            test_result_raw = subprocess.run(
                run,
                shell=True,
                cwd=Path.cwd(),
                env={**os.environ, "PYTHON": str(python_bin)},
            )
            test_result = CommandResult(
                returncode=test_result_raw.returncode,
                stdout="",
                stderr="",
            )

            if test_result.returncode == 0:
                output.success("Test passed (exit 0) - marking as good")
                bisect_result = runner.run_git(["bisect", "good"], REPO_DIR, check=True)
            elif test_result.returncode == 125:
                output.warning("Test requested skip (exit 125) - skipping commit")
                bisect_result = runner.run_git(["bisect", "skip"], REPO_DIR, check=True)
            elif 1 <= test_result.returncode < 128:
                output.error(
                    f"✗ Test failed (exit {test_result.returncode}) - marking as bad"
                )
                bisect_result = runner.run_git(["bisect", "bad"], REPO_DIR, check=True)
            else:
                output.error(f"Test exited with code {test_result.returncode} >= 128")
                raise typer.Exit(1)

            # Check if bisect completed
            if "is the first bad commit" in bisect_result.stdout:
                break

            # Show steps remaining after each bisect step
            if "Bisecting:" in bisect_result.stdout:
                import re

                match = re.search(
                    r"Bisecting: (\d+) revisions? left.*?\(roughly (\d+) steps?\)",
                    bisect_result.stdout,
                )
                if match:
                    revisions = match.group(1)
                    steps = match.group(2)
                    output.info(
                        f"[dim]→ {revisions} revisions left (roughly {steps} steps)[/dim]"
                    )

        # Show final result
        result = runner.run_git(["bisect", "log"], REPO_DIR)

        output.success("\n[bold green]Bisect complete![/bold green]")
        # Extract and show the first bad commit from the log
        for line in result.stdout.splitlines():
            if line.startswith("# first bad commit:"):
                commit_hash = (
                    line.split("[")[1].split("]")[0] if "[" in line else "unknown"
                )
                output.info(f"\nFirst bad commit: [bold]{commit_hash}[/bold]")

                # Show commit details
                commit_result = runner.run_git(
                    [
                        "show",
                        "--no-patch",
                        "--format=%H%n%an <%ae>%n%ad%n%s",
                        commit_hash,
                    ],
                    REPO_DIR,
                )
                if commit_result.success:
                    output.info(commit_result.stdout)
                break

    except subprocess.CalledProcessError as e:
        output.error(f"Bisect failed: {e}")
        raise typer.Exit(1)
    finally:
        # Clean up bisect state
        runner.run_git(["bisect", "reset"], REPO_DIR)
