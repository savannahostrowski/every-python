import os
import platform
import shutil
import subprocess
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from typing_extensions import Annotated

from every_python.output import get_output
from every_python.runner import CommandResult, CommandRunner, get_runner
from every_python.utils import (
    BuildInfo,
    python_binary_location,
    check_llvm_available,
    get_llvm_version_for_commit,
)

app = typer.Typer()
console = Console()

BASE_DIR = Path.home() / ".every-python"
REPO_DIR = BASE_DIR / "cpython"
BUILDS_DIR = BASE_DIR / "builds"
CPYTHON_REPO = "https://github.com/python/cpython.git"


def ensure_repo() -> Path:
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

        output.success("✓ Repository cloned successfully")

    return REPO_DIR


def resolve_ref(ref: str) -> str:
    """Resolve a git ref (tag, branch, commit) to a full commit hash."""
    ensure_repo()
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


def build_python(commit: str, enable_jit: bool = False, verbose: bool = False) -> Path:
    """Build Python at the given commit."""
    ensure_repo()
    runner = get_runner()
    output = get_output()

    # Check JIT availability if requested
    if enable_jit:
        llvm_version = get_llvm_version_for_commit(commit, REPO_DIR)

        if not llvm_version:
            output.warning("Warning: JIT not available in this commit")
            if not typer.confirm("Continue building without JIT?", default=True):
                raise typer.Exit(0)
            enable_jit = False
        elif not check_llvm_available(llvm_version):
            output.warning(f"Warning: LLVM {llvm_version} not found")
            if platform.system() == "Darwin":
                output.info(f"Install with: brew install llvm@{llvm_version}")
            elif platform.system() == "Linux":
                output.info(
                    f"Install with: apt install llvm-{llvm_version} clang-{llvm_version} lld-{llvm_version}"
                )
            else:  # Windows
                output.info(
                    f"Install LLVM {llvm_version} from https://github.com/llvm/llvm-project/releases"
                )
            if not typer.confirm("Continue building without JIT?", default=True):
                raise typer.Exit(0)
            enable_jit = False
        else:
            output.status(f"Building with JIT (LLVM {llvm_version})")

    # Determine build directory based on final JIT flag (after availability checks)
    build_info = BuildInfo(commit=commit, jit_enabled=enable_jit)
    build_dir = build_info.get_path(BUILDS_DIR)

    if build_dir.exists():
        output.success(
            f"Build {commit[:7]}{build_info.suffix} already exists, skipping build"
        )
        return build_dir

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Checkout the commit
        task = progress.add_task(f"Checking out {commit[:7]}...", total=None)
        result = runner.run_git(["checkout", commit], REPO_DIR)

        if not result.success:
            progress.stop()
            output.error(f"Failed to checkout {commit}: {result.stderr}")
            raise typer.Exit(1)

        # Configure
        progress.update(task, description="Configuring build...")

        if platform.system() == "Windows":
            configure_args = ["PCbuild\\build.bat", "-c", "Debug"]
            if enable_jit:
                configure_args.append("--experimental-jit")
        else:
            configure_args = [
                "./configure",
                "--prefix",
                str(build_dir),
                "--with-pydebug",
            ]

            # Add JIT flag if enabled
            if enable_jit:
                configure_args.append("--experimental-jit")
        if verbose:
            progress.stop()
            output.status(f"Running: {' '.join(configure_args)}")

        configure_result = runner.run(
            configure_args,
            cwd=REPO_DIR,
            capture_output=not verbose,
        )

        if not configure_result.success:
            if not verbose:
                progress.stop()
            output.error(
                f"Configure failed: {configure_result.stderr if not verbose else ''}"
            )
            raise typer.Exit(1)

        # Build and install
        import multiprocessing

        ncpu = multiprocessing.cpu_count()

        if platform.system() == "Windows":
            # Windows: build.bat does both build and "install" (outputs to PCbuild/amd64)
            # The configure step above already ran build.bat, so we're done
            # Just move the output to our build directory
            progress.update(task, description="Copying build artifacts...")
            import shutil

            pcbuild_dir = REPO_DIR / "PCbuild" / "amd64"
            if not pcbuild_dir.exists():
                progress.stop()
                output.error(f"Build output not found at {pcbuild_dir}")
                raise typer.Exit(1)

            build_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(pcbuild_dir, build_dir, dirs_exist_ok=True)
        else:
            # Unix: use make
            if verbose:
                output.status(f"Building with {ncpu} cores (this may a few minutes)...")
                output.status(f"Running: make -j{ncpu}")
            else:
                progress.update(
                    task,
                    description=f"Building with {ncpu} cores (this may a few minutes)...",
                )

            make_result = runner.run(
                ["make", f"-j{ncpu}"],
                cwd=REPO_DIR,
                capture_output=not verbose,
            )

            if not make_result.success:
                if not verbose:
                    progress.stop()
                output.error(
                    f"Build failed: {make_result.stderr if not verbose else ''}"
                )
                raise typer.Exit(1)

            # Install to prefix
            progress.update(task, description="Installing...")
            install_result: CommandResult = runner.run(
                ["make", "install"],
                cwd=REPO_DIR,
            )

            if not install_result.success:
                progress.stop()
                output.error(f"Install failed: {install_result.stderr}")
                raise typer.Exit(1)

        progress.update(task, description=f"[green]✓ Built {commit[:7]}[/green]")

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
    verbose: Annotated[
        bool, typer.Option("--verbose", help="Show build output")
    ] = False,
):
    """Build and install a specific CPython version."""
    output = get_output()
    try:
        commit = resolve_ref(ref)
        output.info(f"Resolved '{ref}' to commit {commit[:7]}")

        build_dir = build_python(commit, enable_jit=jit, verbose=verbose)

        output.success(f"\nSuccessfully built CPython {commit[:7]}")
        output.info(f"Location: {build_dir}")

        # Check if JIT was actually enabled (by checking the build directory name)
        actual_jit = build_dir.name.endswith("-jit")

        run_example = f"every-python run {ref}"
        if actual_jit:
            run_example += " --jit"

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
):
    """Run a command with a specific Python version."""
    output = get_output()
    try:
        commit = resolve_ref(ref)
        build_info = BuildInfo(commit=commit, jit_enabled=jit)
        build_dir = build_info.get_path(BUILDS_DIR)

        if not build_dir.exists():
            output.warning(
                f"Build for {ref}{build_info.suffix} not found, building now..."
            )
            build_dir = build_python(commit, enable_jit=jit)

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
    if not BUILDS_DIR.exists() or not list(BUILDS_DIR.iterdir()):
        output.warning("No builds found.")
        output.info("Run every-python install main to build the latest version.")
        return

    # Get version info for all builds
    runner = get_runner()
    builds_with_version: list[tuple[Path, str, BuildInfo]] = []
    for build in BUILDS_DIR.iterdir():
        build_info = BuildInfo.from_directory(build)
        python_bin = build / "bin" / "python3"

        if python_bin.exists():
            result = runner.run([str(python_bin), "--version"])
            version = result.stdout.strip() if result.success else "unknown"
        else:
            version = "unknown"

        builds_with_version.append((build, version, build_info))

    # Sort by version (descending), then by JIT status (non-JIT first)
    def parse_version(version_str: str) -> tuple[int, int, int, str]:
        """Parse version string into sortable tuple."""
        if version_str == "unknown":
            return (0, 0, 0, "")

        # Extract "Python X.Y.Z" or "Python X.Y.Za1+"
        import re

        match = re.search(r"Python (\d+)\.(\d+)\.(\d+)([a-z0-9+]*)", version_str)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2))
            micro = int(match.group(3))
            suffix = match.group(4)
            return (major, minor, micro, suffix)
        return (0, 0, 0, "")

    builds_with_version.sort(
        key=lambda x: (
            -parse_version(x[1])[0],
            -parse_version(x[1])[1],
            -parse_version(x[1])[2],
            parse_version(x[1])[3],
            x[2].jit_enabled,
        )
    )

    # Create Rich table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Version", style="cyan")
    table.add_column("JIT", justify="center", width=4)
    table.add_column("Date", style="green")
    table.add_column("Commit", style="white", width=7)
    table.add_column("Message", style="dim", no_wrap=False)

    from datetime import datetime

    for build, version, build_info in builds_with_version:
        # Get commit timestamp and message
        commit_info_result = runner.run_git(
            ["log", "-1", "--format=%at|%s", build_info.commit],
            REPO_DIR,
        )

        if commit_info_result.success and commit_info_result.stdout.strip():
            parts = commit_info_result.stdout.strip().split("|", 1)
            commit_timestamp = int(parts[0])
            commit_msg = parts[1] if len(parts) > 1 else ""
            timestamp = datetime.fromtimestamp(commit_timestamp).strftime(
                "%Y-%m-%d %H:%M"
            )
        else:
            timestamp = "unknown"
            commit_msg = ""

        if version != "unknown":
            jit_text = "✓" if build_info.jit_enabled else ""
            table.add_row(
                version.replace("Python ", ""),
                jit_text,
                timestamp,
                build_info.commit[:7],
                commit_msg,
            )
        else:
            table.add_row(
                "[red]incomplete[/red]",
                "",
                timestamp,
                build_info.commit[:7],
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
            output.success("✓ Removed all builds")
        else:
            output.warning("No builds to remove")
    elif ref:
        try:
            commit = resolve_ref(ref)

            # Check for both JIT and non-JIT builds
            removed: list[str] = []
            for jit_enabled in [False, True]:
                build_info = BuildInfo(commit=commit, jit_enabled=jit_enabled)
                build_dir = build_info.get_path(BUILDS_DIR)
                if build_dir.exists():
                    shutil.rmtree(build_dir)
                    removed.append("JIT" if jit_enabled else "non-JIT")

            if removed:
                variants = " and ".join(removed)
                output.success(f"✓ Removed {variants} build(s) for {commit[:7]}")
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
):
    """
    Use git bisect to find the commit that introduced a bug.

    The command should exit with code 0 if the commit is good,
    and non-zero if the commit is bad.

    Example:
        every-python bisect --good v3.13.0 --bad main --run "python test.py"
    """
    ensure_repo()
    runner = get_runner()
    output = get_output()

    try:
        # Resolve refs to commits
        output.info(f"\nResolving good commit: {good}")
        good_commit = resolve_ref(good)
        output.info(f"  → {good_commit[:7]}")

        output.info(f"Resolving bad commit: {bad}")
        bad_commit = resolve_ref(bad)
        output.info(f"  → {bad_commit[:7]}")

        # Start bisect
        output.info("\n[bold]Starting git bisect...[/bold]")

        # Clean up any previous bisect state
        runner.run_git(["bisect", "reset"], REPO_DIR)

        # Reset any local changes in the repo
        runner.run_git(["reset", "--hard"], REPO_DIR)
        runner.run_git(["clean", "-fd"], REPO_DIR)

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

            # Build this commit
            try:
                build_dir = build_python(current_commit, enable_jit=jit)
                python_bin = build_dir / "bin" / "python3"

                if not python_bin.exists():
                    # Build directory exists but python binary is missing - incomplete build
                    output.warning(
                        "Incomplete build detected, cleaning and rebuilding..."
                    )
                    shutil.rmtree(build_dir)

                    # Retry build
                    try:
                        build_dir = build_python(current_commit, enable_jit=jit)
                        python_bin = build_dir / "bin" / "python3"

                        if not python_bin.exists():
                            output.error(
                                "Build failed after retry, skipping commit (exit 125)"
                            )
                            runner.run_git(["bisect", "skip"], REPO_DIR, check=True)
                            continue
                    except Exception:
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

                # Handle exit codes like every-ts
                if test_result.returncode == 0:
                    output.success("✓ Test passed (exit 0) - marking as good")
                    bisect_result = runner.run_git(
                        ["bisect", "good"], REPO_DIR, check=True
                    )
                elif test_result.returncode == 125:
                    output.warning("Test requested skip (exit 125) - skipping commit")
                    bisect_result = runner.run_git(
                        ["bisect", "skip"], REPO_DIR, check=True
                    )
                elif 1 <= test_result.returncode < 128:
                    output.error(
                        f"✗ Test failed (exit {test_result.returncode}) - marking as bad"
                    )
                    bisect_result = runner.run_git(
                        ["bisect", "bad"], REPO_DIR, check=True
                    )
                else:
                    output.error(
                        f"Test exited with code {test_result.returncode} >= 128"
                    )
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

            except Exception as e:
                output.error(f"Error during bisect: {e}")
                output.info("Skipping commit...")
                runner.run_git(["bisect", "skip"], REPO_DIR, check=True)

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
