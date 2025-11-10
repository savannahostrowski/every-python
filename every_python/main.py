import os
import shutil
import subprocess
from pathlib import Path
from typing_extensions import Annotated
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from every_python.utils import check_llvm_available, get_llvm_version_for_commit

app = typer.Typer()
console = Console()

BASE_DIR = Path.home() / ".every-python"
REPO_DIR = BASE_DIR / "cpython"
BUILDS_DIR = BASE_DIR / "builds"
CPYTHON_REPO = "https://github.com/python/cpython.git"


def ensure_repo() -> Path:
    """Ensure CPython repo exists as a blobless clone."""
    if not REPO_DIR.exists():
        console.print(
            "[yellow]First-time setup: cloning CPython repository...[/yellow]"
        )
        console.print(
            "This will download ~200MB and only needs to happen once per version."
        )

        BASE_DIR.mkdir(parents=True, exist_ok=True)

        # Blobless clone to save space and time
        result = subprocess.run(
            ["git", "clone", "--filter=blob:none", CPYTHON_REPO, str(REPO_DIR)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            console.print(f"[red]Failed to clone CPython: {result.stderr}[/red]")
            raise typer.Exit(1)

        console.print("[green]✓ Repository cloned successfully[/green]")

    return REPO_DIR


def resolve_ref(ref: str) -> str:
    """Resolve a git ref (tag, branch, commit) to a full commit hash."""
    ensure_repo()

    # Fetch latest refs
    subprocess.run(
        ["git", "fetch", "--tags", "--quiet"],
        cwd=REPO_DIR,
        capture_output=True,
    )

    # Try to resolve the ref
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # Try with origin/ prefix
        result = subprocess.run(
            ["git", "rev-parse", "--verify", f"origin/{ref}^{{commit}}"],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            console.print(f"[red]Could not resolve '{ref}' to a commit[/red]")
            console.print("Try: main, 3.13, v3.13.0, or a commit hash")
            raise typer.Exit(1)

    return result.stdout.strip()


def build_python(commit: str, enable_jit: bool = False, verbose: bool = False) -> Path:
    """Build Python at the given commit."""
    ensure_repo()

    # Check JIT availability if requested
    if enable_jit:
        llvm_version = get_llvm_version_for_commit(commit, REPO_DIR)

        if not llvm_version:
            console.print("[yellow]Warning: JIT not available in this commit[/yellow]")
            if not typer.confirm("Continue building without JIT?", default=True):
                raise typer.Exit(0)
            enable_jit = False
        elif not check_llvm_available(llvm_version):
            console.print(f"[yellow]Warning: LLVM {llvm_version} not found[/yellow]")
            console.print(f"Install with: brew install llvm@{llvm_version}")
            if not typer.confirm("Continue building without JIT?", default=True):
                raise typer.Exit(0)
            enable_jit = False
        else:
            console.print(f"[cyan]Building with JIT (LLVM {llvm_version})[/cyan]")

    # Determine build directory based on final JIT flag (after availability checks)
    build_suffix = "-jit" if enable_jit else ""
    build_dir = BUILDS_DIR / f"{commit}{build_suffix}"

    if build_dir.exists():
        console.print(
            f"[green]Build {commit[:7]}{build_suffix} already exists, skipping build[/green]"
        )
        return build_dir

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Checkout the commit
        task = progress.add_task(f"Checking out {commit[:7]}...", total=None)
        result = subprocess.run(
            ["git", "checkout", commit],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            progress.stop()
            console.print(f"[red]Failed to checkout {commit}: {result.stderr}[/red]")
            raise typer.Exit(1)

        # Configure
        progress.update(task, description="Configuring build...")
        configure_args = ["./configure", "--prefix", str(build_dir), "--with-pydebug"]

        # Add JIT flag if enabled
        if enable_jit:
            configure_args.append("--enable-experimental-jit")

        if verbose:
            progress.stop()
            console.print(f"[cyan]Running: {' '.join(configure_args)}[/cyan]")

        configure_result = subprocess.run(
            configure_args,
            cwd=REPO_DIR,
            capture_output=not verbose,
            text=True,
        )

        if configure_result.returncode != 0:
            if not verbose:
                progress.stop()
            console.print(
                f"[red]Configure failed: {configure_result.stderr if not verbose else ''}[/red]"
            )
            raise typer.Exit(1)

        # Build
        import multiprocessing

        ncpu = multiprocessing.cpu_count()

        if verbose:
            console.print(
                f"[cyan]Building with {ncpu} cores (this may a few minutes)...[/cyan]"
            )
            console.print(f"[cyan]Running: make -j{ncpu}[/cyan]")
        else:
            progress.update(
                task,
                description=f"Building with {ncpu} cores (this may a few minutes)...",
            )

        make_result = subprocess.run(
            ["make", f"-j{ncpu}"],
            cwd=REPO_DIR,
            capture_output=not verbose,
            text=True,
        )

        if make_result.returncode != 0:
            if not verbose:
                progress.stop()
            console.print(
                f"[red]Build failed: {make_result.stderr if not verbose else ''}[/red]"
            )
            raise typer.Exit(1)

        # Install to prefix
        progress.update(task, description="Installing...")
        install_result = subprocess.run(
            ["make", "install"],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
        )

        if install_result.returncode != 0:
            progress.stop()
            console.print(f"[red]Install failed: {install_result.stderr}[/red]")
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
    try:
        commit = resolve_ref(ref)
        console.print(f"Resolved '{ref}' to commit {commit[:7]}")

        build_dir = build_python(commit, enable_jit=jit, verbose=verbose)

        console.print(
            f"\n[bold green]Successfully built CPython {commit[:7]}[/bold green]"
        )
        console.print(f"Location: {build_dir}")

        # Check if JIT was actually enabled (by checking the build directory name)
        actual_jit = build_dir.name.endswith("-jit")

        run_example = f"every-python run {ref}"
        if actual_jit:
            run_example += " --jit"

        run_example += " -- python --version"
        console.print(f"\nRun with: [bold]{run_example}[/bold]")

    except subprocess.CalledProcessError as e:
        console.print(f"[red]Command failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def run(
    ref: Annotated[str, typer.Argument(help="Git ref to use")],
    command: Annotated[list[str], typer.Argument(help="Command to execute")],
    jit: Annotated[bool, typer.Option("--jit", help="Use JIT-enabled build")] = False,
):
    """Run a command with a specific Python version."""
    try:
        commit = resolve_ref(ref)
        build_suffix = "-jit" if jit else ""
        build_dir = BUILDS_DIR / f"{commit}{build_suffix}"

        if not build_dir.exists():
            console.print(
                f"[yellow]Build for {ref}{build_suffix} not found, building now...[/yellow]"
            )
            build_dir = build_python(commit, enable_jit=jit)

        python_bin = build_dir / "bin" / "python3"

        if not python_bin.exists():
            console.print(f"[red]Python binary not found at {python_bin}[/red]")
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
        console.print(f"[red]Command failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list_builds():
    """List locally built Python versions."""
    if not BUILDS_DIR.exists() or not list(BUILDS_DIR.iterdir()):
        console.print("[yellow]No builds found.[/yellow]")
        console.print(
            "Run [bold]every-python install main[/bold] to build the latest version."
        )
        return

    # Get version info for all builds
    builds_with_version: list[tuple[Path, str, bool]] = []
    for build in BUILDS_DIR.iterdir():
        is_jit = build.name.endswith("-jit")
        python_bin = build / "bin" / "python3"

        if python_bin.exists():
            result = subprocess.run(
                [str(python_bin), "--version"],
                capture_output=True,
                text=True,
            )
            version = result.stdout.strip() if result.returncode == 0 else "unknown"
        else:
            version = "unknown"

        builds_with_version.append((build, version, is_jit))

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
            x[2],
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

    for build, version, is_jit in builds_with_version:
        commit = build.name.replace("-jit", "") if is_jit else build.name

        # Get commit timestamp and message
        commit_info_result = subprocess.run(
            ["git", "log", "-1", "--format=%at|%s", commit],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
        )

        if commit_info_result.returncode == 0 and commit_info_result.stdout.strip():
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
            jit_text = "✓" if is_jit else ""
            table.add_row(
                version.replace("Python ", ""),
                jit_text,
                timestamp,
                commit[:7],
                commit_msg,
            )
        else:
            table.add_row(
                "[red]incomplete[/red]",
                "",
                timestamp,
                commit[:7],
                "",
            )

    console.print(table)


@app.command()
def clean(
    ref: Annotated[str | None, typer.Argument(help="Git ref to remove")] = None,
    all: Annotated[bool, typer.Option("--all", help="Remove all builds")] = False,
):
    """Remove built Python versions to free up space."""
    if all:
        if BUILDS_DIR.exists():
            shutil.rmtree(BUILDS_DIR)
            console.print("[green]✓ Removed all builds[/green]")
        else:
            console.print("[yellow]No builds to remove[/yellow]")
    elif ref:
        try:
            commit = resolve_ref(ref)

            # Check for both JIT and non-JIT builds
            removed: list[str] = []
            for suffix in ["", "-jit"]:
                build_dir = BUILDS_DIR / f"{commit}{suffix}"
                if build_dir.exists():
                    shutil.rmtree(build_dir)
                    removed.append("-jit" if suffix == "-jit" else "non-JIT")

            if removed:
                variants = " and ".join(removed)
                console.print(
                    f"[green]✓ Removed {variants} build(s) for {commit[:7]}[/green]"
                )
            else:
                console.print(f"[yellow]No builds found for {commit[:7]}[/yellow]")
        except typer.Exit:
            pass
    else:
        console.print("[red]Specify a ref to remove or use --all[/red]")
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

    try:
        # Resolve refs to commits
        console.print(f"\nResolving good commit: {good}")
        good_commit = resolve_ref(good)
        console.print(f"  → {good_commit[:7]}")

        console.print(f"Resolving bad commit: {bad}")
        bad_commit = resolve_ref(bad)
        console.print(f"  → {bad_commit[:7]}")

        # Start bisect
        console.print("\n[bold]Starting git bisect...[/bold]")

        # Clean up any previous bisect state
        subprocess.run(["git", "bisect", "reset"], cwd=REPO_DIR, capture_output=True)

        # Reset any local changes in the repo
        subprocess.run(["git", "reset", "--hard"], cwd=REPO_DIR, capture_output=True)
        subprocess.run(["git", "clean", "-fd"], cwd=REPO_DIR, capture_output=True)

        subprocess.run(["git", "bisect", "start"], cwd=REPO_DIR, check=True)
        subprocess.run(["git", "bisect", "bad", bad_commit], cwd=REPO_DIR, check=True)

        # Capture initial bisect output to show steps remaining
        initial_result = subprocess.run(
            ["git", "bisect", "good", good_commit],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            check=True,
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
                console.print(
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
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_DIR,
                capture_output=True,
                text=True,
                check=True,
            )
            current_commit = result.stdout.strip()

            console.print(
                f"\n[bold cyan]Testing commit {current_commit[:7]}...[/bold cyan]"
            )

            # Build this commit
            try:
                build_dir = build_python(current_commit, enable_jit=jit)
                python_bin = build_dir / "bin" / "python3"

                if not python_bin.exists():
                    # Build directory exists but python binary is missing - incomplete build
                    console.print(
                        "[yellow]Incomplete build detected, cleaning and rebuilding...[/yellow]"
                    )
                    shutil.rmtree(build_dir)

                    # Retry build
                    try:
                        build_dir = build_python(current_commit, enable_jit=jit)
                        python_bin = build_dir / "bin" / "python3"

                        if not python_bin.exists():
                            console.print(
                                "[red]Build failed after retry, skipping commit (exit 125)[/red]"
                            )
                            subprocess.run(
                                ["git", "bisect", "skip"], cwd=REPO_DIR, check=True
                            )
                            continue
                    except Exception:
                        console.print(
                            "[red]Build failed, skipping commit (exit 125)[/red]"
                        )
                        subprocess.run(
                            ["git", "bisect", "skip"], cwd=REPO_DIR, check=True
                        )
                        continue

                # Run the test command
                console.print(f"Running: {run}")
                test_result = subprocess.run(
                    run,
                    shell=True,
                    cwd=Path.cwd(),
                    env={**os.environ, "PYTHON": str(python_bin)},
                )

                # Handle exit codes like every-ts
                if test_result.returncode == 0:
                    console.print(
                        "[green]✓ Test passed (exit 0) - marking as good[/green]"
                    )
                    bisect_result = subprocess.run(
                        ["git", "bisect", "good"],
                        cwd=REPO_DIR,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                elif test_result.returncode == 125:
                    console.print(
                        "[yellow]Test requested skip (exit 125) - skipping commit[/yellow]"
                    )
                    bisect_result = subprocess.run(
                        ["git", "bisect", "skip"],
                        cwd=REPO_DIR,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                elif 1 <= test_result.returncode < 128:
                    console.print(
                        f"[red]✗ Test failed (exit {test_result.returncode}) - marking as bad[/red]"
                    )
                    bisect_result = subprocess.run(
                        ["git", "bisect", "bad"],
                        cwd=REPO_DIR,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                else:
                    console.print(
                        f"[red]Test exited with code {test_result.returncode} >= 128[/red]"
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
                        console.print(
                            f"[dim]→ {revisions} revisions left (roughly {steps} steps)[/dim]"
                        )

            except Exception as e:
                console.print(f"[red]Error during bisect: {e}[/red]")
                console.print("Skipping commit...")
                subprocess.run(["git", "bisect", "skip"], cwd=REPO_DIR, check=True)

        # Show final result
        result = subprocess.run(
            ["git", "bisect", "log"],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
        )

        console.print("\n[bold green]Bisect complete![/bold green]")
        # Extract and show the first bad commit from the log
        for line in result.stdout.splitlines():
            if line.startswith("# first bad commit:"):
                commit_hash = (
                    line.split("[")[1].split("]")[0] if "[" in line else "unknown"
                )
                console.print(f"\nFirst bad commit: [bold]{commit_hash}[/bold]")

                # Show commit details
                commit_result = subprocess.run(
                    [
                        "git",
                        "show",
                        "--no-patch",
                        "--format=%H%n%an <%ae>%n%ad%n%s",
                        commit_hash,
                    ],
                    cwd=REPO_DIR,
                    capture_output=True,
                    text=True,
                )
                if commit_result.returncode == 0:
                    console.print(commit_result.stdout)
                break

    except subprocess.CalledProcessError as e:
        console.print(f"[red]Bisect failed: {e}[/red]")
        raise typer.Exit(1)
    finally:
        # Clean up bisect state
        subprocess.run(["git", "bisect", "reset"], cwd=REPO_DIR, capture_output=True)
