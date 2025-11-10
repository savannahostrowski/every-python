"""Command runner abstraction for testability."""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CommandResult:
    """Result of a command execution."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        """Check if command succeeded."""
        return self.returncode == 0

    @classmethod
    def from_subprocess(
        cls, result: subprocess.CompletedProcess[str]
    ) -> "CommandResult":
        """Create from subprocess result."""
        return cls(
            returncode=result.returncode,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
        )


class CommandRunner:
    """Encapsulates subprocess operations for testing and consistency."""

    def run(
        self,
        cmd: list[str],
        cwd: Path | None = None,
        capture_output: bool = True,
        check: bool = False,
        **kwargs: Any,
    ) -> CommandResult:
        """Run a command and return a normalized result."""
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=capture_output,
            text=True,
            check=check,
            **kwargs,
        )
        return CommandResult.from_subprocess(result)

    def run_git(
        self, args: list[str], repo_dir: Path, check: bool = False
    ) -> CommandResult:
        """Run a git command in the specified repo."""
        return self.run(["git", *args], cwd=repo_dir, check=check)


# Default instance for dependency injection
_default_runner: CommandRunner | None = None


def get_runner() -> CommandRunner:
    """Get the default command runner (can be overridden in tests)."""
    global _default_runner
    if _default_runner is None:
        _default_runner = CommandRunner()
    return _default_runner


def set_runner(runner: CommandRunner) -> None:
    """Set the command runner (for testing)."""
    global _default_runner
    _default_runner = runner
