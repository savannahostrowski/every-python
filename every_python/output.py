"""Output handler abstraction for testability and flexibility."""

import os
import sys
from abc import ABC, abstractmethod

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn


class OutputHandler(ABC):
    """Abstract output handler for user messages."""

    @abstractmethod
    def info(self, message: str) -> None:
        """Print informational message."""
        pass

    @abstractmethod
    def success(self, message: str) -> None:
        """Print success message."""
        pass

    @abstractmethod
    def warning(self, message: str) -> None:
        """Print warning message."""
        pass

    @abstractmethod
    def error(self, message: str) -> None:
        """Print error message."""
        pass

    @abstractmethod
    def status(self, message: str) -> None:
        """Print status/progress message."""
        pass


class RichOutputHandler(OutputHandler):
    """Rich console output handler."""

    def __init__(self, console: Console | None = None, use_unicode: bool = True):
        self.console = console or Console()
        self.use_unicode = use_unicode

    def _format_success(self, message: str) -> str:
        """Format success message with optional checkmark."""
        if self.use_unicode:
            return f"✓ {message}"
        return message

    def info(self, message: str) -> None:
        self.console.print(message)

    def success(self, message: str) -> None:
        self.console.print(f"[green]{self._format_success(message)}[/green]")

    def warning(self, message: str) -> None:
        self.console.print(f"[yellow]{message}[/yellow]")

    def error(self, message: str) -> None:
        self.console.print(f"[red]{message}[/red]")

    def status(self, message: str) -> None:
        self.console.print(f"[cyan]{message}[/cyan]")


class QuietOutputHandler(OutputHandler):
    """Silent output handler for testing."""

    def __init__(self):
        self.messages: list[tuple[str, str]] = []

    def info(self, message: str) -> None:
        self.messages.append(("info", message))

    def success(self, message: str) -> None:
        self.messages.append(("success", message))

    def warning(self, message: str) -> None:
        self.messages.append(("warning", message))

    def error(self, message: str) -> None:
        self.messages.append(("error", message))

    def status(self, message: str) -> None:
        self.messages.append(("status", message))


# Default instance for dependency injection
_default_output: OutputHandler | None = None


# Helpers for determining output capabilities
def _should_use_ascii() -> bool:
    """Check if we should use ASCII-only output (no Unicode)."""
    # Check if we're in CI with limited encoding support
    if os.getenv("CI") and sys.stdout.encoding in ("cp1252", "ascii"):
        return True
    return False


def create_progress(console: Console) -> Progress:
    """Create a Progress instance with appropriate settings for the environment."""
    if _should_use_ascii():
        # Use simple text-only progress without spinners for ASCII-only environments
        return Progress(
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        )
    else:
        # Use fancy spinners for Unicode-capable environments
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        )


def jit_indicator() -> str:
    """Get the JIT indicator based on environment."""
    if _should_use_ascii():
        return "[JIT]"
    else:
        return "✓"


def get_output() -> OutputHandler:
    """Get the default output handler."""
    global _default_output
    if _default_output is None:
        # Force ASCII-safe output in CI environments with limited encoding
        force_ascii = _should_use_ascii()
        console = (
            Console(
                legacy_windows=False,  # Disable legacy Windows rendering
                force_terminal=not force_ascii,  # Disable terminal features if ASCII-only
            )
            if force_ascii
            else Console()
        )
        _default_output = RichOutputHandler(console, use_unicode=not force_ascii)
    return _default_output


def set_output(handler: OutputHandler) -> None:
    """Set the output handler (for testing)."""
    global _default_output
    _default_output = handler
