"""Output handler abstraction for testability and flexibility."""

from abc import ABC, abstractmethod
from rich.console import Console


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

    def __init__(self, console: Console | None = None):
        self.console = console or Console()

    def info(self, message: str) -> None:
        self.console.print(message)

    def success(self, message: str) -> None:
        self.console.print(f"[green]{message}[/green]")

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


def get_output() -> OutputHandler:
    """Get the default output handler."""
    global _default_output
    if _default_output is None:
        _default_output = RichOutputHandler()
    return _default_output


def set_output(handler: OutputHandler) -> None:
    """Set the output handler (for testing)."""
    global _default_output
    _default_output = handler
