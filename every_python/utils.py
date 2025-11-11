import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


def get_llvm_version_for_commit(commit: str, repo_dir: Path) -> str | None:
    """Return the LLVM version required for the given CPython commit."""
    try:
        result = subprocess.run(
            ["git", "show", f"{commit}:Tools/jit/_llvm.py"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )

        match = re.search(r'_LLVM_VERSION\s*=\s*["\']?(\d+)["\']?', result.stdout)

        if match:
            return match.group(1)
    except subprocess.CalledProcessError:
        # File doesn't exist - JIT not available in this commit
        pass

    return None


def check_llvm_available(version: str) -> bool:
    """
    Check if a specific LLVM version is available on the system.

    Searches for clang and llvm-readobj with the correct version.
    """
    required_tools = ["clang", "llvm-readobj"]

    for tool in required_tools:
        if not _check_tool_available(tool, version):
            return False

    return True


def _check_tool_available(tool: str, version: str) -> bool:
    """Check if a specific tool with the given version is available."""
    # Try versioned tool first (e.g., clang-20)
    versioned_tool = f"{tool}-{version}"
    if _check_tool_version(versioned_tool, version):
        return True

    # Try unversioned tool in PATH
    if _check_tool_version(tool, version):
        return True

    # Platform-specific checks
    if platform.system() == "Darwin":
        # macOS: Try Homebrew installation
        brew_tool = _get_homebrew_llvm_tool(tool, version)
        if brew_tool and _check_tool_version(brew_tool, version):
            return True
    elif platform.system() == "Windows":
        # Windows: Check Program Files
        windows_tool = _get_windows_llvm_tool(tool, version)
        if windows_tool and _check_tool_version(windows_tool, version):
            return True

    return False


def _get_windows_llvm_tool(tool: str, version: str) -> str | None:
    """Get the path to an LLVM tool from Windows installation."""
    # Common Windows LLVM installation paths
    program_files = Path("C:/Program Files")
    program_files_x86 = Path("C:/Program Files (x86)")

    possible_paths = [
        program_files / f"LLVM-{version}" / "bin" / f"{tool}.exe",
        program_files / "LLVM" / "bin" / f"{tool}.exe",
        program_files_x86 / f"LLVM-{version}" / "bin" / f"{tool}.exe",
        program_files_x86 / "LLVM" / "bin" / f"{tool}.exe",
    ]

    for tool_path in possible_paths:
        if tool_path.exists():
            try:
                result = subprocess.run(
                    [str(tool_path), "--version"],
                    capture_output=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    return str(tool_path)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

    return None


def _get_homebrew_llvm_tool(tool: str, version: str | None = None) -> str | None:
    """Get the path to an LLVM tool from Homebrew installation."""
    # Try version-specific formula first (e.g., llvm@20)
    if version:
        try:
            result = subprocess.run(
                ["brew", "--prefix", f"llvm@{version}"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            llvm_prefix = result.stdout.strip()
            tool_path = f"{llvm_prefix}/bin/{tool}"

            # Check if the tool exists
            if (
                subprocess.run(
                    [tool_path, "--version"],
                    capture_output=True,
                    timeout=5,
                ).returncode
                == 0
            ):
                return tool_path
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            pass

    # Try default llvm formula
    try:
        result = subprocess.run(
            ["brew", "--prefix", "llvm"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        llvm_prefix = result.stdout.strip()
        tool_path = f"{llvm_prefix}/bin/{tool}"

        # Check if the tool exists
        if (
            subprocess.run(
                [tool_path, "--version"],
                capture_output=True,
                timeout=5,
            ).returncode
            == 0
        ):
            return tool_path
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        pass

    return None


def _check_tool_version(tool_name: str, expected_version: str) -> bool:
    """Check if the given tool matches the required version."""
    try:
        result = subprocess.run(
            [tool_name, "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )

        # Look for version pattern like "version 20.1.8"
        pattern = rf"version\s+{expected_version}\.\d+\.\d+"
        return bool(re.search(pattern, result.stdout))
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return False


def python_binary_location(builds_dir: Path, build_info: "BuildInfo") -> Path:
    """Get the path to the Python binary for a given build."""
    if platform.system() == "Windows":
        return builds_dir / build_info.directory_name / "python.exe"
    else:
        return builds_dir / build_info.directory_name / "bin" / "python3"


@dataclass
class BuildInfo:
    """Information about a Python build."""

    commit: str
    jit_enabled: bool

    @property
    def suffix(self) -> str:
        """Get the build suffix."""
        return "-jit" if self.jit_enabled else ""

    @property
    def directory_name(self) -> str:
        """Get the build directory name."""
        return f"{self.commit}{self.suffix}"

    def get_path(self, builds_dir: Path) -> Path:
        """Get the full build directory path."""
        return builds_dir / self.directory_name

    @classmethod
    def from_directory_name(cls, name: str) -> "BuildInfo":
        """Parse build info from directory name."""
        if name.endswith("-jit"):
            return cls(commit=name[:-4], jit_enabled=True)
        return cls(commit=name, jit_enabled=False)

    @classmethod
    def from_directory(cls, path: Path) -> "BuildInfo":
        """Parse build info from directory path."""
        return cls.from_directory_name(path.name)
