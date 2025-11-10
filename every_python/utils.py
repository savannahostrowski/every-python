import re
import subprocess
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

    # Try Homebrew installation (checks both llvm@{version} and llvm)
    brew_tool = _get_homebrew_llvm_tool(tool, version)
    if brew_tool and _check_tool_version(brew_tool, version):
        return True

    return False


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
