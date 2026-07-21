# every-python

Build and run any commit of CPython, inspired by [every-ts](https://github.com/jakebailey/every-ts).

See the [changelog](CHANGELOG.md) for release notes and version comparisons.

## Why does this exist?

Building CPython from source is time-consuming. `every-python` makes it easy to:
- Test your code against different Python versions
- Reproduce bugs in specific commits
- Test experimental features like the JIT compiler
- Bisect to find which commit introduced a regression

## Features

- **Build any CPython commit** - main, release tags, or specific commits
- **Build with experimental JIT support** - Build with `--enable-experimental-jit` (includes LLVM version detection)
- **Build with PGO + LTO** - Optimized release builds via `--pgo`
- **Build free-threaded (no-GIL)** - Build with `--disable-gil` via `--nogil` (3.13+)
- **Smart caching** - Builds cached in `~/.every-python/builds/` for instant reuse
- **Git bisect integration** - Automatically find which commit introduced a bug

## How it works

`every-python` makes a [blobless clone](https://github.blog/open-source/git/get-up-to-speed-with-partial-clone-and-shallow-clone/) of the CPython repository (~200MB), checks out the version you want, and builds it locally with `--with-pydebug`. Built versions are cached in `~/.every-python/builds/` for reuse.

## Installation

```bash
uv tool install every-python
# or
pipx install every-python
# or
pip install every-python

# Development install from source
git clone https://github.com/yourusername/every-python.git
cd every-python
uv sync
```

## Requirements

- Git
- [CPython build dependencies](https://devguide.python.org/getting-started/setup-building/)
- [LLVM](https://github.com/python/cpython/blob/main/Tools/jit/README.md), for JIT builds

### Platform support

- **macOS / Linux**: builds via `./configure --with-pydebug` + `make`.
- **Windows**: builds via `PCbuild\build.bat -c Debug -p <arch>` (requires Visual Studio with C++ workload, as documented in [PCbuild/readme.txt](https://github.com/python/cpython/blob/main/PCbuild/readme.txt)). The platform is auto-detected from the host (`x64`, `ARM64`, or `Win32`) and artifacts are copied from the matching `PCbuild\<arch>` directory into the cache. The resulting binary is `python_d.exe`.

## Usage

### Build and install a Python version

```bash
# Build from main branch
every-python install main

# Build from a release tag
every-python install v3.13.0

# Build from a specific commit
every-python install abc123d

# Show build output (useful for debugging build failures)
every-python install main --verbose

# Limit parallel compilation to 8 jobs
every-python install main --jobs 8
```

### Reuse compilation results with ccache

On macOS and Linux, use [`ccache`](https://ccache.dev/) to make builds of nearby
commits substantially faster:

```bash
# macOS
brew install ccache

# ccache is used automatically when available
every-python install main

# Require ccache, or explicitly disable it
every-python install main --ccache
every-python install main --no-ccache
```

The first build populates the cache. Later builds reuse unchanged compilation
results even though every-python cleans its CPython checkout between builds.
Automatic detection also applies to builds from `run` and `bisect`. Pass
`--ccache` to require it (and fail if it is unavailable), or `--no-ccache` to
disable it. Every-python respects an existing `CC` setting and otherwise uses
Clang on macOS or the default C compiler on Linux. Windows builds continue
normally unless `--ccache` is explicitly requested, which is unsupported.

### Run Python with a specific version

```bash
# Run Python REPL
every-python run main -- python

# Run a script
every-python run v3.13.0 -- python your_script.py

# Run with arguments
every-python run main -- python -c "print('Hello!')"
```

If the version isn't built yet, it will build it automatically.

### Build with JIT compiler (experimental)

Build Python with the experimental JIT compiler:

```bash
# Build with JIT enabled
every-python install main --jit

# Run with JIT-enabled build
every-python run main --jit -- python -c "print('Hello from JIT!')"

# Bisect with JIT to find JIT-specific bugs
every-python bisect --good v3.13.0 --bad main --jit --run "python test.py"
```

**JIT Requirements:**
`every-python` will attempt to detect the correct version of LLVM for the commit being built from `LLVM_VERSION` specified in `Tools/jit/_llvm.py` at the time of the commit. If you are missing the required LLVM version, you will see an error during the build. For more information on installing LLVM, see [CPython JIT documentation](https://github.com/python/cpython/blob/main/Tools/jit/README.md).

Note: LLVM is only needed at build time, not at runtime. JIT and non-JIT builds are stored separately.

### Build with PGO + LTO

Build Python with profile-guided optimization and link-time optimization:

```bash
# Build with PGO + LTO
every-python install main --pgo

# Combine with JIT
every-python install main --jit --pgo
```

On Unix this maps to `--enable-optimizations`; on Windows it triggers the `PGInstrument`/`PGUpdate` configuration cycle. Both produce PGO + LTO builds. Expect the build to take significantly longer than a regular debug build.

### Build free-threaded (no-GIL)

Build Python with the GIL disabled (PEP 703, available in 3.13+):

```bash
# Build a free-threaded Python
every-python install main --nogil

# Combine flags
every-python install main --pgo --nogil
```

`every-python` checks whether the commit supports `--disable-gil` before building. Pre-3.13 commits silently accept the flag but produce a regular GIL-enabled build, so the tool will warn and ask before continuing.

**Note:** `--jit` and `--nogil` together is a degenerate configuration upstream. The build succeeds and free-threading is enabled, but the JIT code is compiled in and never actually used at runtime ([configure.ac](https://github.com/python/cpython/blob/main/configure.ac) emits a warning, see [GH-133171](https://github.com/python/cpython/issues/133171)).

### List built versions

```bash
every-python list-builds
```

### Clean up builds

```bash
# Remove a specific build
every-python clean v3.13.0

# Remove all builds
every-python clean --all
```

### Bisect to find bugs

Use git bisect to automatically find which commit introduced a bug:

```bash
# Find when a test started failing
# Exit with code 0 = good commit, 1 = bad commit
every-python bisect \
  --good v3.13.0 \
  --bad main \
  --run "python test_my_feature.py"

# Bisect with JIT-enabled builds
every-python bisect \
  --good v3.12.0 \
  --bad main \
  --jit \
  --run "python test_jit_api.py"
```

This will:
1. Resolve the good and bad commits
2. Start a git bisect
3. Build each commit that git bisect tests
4. Run your test command
5. Automatically mark commits as good/bad based on exit code:
   - Exit 0 = good commit
   - Exit 1-127 (except 125) = bad commit
   - Exit 125 = skip this commit (can't test it)
6. Find the exact commit that introduced the change

**Note on bisecting across branches:** Bisecting between release tags and main can be tricky due to backporting. For best results, bisect within a single branch (e.g., use commit hashes on main instead of crossing from v3.12.0 to main).

**Writing test scripts for bisect:** Your test script should exit with code 0 for "good" (old/expected behavior) and 1 for "bad" (new/broken behavior). For example, if I wanted to find when `_jit` was added to the `sys` module, I could use this script:

```python
import sys
# Exit 0 (good) = feature doesn't exist yet
# Exit 1 (bad) = feature exists
if hasattr(sys, "_jit"):
    sys.exit(1)  # Feature exists - mark as "bad"
sys.exit(0)  # Feature doesn't exist - mark as "good"
```

## Project Structure

```
~/.every-python/
├── cpython/          # Blobless clone of CPython repository
└── builds/           # Cached builds
    ├── abc123d/              # Build for commit abc123d
    ├── abc123d-jit/          # JIT build
    ├── abc123d-pgo/          # PGO + LTO build
    ├── abc123d-nogil/        # Free-threaded build
    ├── abc123d-jit-pgo-nogil/  # All flags combined
    └── def456e/              # Build for commit def456e
```

## License

MIT
