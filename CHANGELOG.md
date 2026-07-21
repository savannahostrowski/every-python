# Changelog

All notable changes to every-python are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

* ✨ Add support for custom CPython repositories. PR [#57](https://github.com/savannahostrowski/every-python/pull/57) by [@savannahostrowski](https://github.com/savannahostrowski).
* 📝 Add automated changelog updates. PR [#56](https://github.com/savannahostrowski/every-python/pull/56) by [@savannahostrowski](https://github.com/savannahostrowski).
### Added

- Add `--repo` support for building and bisecting CPython forks.
- Add optional compiler caching with `--ccache` on macOS and Linux.
- Add `--jobs` to control parallel compilation for Unix builds.
- Add `--version` to display the installed every-python version.

### Changed

- Probe cached Python versions concurrently to make `list-builds` faster.

### Fixed

- Handle Typer releases that vendor Click without raising an exception.

## [0.6.0] - 2026-05-06

### Added

- Add free-threaded CPython builds with `--nogil`.
- Add optimized PGO and LTO builds with `--pgo`.
- Add Windows ARM build support.

### Changed

- Migrate static type checking from mypy to ty.
- Pin GitHub Actions to commit SHAs and configure Dependabot updates.

## [0.5.1] - 2026-04-08

### Fixed

- Include `typing-extensions` as a runtime dependency.
- Correct project URLs in package metadata.

## [0.5.0] - 2026-01-07

### Changed

- Clean the CPython checkout before every build.
- Improve build failure diagnostics.

### Fixed

- Locate older `python3.0` binaries correctly.

[Unreleased]: https://github.com/savannahostrowski/every-python/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/savannahostrowski/every-python/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/savannahostrowski/every-python/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/savannahostrowski/every-python/compare/v0.4.1...v0.5.0
