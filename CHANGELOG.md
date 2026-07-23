# Changelog

All notable changes to every-python are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

* Bump astral-sh/setup-uv from 8.3.2 to 9.0.0. PR [#62](https://github.com/savannahostrowski/every-python/pull/62) by [@dependabot[bot]](https://github.com/apps/dependabot).
* Bump ruff from 0.15.21 to 0.15.22. PR [#64](https://github.com/savannahostrowski/every-python/pull/64) by [@dependabot[bot]](https://github.com/apps/dependabot).
* 🔖 Bump for 0.7.0 release. PR [#59](https://github.com/savannahostrowski/every-python/pull/59) by [@savannahostrowski](https://github.com/savannahostrowski).

## [0.7.0] - 2026-07-21

* ⚡️ Add reference cloning support. PR [#58](https://github.com/savannahostrowski/every-python/pull/58) by [@savannahostrowski](https://github.com/savannahostrowski).
* ✨ Add support for custom CPython repositories. PR [#57](https://github.com/savannahostrowski/every-python/pull/57) by [@savannahostrowski](https://github.com/savannahostrowski).
* 📝 Add automated changelog updates. PR [#56](https://github.com/savannahostrowski/every-python/pull/56) by [@savannahostrowski](https://github.com/savannahostrowski).

### Added

- Add `--reference-repo` and `EVERY_PYTHON_REFERENCE_REPO` to accelerate
  first-time managed clones by reusing Git objects from a local CPython checkout.
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

[Unreleased]: https://github.com/savannahostrowski/every-python/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/savannahostrowski/every-python/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/savannahostrowski/every-python/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/savannahostrowski/every-python/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/savannahostrowski/every-python/compare/v0.4.1...v0.5.0
