"""Internal version helper for PanoSpace.

This file follows the *single‑source* version pattern: the version string is
kept in **one** place and imported elsewhere (e.g. in ``panospace.__init__`` or
setup metadata) to avoid mismatches.

During *installed* runs (pip/conda), the version is obtained from
``importlib.metadata``.  When running from a *git checkout* without
installation, we fall back to a placeholder ``0+unknown``.
"""
from __future__ import annotations

from importlib import metadata as _metadata


def _get_version() -> str:
    """Return the current version of *panospace*.

    If the package is not installed (e.g. editable mode without a
    build backend that writes the version), a default string is
    returned so that importing *panospace* never fails.
    """
    try:
        return _metadata.version("panospace")
    except _metadata.PackageNotFoundError:  # pragma: no cover – dev tree only
        return "0+unknown"


__version__: str = _get_version()

__all__: list[str] = ["__version__", "_get_version"]
