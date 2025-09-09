"""panospace._core
=================
Low-level implementation layer: *all* compute-heavy or backend-specific
algorithms live here.  By funnelling public calls through lightweight wrappers
in :pymod:`panospace.tl` / :pymod:`panospace.pl`, we can swap
implementations, add GPU kernels, or expose new methods without breaking the
user-facing API.

Sub-packages
------------
* :pymod:`panospace._core.detection`   - nuclei / cell detection back-ends
* :pymod:`panospace._core.annotation`  - cell-type annotation algorithms
* :pymod:`panospace._core.prediction`  - gene-expression prediction & super-resolution
* :pymod:`panospace._core.microenv`    - micro-environment statistics & L-R inference
* :pymod:`panospace._core.superres`    - experimental whole-slide super-resolution utilities

A *lightweight plugin registry* lives in this ``__init__`` so that external
packages can register new back-ends via Python *entry points* or at import
time without editing the PanoSpace source tree.

Example
~~~~~~~
>>> from panospace._core import register, get
>>> def my_detector(sdata, **kw):
...     ...
>>> register("detection", "my_net", my_detector)
>>> fn = get("detection", "my_net")
>>> fn(sdata)  # doctest: +SKIP
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Callable, Dict
import logging

logger = logging.getLogger("panospace._core")

# -----------------------------------------------------------------------------
# Internal registry ----------------------------------------------------------------
# category -> name -> callable
# e.g. "detection" -> {"cellvit": <func>, "stardist": <func>}
_REGISTRY: Dict[str, Dict[str, Callable]] = {}


# Public helpers --------------------------------------------------------------------

def register(category: str, name: str, func: Callable, *, overwrite: bool = False) -> None:
    """Register *func* under ``category/name``.

    Parameters
    ----------
    category
        Logical group, e.g. ``"detection"``.
    name
        Unique backend identifier inside *category*.
    func
        Callable implementation.
    overwrite
        If *False* (default), trying to register an existing ``category/name``
        raises ``ValueError``; set *True* to replace.
    """
    sub = _REGISTRY.setdefault(category, {})
    if name in sub and not overwrite:
        raise ValueError(f"Backend '{category}/{name}' already registered; use overwrite=True to replace.")
    sub[name] = func
    logger.debug("Registered backend %s/%s -> %s", category, name, func)


def get(category: str, name: str) -> Callable:
    """Retrieve the callable previously registered under *category* / *name*.

    Raises
    ------
    KeyError
        If the pair has not been registered.
    """
    try:
        return _REGISTRY[category][name]
    except KeyError as e:
        raise KeyError(f"Backend '{category}/{name}' is not registered.") from e


def available(category: str) -> MappingProxyType:
    """Return a *read-only* mapping of all back-ends within *category*."""
    return MappingProxyType(_REGISTRY.get(category, {}))


# -----------------------------------------------------------------------------
# Dunder exports --------------------------------------------------------------------
__all__ = [
    "register",
    "get",
    "available",
]
