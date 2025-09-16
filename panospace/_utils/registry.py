"""
PanoSpace Plugin Registry
=========================
Registry utility to manage and dynamically discover plugins via entry points.
"""

from importlib import metadata
from typing import Any, Dict


class Registry:
    """A simple registry to store and retrieve plugin implementations."""

    def __init__(self, group: str):
        """Initialize the registry with a specific entry-point group.

        Parameters
        ----------
        group : str
            The name of the entry-point group to load plugins from.
        """
        self.group = group
        self.plugins: Dict[str, Any] = {}
        self.load_plugins()

    def load_plugins(self):
        """Load all plugins registered under the entry-point group."""
        for entry_point in metadata.entry_points(group=self.group):
            self.plugins[entry_point.name] = entry_point.load()

    def get(self, name: str) -> Any:
        """Retrieve a plugin by name.

        Parameters
        ----------
        name : str
            Name of the plugin to retrieve.

        Returns
        -------
        Any
            The loaded plugin implementation.

        Raises
        ------
        KeyError
            If the plugin name is not found in the registry.
        """
        try:
            return self.plugins[name]
        except KeyError:
            raise KeyError(f"Plugin '{name}' not found in registry '{self.group}'")


# Default global registry for PanoSpace plugins
registry = Registry(group="panospace.plugins")
