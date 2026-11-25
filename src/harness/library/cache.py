"""Loaded plugin versions kept in this process, by content hash.

A version never changes, so an entry never goes stale: the limit only bounds memory.
"""

from __future__ import annotations

from collections import OrderedDict

from .plugin import Plugin


class PluginCache:
    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._plugins: OrderedDict[str, Plugin] = OrderedDict()

    def get(self, content_hash: str) -> Plugin | None:
        plugin = self._plugins.get(content_hash)
        if plugin is not None:
            self._plugins.move_to_end(content_hash)
        return plugin

    def put(self, plugin: Plugin) -> None:
        self._plugins[plugin.content_hash] = plugin
        self._plugins.move_to_end(plugin.content_hash)
        while len(self._plugins) > self._limit:
            self._plugins.popitem(last=False)
