import weakref
from collections.abc import Callable, Hashable
from typing import Any


class StateRangeCache:
    """Color ranges computed per state, keyed by the state object."""

    def __init__(self):
        self._by_state: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def get(self, state: object, key: Hashable, compute: Callable[[], Any]) -> Any:
        """The cached value for (state, key), computing it on first request."""
        entries = self._by_state.setdefault(state, {})
        if key not in entries:
            entries[key] = compute()
        return entries[key]
