from collections import OrderedDict

class DisabledStateCacheManager:
    def enabled(self):
        return False

    def get(self, key: tuple):
        return None

    def add(self, key: tuple, state) -> None:
        pass

class StateCacheManager:
    def __init__(self, max_size=4):
        self._cache = OrderedDict()
        self.max_size = max_size

    def enabled(self):
        return True

    def get(self, key: tuple):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def add(self, key: tuple, state) -> None:
        if key not in self._cache and len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
        self._cache[key] = state
        self._cache.move_to_end(key)