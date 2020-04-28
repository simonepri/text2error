from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

T = TypeVar("T")  # pylint: disable=invalid-name


class KeyedSingletonLoader:
    def __init__(self):
        self.cache: Dict[T, Any] = {}

    def loaded(self, key: T) -> bool:
        return key in self.cache

    def __getitem__(self, key: T):
        if not self.loaded(key):
            raise ValueError("The provided key is not loaded: " + str(key))
        return self.cache[key]

    def load(self, key: T, singleton_provider: Callable[[T], Any]) -> Any:
        if not self.loaded(key):
            self.cache[key] = 0, singleton_provider(key)
        key_cache = self.cache[key]
        self.cache[key] = key_cache[0] + 1, key_cache[1]
        return key_cache[1]

    def unload(self, key: T) -> None:
        if not self.loaded(key):
            return
        key_cache = self.cache[key]
        if key_cache[0] == 1:
            del self.cache[key]
        else:
            self.cache[key] = key_cache[0] - 1, key_cache[1]
