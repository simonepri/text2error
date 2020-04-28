from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

import numpy as np

from ..abc.base import RandomTextEditsGenerator


class RemoveRandomChar(RandomTextEditsGenerator):
    # pylint: disable=too-few-public-methods

    def generate(self, text: str) -> str:
        chars = np.array(list(text))

        remotions = self._get_edits_num(len(chars), len(chars))
        if remotions == 0:
            return text
        if remotions > len(chars):
            raise ValueError("Too many remotions")

        return self._remove(chars, remotions)

    def _remove(self, chars: np.array, remotions: int) -> str:
        indexes_to_remove = self.rng.choice(len(chars), remotions, replace=False)
        return self._remove_at_indexes(chars, indexes_to_remove)

    def _remove_at_indexes(self, chars: np.array, indexes: np.array) -> str:
        # pylint: disable=no-self-use
        remaining_indexes = np.setdiff1d(
            np.arange(len(chars)), indexes, assume_unique=True
        )
        new_chars = chars[remaining_indexes]
        return "".join(new_chars)
