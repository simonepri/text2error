from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

import numpy as np

from ..abc.base import RandomTextEditsGenerator


class DuplicateRandomChar(RandomTextEditsGenerator):
    # pylint: disable=too-few-public-methods

    def generate(self, text: str) -> str:
        chars = np.array(list(text))

        duplications = self._get_edits_num(len(chars), len(chars))
        if duplications == 0:
            return text
        if duplications > len(chars):
            raise ValueError("Too many duplications")

        return self._duplicate(chars, duplications)

    def _duplicate(self, chars: np.array, duplications: int) -> str:
        indexes_to_duplicate = self.rng.choice(len(chars), duplications, replace=False)
        return self._duplicate_at_indexes(chars, indexes_to_duplicate)

    def _duplicate_at_indexes(self, chars: np.array, indexes: np.array) -> str:
        # pylint: disable=no-self-use
        new_chars = np.insert(chars, indexes, chars[indexes])
        return "".join(new_chars)
