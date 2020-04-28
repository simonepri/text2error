from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

import numpy as np

from ..abc.base import RandomTextEditsGenerator
from .....utils.random import non_adjacent_choice


class TransposeRandomChar(RandomTextEditsGenerator):
    # pylint: disable=too-few-public-methods
    def generate(self, text: str) -> str:
        chars = np.array(list(text))

        transpositions = self._get_edits_num(len(chars), len(chars) // 2)
        if transpositions == 0:
            return text
        if transpositions > len(chars) // 2:
            raise ValueError("Too many transpositions")

        return self._transpose(chars, transpositions)

    def _transpose(self, chars: np.array, transpositions: int) -> str:
        indexes_to_transpose = non_adjacent_choice(
            len(chars) - 1, transpositions, rng=self.rng
        )
        return self._transpose_at_indexes(chars, indexes_to_transpose)

    def _transpose_at_indexes(self, chars: np.array, indexes: np.array) -> str:
        # pylint: disable=no-self-use
        new_chars = chars.copy()
        new_chars[indexes] = chars[indexes + 1]
        new_chars[indexes + 1] = chars[indexes]
        return "".join(new_chars)
