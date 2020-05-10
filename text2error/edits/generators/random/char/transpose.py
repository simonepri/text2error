from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

from ..abc.base import RandomTextEditsGenerator
from ....edit import TextEdit
from .....utils.random import non_adjacent_sample


class TransposeRandomChar(RandomTextEditsGenerator):
    # pylint: disable=too-few-public-methods

    def generate(self, text: str) -> List[TextEdit]:
        chars_num = len(text)

        transpositions = self._get_edits_num(chars_num, chars_num // 2)
        if transpositions == 0:
            return []
        if transpositions > chars_num // 2:
            raise ValueError("Too many transpositions")

        indexes = non_adjacent_sample(chars_num - 1, transpositions, rng=self.rng)
        # The indexes from non_adjacent_sample are already sorted.

        start_gen = indexes
        text_gen = (text[i + 1 : None if i == 0 else i - 1 : -1] for i in indexes)
        edits = [TextEdit(t, start=s, end=s + 2) for t, s in zip(text_gen, start_gen)]

        return edits
