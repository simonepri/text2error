from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

from ..abc.base import RandomTextEditsGenerator
from ....edit import TextEdit


class DuplicateRandomChar(RandomTextEditsGenerator):
    # pylint: disable=too-few-public-methods

    def generate(self, text: str) -> List[TextEdit]:
        chars_num = len(text)

        duplications = self._get_edits_num(chars_num, chars_num)
        if duplications == 0:
            return []
        if duplications > chars_num:
            raise ValueError("Too many duplications")

        indexes = self.rng.sample(range(chars_num), k=duplications)
        indexes.sort()

        start_gen = (s + i for i, s in enumerate(indexes))
        text_gen = map(text.__getitem__, indexes)
        edits = [TextEdit(t, start=s) for t, s in zip(text_gen, start_gen)]

        return edits
