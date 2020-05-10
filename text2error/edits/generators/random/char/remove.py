from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

from ..abc.base import RandomTextEditsGenerator
from ....edit import TextEdit


class RemoveRandomChar(RandomTextEditsGenerator):
    # pylint: disable=too-few-public-methods

    def generate(self, text: str) -> List[TextEdit]:
        chars_num = len(text)

        remotions = self._get_edits_num(chars_num, chars_num)
        if remotions == 0:
            return []
        if remotions > chars_num:
            raise ValueError("Too many remotions")

        indexes = self.rng.sample(range(chars_num), k=remotions)
        indexes.sort()

        start_gen = (s - i for i, s in enumerate(indexes))
        edits = [TextEdit("", start=s, end=s + 1) for s in start_gen]

        return edits
