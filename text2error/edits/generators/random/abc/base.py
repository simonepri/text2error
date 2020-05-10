from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import
from abc import abstractmethod

import random

from ...abc.base import TextEditsGenerator
from ....edit import TextEdit
from .....utils.misc import resolve_optional


class RandomTextEditsGenerator(TextEditsGenerator):
    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        rng: Optional[random.Random] = None,
        edits_num: Optional[Union[int, Callable[[int, Optional[int]], int]]] = None,
    ) -> None:
        super().__init__(edits_num)

        self.rng = resolve_optional(rng, random._inst)  # type: ignore

    @abstractmethod
    def generate(self, text: str) -> List[TextEdit]:
        ...  # pragma: no cover
