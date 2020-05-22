from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import
from abc import abstractmethod

import random

from ...abc.base import RandomTextEditsGenerator
from .....edit import TextEdit


class MaskedLMRandomTextEditsGenerator(RandomTextEditsGenerator):
    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        model_name: str,
        rng: Optional[random.Random] = None,
        edits_num: Optional[Union[int, Callable[[int, Optional[int]], int]]] = None,
    ) -> None:
        super().__init__(rng, edits_num)

        self.model_name = model_name

    def __del__(self) -> None:
        ...

    @abstractmethod
    def generate(self, text: str) -> List[TextEdit]:
        ...  # pragma: no cover
