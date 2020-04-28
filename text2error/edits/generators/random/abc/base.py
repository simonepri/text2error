from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import
from abc import abstractmethod

import numpy as np

from ...abc.base import TextEditsGenerator
from .....utils.misc import resolve_optional


class RandomTextEditsGenerator(TextEditsGenerator):
    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        rng: Optional[np.random.Generator] = None,
        edits_num: Optional[Union[int, Callable[[int, Optional[int]], int]]] = None,
    ) -> None:
        super().__init__(edits_num)

        self.rng: np.random.Generator = resolve_optional(rng, np.random.default_rng())

    @abstractmethod
    def generate(self, text: str) -> str:
        ...  # pragma: no cover
