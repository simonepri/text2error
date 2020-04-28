from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import
from abc import ABC, abstractmethod

from ....utils.misc import resolve_optional_value_or_callable


class TextEditsGenerator(ABC):
    # pylint: disable=too-few-public-methods
    def __init__(
        self,
        edits_num: Optional[Union[int, Callable[[int, Optional[int]], int]]] = None,
    ):
        self.edits_num = edits_num

    def _get_edits_num(self, seq_len: int, max_edits_num: Optional[int]) -> int:
        edits_num = resolve_optional_value_or_callable(
            self.edits_num,
            1 if max_edits_num is None else min(1, max_edits_num),
            seq_len,
            max_edits_num,
        )
        return edits_num

    @abstractmethod
    def generate(self, text: str) -> str:
        ...  # pragma: no cover
