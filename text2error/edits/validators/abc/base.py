from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

from abc import ABC, abstractmethod


class TextEditsValidator(ABC):
    # pylint: disable=too-few-public-methods
    @abstractmethod
    def validate(self, source_text: str, modified_text: str) -> bool:
        ...  # pragma: no cover
