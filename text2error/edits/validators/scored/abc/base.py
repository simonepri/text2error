from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

from abc import abstractmethod

from ...abc.base import TextEditsValidator
from .....utils.misc import resolve_optional


class ScoredTextEditsValidator(TextEditsValidator):
    # pylint: disable=too-few-public-methods
    def __init__(
        self,
        scoring_comp: Callable[[float, float], bool],
        scoring_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.scoring_comp = scoring_comp
        self.scoring_options: Dict[str, Any] = resolve_optional(scoring_options, {})

    @abstractmethod
    def validate(self, source_text: str, modified_text: str) -> bool:
        ...  # pragma: no cover
