from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

from .edits.validators.abc.base import TextEditsValidator

from .utils.misc import resolve_optional, resolve_value_or_callable


class Text2ErrorValidator:
    # pylint: disable=too-few-public-methods
    def __init__(
        self,
        validators: List[TextEditsValidator],
        validation_order: Optional[
            Union[List[int], Callable[[int, str, str], List[int]]]
        ] = None,
    ) -> None:
        self.validators = list(validators)
        self.validation_order: Union[
            List[int], Callable[[int, str, str], List[int]]
        ] = resolve_optional(validation_order, list(range(len(self))))

    def __len__(self) -> int:
        return len(self.validators)

    def __call__(self, source_text: str, modified_text: str) -> bool:
        validation_order = resolve_value_or_callable(
            self.validation_order, len(self), source_text, modified_text
        )

        valid = True
        for idx in validation_order:
            valid = valid and self.validators[idx].validate(source_text, modified_text)
            if not valid:
                break

        return valid
