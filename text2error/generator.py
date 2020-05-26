from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

from .edits.edit import TextEdit
from .edits.generators.abc.base import TextEditsGenerator
from .utils.misc import resolve_optional, resolve_value_or_callable


class Text2ErrorGenerator:
    # pylint: disable=too-few-public-methods
    def __init__(
        self,
        generators: List[TextEditsGenerator],
        generation_order: Optional[Union[List[int], Callable[[int], List[int]]]] = None,
    ) -> None:
        self.generators = list(generators)
        self.generation_order: Union[
            List[int], Callable[[int], List[int]]
        ] = resolve_optional(generation_order, list(range(len(self))))

    def __len__(self) -> int:
        return len(self.generators)

    def __call__(self, text: str) -> Tuple[str, List[TextEdit]]:
        generation_order = resolve_value_or_callable(self.generation_order, len(self))

        all_edits = []
        for idx in generation_order:
            edits = self.generators[idx].generate(text)
            all_edits.extend(edits)
            text = TextEdit.apply(text, edits)
        return text, all_edits
