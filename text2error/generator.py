from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

from .edits.generators.abc.base import TextEditsGenerator

from .utils.misc import resolve_optional, resolve_value_or_callable


class Text2ErrorGenerator:
    # pylint: disable=too-few-public-methods
    def __init__(
        self,
        generators: List[TextEditsGenerator],
        generation_order: Optional[Union[List[int], Callable[[int], List[int]]]] = None,
        generation_mask: Optional[
            Union[List[bool], Callable[[int], List[bool]]]
        ] = None,
    ) -> None:
        self.generators = list(generators)
        self.generation_order: Union[
            List[int], Callable[[int], List[int]]
        ] = resolve_optional(generation_order, list(range(len(self))))
        self.generation_mask: Union[
            List[bool], Callable[[int], List[bool]]
        ] = resolve_optional(generation_mask, [True] * len(self))

    def __len__(self) -> int:
        return len(self.generators)

    def __call__(self, text: str) -> str:
        generation_order = resolve_value_or_callable(self.generation_order, len(self))
        generation_mask = resolve_value_or_callable(self.generation_mask, len(self))

        for idx in generation_order:
            if not generation_mask[idx]:
                continue
            text = self.generators[idx].generate(text)

        return text