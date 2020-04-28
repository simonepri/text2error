from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

import numpy as np
import torch as pt

from .abc.base import MaskedLMRandomTextEditsGenerator
from .....utils.random import non_adjacent_choice


class TransposeRandomMLMToken(MaskedLMRandomTextEditsGenerator):
    # pylint: disable=too-few-public-methods

    def generate(self, text: str) -> str:
        non_special_ids = self._tokenize(text)

        transpositions = self._get_edits_num(
            len(non_special_ids), len(non_special_ids) // 2
        )
        if transpositions == 0:
            return text
        if transpositions > len(non_special_ids) // 2:
            raise ValueError("Too many transpositions")

        return self._transpose(non_special_ids, transpositions)

    def _transpose(self, non_special_ids: pt.Tensor, transpositions: int) -> str:
        indexes_to_transpose = non_adjacent_choice(
            len(non_special_ids) - 1, transpositions, rng=self.rng
        )
        return self._transpose_at_indexes(non_special_ids, indexes_to_transpose)

    def _transpose_at_indexes(
        self, non_special_ids: pt.Tensor, indexes: np.array,
    ) -> str:
        # output_ids.shape = [len(non_special_ids)]
        output_ids = non_special_ids.clone()
        output_ids[indexes] = non_special_ids[indexes + 1]
        output_ids[indexes + 1] = non_special_ids[indexes]

        return self.tokenizer.decode(output_ids.tolist())
