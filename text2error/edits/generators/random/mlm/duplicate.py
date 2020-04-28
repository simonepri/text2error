from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

import numpy as np
import torch as pt

from .abc.base import MaskedLMRandomTextEditsGenerator


class DuplicateRandomMLMToken(MaskedLMRandomTextEditsGenerator):
    # pylint: disable=too-few-public-methods

    def generate(self, text: str) -> str:
        non_special_ids = self._tokenize(text)

        duplications = self._get_edits_num(len(non_special_ids), len(non_special_ids))
        if duplications == 0:
            return text
        if duplications > len(non_special_ids):
            raise ValueError("Too many duplications")

        return self._duplicates(non_special_ids, duplications)

    def _duplicates(self, non_special_ids: pt.Tensor, duplications: int) -> str:
        indexes_to_duplicates = self.rng.choice(
            len(non_special_ids), duplications, replace=False
        )
        return self._duplicates_at_indexes(non_special_ids, indexes_to_duplicates)

    def _duplicates_at_indexes(
        self, non_special_ids: pt.Tensor, indexes: np.array,
    ) -> str:
        # output_ids.shape = [len(non_special_ids) + len(indexes)]
        output_ids = np.insert(non_special_ids, indexes, non_special_ids[indexes])

        return self.tokenizer.decode(output_ids.tolist())
