from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

import numpy as np
import torch as pt

from .abc.base import MaskedLMRandomTextEditsGenerator


class RemoveRandomMLMToken(MaskedLMRandomTextEditsGenerator):
    # pylint: disable=too-few-public-methods

    def generate(self, text: str) -> str:
        non_special_ids = self._tokenize(text)

        remotions = self._get_edits_num(len(non_special_ids), len(non_special_ids))
        if remotions == 0:
            return text
        if remotions > len(non_special_ids):
            raise ValueError("Too many remotions")

        return self._remove(non_special_ids, remotions)

    def _remove(self, non_special_ids: pt.Tensor, remotions: int) -> str:
        indexes_to_remove = self.rng.choice(
            len(non_special_ids), remotions, replace=False
        )
        return self._remove_at_indexes(non_special_ids, indexes_to_remove)

    def _remove_at_indexes(self, non_special_ids: pt.Tensor, indexes: np.array,) -> str:
        # remaining_indexes.shape = [len(non_special_ids) - len(indexes)]
        remaining_indexes = pt.from_numpy(
            np.setdiff1d(np.arange(len(non_special_ids)), indexes, assume_unique=True)
        )

        # output_ids.shape = [len(non_special_ids) - len(indexes)]
        output_ids = non_special_ids[remaining_indexes]

        return self.tokenizer.decode(output_ids.tolist())
