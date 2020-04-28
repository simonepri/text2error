from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

import numpy as np
import torch as pt

from .abc.base import MaskedLMRandomTextEditsGenerator


class InsertRandomMLMToken(MaskedLMRandomTextEditsGenerator):
    # pylint: disable=too-few-public-methods

    def generate(self, text: str) -> str:
        non_special_ids = self._tokenize(text)

        insertions = self._get_edits_num(len(non_special_ids), None)
        if insertions == 0:
            return text

        return self._insert(non_special_ids, insertions)

    def _insert(self, non_special_ids: pt.Tensor, insertions: int) -> str:
        indexes_to_insert = self.rng.choice(
            len(non_special_ids) or 1, insertions, replace=True
        )
        return self._insert_at_indexes(non_special_ids, indexes_to_insert)

    def _insert_at_indexes(self, non_special_ids: pt.Tensor, indexes: np.array) -> str:
        indexes.sort()
        # masked_ids.shape = [len(non_special_ids) + len(indexes)]
        non_special_ids = np.insert(
            non_special_ids, indexes, self.tokenizer.mask_token_id
        )
        indexes += np.arange(indexes.size)

        # TODO: Avoid to decode and encode.
        text = self.tokenizer.decode(non_special_ids.tolist())
        masked_ids, non_special_mask, _ = self._tokenize_for_model(text)

        indexes = pt.from_numpy(indexes)
        _, new_non_special_ids = self._predict_masks_at_indexes(
            indexes, masked_ids, non_special_mask, non_special_ids[indexes]
        )

        return self.tokenizer.decode(new_non_special_ids.tolist())
