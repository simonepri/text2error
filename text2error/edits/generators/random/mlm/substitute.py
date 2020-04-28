from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

import numpy as np
import torch as pt

from .abc.base import MaskedLMRandomTextEditsGenerator


class SubstituteRandomMLMToken(MaskedLMRandomTextEditsGenerator):
    # pylint: disable=too-few-public-methods

    def generate(self, text: str) -> str:
        ids, non_special_mask, non_special_ids = self._tokenize_for_model(text)

        substitutions = self._get_edits_num(len(non_special_ids), len(non_special_ids))
        if substitutions == 0:
            return text
        if substitutions > len(non_special_ids):
            raise ValueError("Too many substitutions")

        return self._substitute(ids, non_special_mask, non_special_ids, substitutions)

    def _substitute(
        self,
        ids: pt.Tensor,
        non_special_mask: pt.Tensor,
        non_special_ids: pt.Tensor,
        substitutions: int,
    ) -> str:
        # pylint: disable=too-many-arguments

        indexes_to_substitute = self.rng.choice(
            len(non_special_ids), substitutions, replace=False
        )
        return self._substitute_at_indexes(
            ids, non_special_mask, non_special_ids, indexes_to_substitute
        )

    def _substitute_at_indexes(
        self,
        ids: pt.Tensor,
        non_special_mask: pt.Tensor,
        non_special_ids: pt.Tensor,
        indexes: np.array,
    ) -> str:
        # pylint: disable=too-many-arguments

        indexes = pt.from_numpy(indexes)

        # masked_non_special_ids.shape = [len(non_special_ids)]
        masked_non_special_ids = non_special_ids.scatter(
            0, indexes, self.tokenizer.mask_token_id
        )
        # masked_ids.shape = [splits, max_len]
        masked_ids = ids.masked_scatter(non_special_mask, masked_non_special_ids)
        _, new_non_special_ids = self._predict_masks_at_indexes(
            indexes, masked_ids, non_special_mask, non_special_ids[indexes]
        )

        return self.tokenizer.decode(new_non_special_ids.tolist())
