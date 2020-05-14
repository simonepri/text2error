from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

from array import array

import numpy as np
import torch as pt
from transformers.tokenization_utils import BatchEncoding

from .abc.base import MaskedLMRandomTextEditsGenerator
from ....edit import TextEdit


class InsertRandomMLMToken(MaskedLMRandomTextEditsGenerator):
    # pylint: disable=too-few-public-methods

    def generate(self, text: str) -> List[TextEdit]:
        # pylint: disable=too-many-locals,invalid-name

        encoding = self._encode(text)
        token_ids = encoding["input_ids"]
        indexes = self._get_possible_indexes(encoding)
        num_pos = len(indexes)

        insertions = self._get_edits_num(num_pos, None)
        if insertions == 0:
            return []

        indexes = np.array(self.rng.choices(indexes, k=insertions))
        indexes.sort()

        token_ids = token_ids if len(token_ids) > 0 else array("l")
        masked_token_ids = np.insert(token_ids, indexes, self.tokenizer.mask_token_id)
        pt_masked_token_ids = pt.from_numpy(masked_token_ids)
        # TODO: Avoid roundtrip decoding and encoding.
        new_text = self.tokenizer.decode(pt_masked_token_ids.tolist())
        new_encoding = self._encode_for_model(new_text)
        masked_ids = new_encoding["input_ids"]
        token_mask = new_encoding["special_tokens_mask"] == 0
        mask_indexes = indexes + np.arange(len(indexes))
        pt_mask_indexes = pt.from_numpy(mask_indexes)
        _, new_token_ids = self._predict_masks_at_indexes(
            pt_mask_indexes, masked_ids, token_mask, pt_masked_token_ids[mask_indexes],
        )
        predictions = new_token_ids[mask_indexes].tolist()

        edits = []
        offset = 0
        for pi, i in enumerate(indexes):
            start = 0 if i == 0 else encoding.token_to_chars(i - 1).end

            new_text = self._ids_to_string([predictions[pi]])
            if start + offset == 0:
                # If we are at the beginning of a sentence.
                if len(new_text) > 1 and new_text[0] == " ":
                    # Avoid inserting the space before the token if present.
                    new_text = new_text[1:]
                if new_text[-1] != " ":
                    # Add a space after the token if not present.
                    new_text = new_text + " "

            edits.append(TextEdit(new_text, start=start + offset))
            offset += len(new_text)

        return edits

    def _get_possible_indexes(self, encoding: BatchEncoding) -> List[int]:
        num_tok = len(encoding["input_ids"])
        indexes = [0]
        i = self._get_next_char_span_index(encoding, 0)
        while i < num_tok:
            indexes.append(i)
            i = self._get_next_char_span_index(encoding, i)
        indexes.append(i)  # Index after the last token.
        return indexes
