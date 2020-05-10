from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

import torch as pt

from .abc.base import MaskedLMRandomTextEditsGenerator
from ....edit import TextEdit


class SubstituteRandomMLMToken(MaskedLMRandomTextEditsGenerator):
    # pylint: disable=too-few-public-methods

    def generate(self, text: str) -> List[TextEdit]:
        # pylint: disable=too-many-locals,invalid-name

        encoding = self._encode(text)
        token_ids = encoding["input_ids"]
        num_tokens = len(token_ids)

        substitutions = self._get_edits_num(num_tokens, num_tokens)
        if substitutions == 0:
            return []
        if substitutions > num_tokens:
            raise ValueError("Too many substitutions")

        indexes = self.rng.sample(range(num_tokens), k=substitutions)
        indexes.sort()

        model_encoding = self._encode_for_model(text)
        ids = model_encoding["input_ids"]
        token_mask = model_encoding["special_tokens_mask"] == 0
        # pylint: disable=not-callable
        pt_token_ids = pt.tensor(token_ids)
        pt_indexes = pt.tensor(indexes)
        masked_ids = pt_token_ids.scatter(0, pt_indexes, self.tokenizer.mask_token_id)
        masked_ids = ids.masked_scatter(token_mask, masked_ids)
        _, new_token_ids = self._predict_masks_at_indexes(
            pt_indexes, masked_ids, token_mask, pt_token_ids[indexes]
        )
        predictions = new_token_ids[indexes]

        edits = []
        offset = 0
        for pi, i in enumerate(indexes):
            word_span = encoding.token_to_chars(i)

            start = word_span.start
            end = word_span.end
            if start > 0 and text[start - 1] == " ":
                # Remove the space before the token if present.
                start -= 1

            new_text = self._id_to_string(int(predictions[pi].item()))
            if start + offset == 0:
                # If we are at the beginning of a sentence.
                if new_text[0] == " ":
                    # Avoid inserting the space before the token if present.
                    new_text = new_text[1:]

            edits.append(TextEdit(new_text, start=start + offset, end=end + offset))
            offset += len(new_text) - (end - start)

        return edits
