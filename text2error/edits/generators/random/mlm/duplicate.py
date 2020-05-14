from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

from transformers.tokenization_utils import BatchEncoding

from .abc.base import MaskedLMRandomTextEditsGenerator
from ....edit import TextEdit


class DuplicateRandomMLMToken(MaskedLMRandomTextEditsGenerator):
    # pylint: disable=too-few-public-methods

    def generate(self, text: str) -> List[TextEdit]:
        # pylint: disable=too-many-locals
        encoding = self._encode(text)
        token_ids = encoding["input_ids"]
        indexes = self._get_possible_indexes(encoding)
        num_pos = len(indexes)

        duplications = self._get_edits_num(num_pos, None)
        if duplications == 0:
            return []

        indexes = self.rng.choices(indexes, k=duplications)
        indexes.sort()

        edits = []
        offset = 0
        for i in indexes:
            j = self._get_next_char_span_index(encoding, i)
            word_span = encoding.token_to_chars(i)

            start = word_span.end
            new_text = self._ids_to_string(token_ids[i:j])

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
        return indexes
