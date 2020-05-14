from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

from transformers.tokenization_utils import BatchEncoding

from .abc.base import MaskedLMRandomTextEditsGenerator
from ....edit import TextEdit
from .....utils.random import non_adjacent_sample


class TransposeRandomMLMToken(MaskedLMRandomTextEditsGenerator):
    # pylint: disable=too-few-public-methods

    def generate(self, text: str) -> List[TextEdit]:
        # pylint: disable=too-many-locals
        encoding = self._encode(text)
        token_ids = encoding["input_ids"]
        indexes = self._get_possible_indexes(encoding)
        num_pos = len(indexes)

        transpositions = self._get_edits_num(num_pos, (num_pos + 1) // 2)
        if transpositions == 0:
            return []
        if transpositions > (num_pos + 1) // 2:
            raise ValueError("Too many transpositions")

        sampled = non_adjacent_sample(num_pos, transpositions, rng=self.rng)
        indexes = [indexes[i] for i in sampled]
        # The indexes from non_adjacent_sample are already sorted.

        edits = []
        offset = 0
        for i in indexes:
            j = self._get_next_char_span_index(encoding, i)
            k = self._get_next_char_span_index(encoding, j)
            word_span_1 = encoding.token_to_chars(i)
            word_span_2 = encoding.token_to_chars(j)
            start, end = word_span_1.start, word_span_2.end
            if start > 0 and text[start - 1] == " ":
                # Remove the space before the first token if present.
                start -= 1

            new_text = self._ids_to_string(token_ids[j:k]) + self._ids_to_string(
                token_ids[i:j]
            )
            if start + offset == 0:
                # If we are the beginning of the text.
                if new_text[0] == " ":
                    # Avoid inserting the space before the token if present.
                    new_text = new_text[1:]

            edits.append(TextEdit(new_text, start=start + offset, end=end + offset))
            offset += len(new_text) - (end - start)
        return edits

    def _get_possible_indexes(self, encoding: BatchEncoding) -> List[int]:
        num_tok = len(encoding["input_ids"])
        indexes = [0]
        i = self._get_next_char_span_index(encoding, 0)
        while i < num_tok:
            indexes.append(i)
            i = self._get_next_char_span_index(encoding, i)
        indexes.pop()  # The last position can't be selected.
        return indexes
