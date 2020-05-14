from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

from transformers.tokenization_utils import BatchEncoding

from .abc.base import MaskedLMRandomTextEditsGenerator
from ....edit import TextEdit


class RemoveRandomMLMToken(MaskedLMRandomTextEditsGenerator):
    # pylint: disable=too-few-public-methods

    def generate(self, text: str) -> List[TextEdit]:
        encoding = self._encode(text)
        indexes = self._get_possible_indexes(encoding)
        num_pos = len(indexes)

        remotions = self._get_edits_num(num_pos, num_pos)
        if remotions == 0:
            return []
        if remotions > num_pos:
            raise ValueError("Too many remotions")

        indexes = self.rng.sample(indexes, k=remotions)
        indexes.sort()

        edits = []
        offset = 0
        for i in indexes:
            word_span = encoding.token_to_chars(i)
            start, end = word_span.start, word_span.end

            if start + offset == 0:
                # If we are at the beginning of the text.
                if end != len(text) and text[end] == " ":
                    # Remove the space after the token if present.
                    end += 1
            else:
                # Otherwise
                if start > 0 and text[start - 1] == " ":
                    # Remove the space before the token if present.
                    start -= 1

            edits.append(TextEdit("", start=start + offset, end=end + offset))
            offset -= end - start
        return edits

    def _get_possible_indexes(self, encoding: BatchEncoding) -> List[int]:
        num_tok = len(encoding["input_ids"])
        indexes = [0]
        i = self._get_next_char_span_index(encoding, 0)
        while i < num_tok:
            indexes.append(i)
            i = self._get_next_char_span_index(encoding, i)
        return indexes
