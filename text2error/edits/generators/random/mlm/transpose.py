from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

from .abc.base import MaskedLMRandomTextEditsGenerator
from ....edit import TextEdit
from .....utils.random import non_adjacent_choice


class TransposeRandomMLMToken(MaskedLMRandomTextEditsGenerator):
    # pylint: disable=too-few-public-methods

    def generate(self, text: str) -> List[TextEdit]:
        encoding = self._encode(text)
        token_ids = encoding["input_ids"]
        num_tokens = len(token_ids)

        transpositions = self._get_edits_num(num_tokens, num_tokens // 2)
        if transpositions == 0:
            return []
        if transpositions > num_tokens // 2:
            raise ValueError("Too many transpositions")

        indexes = non_adjacent_choice(num_tokens - 1, transpositions, rng=self.rng)
        # The indexes from non_adjacent_choice are already sorted.

        edits = []
        offset = 0
        for i in indexes:
            word_span_1 = encoding.token_to_chars(i)
            word_span_2 = encoding.token_to_chars(i + 1)
            start, end = word_span_1.start, word_span_2.end
            if start > 0 and text[start - 1] == " ":
                # Remove the space before the first token if present.
                start -= 1

            new_text = self._id_to_string(token_ids[i + 1]) + self._id_to_string(
                token_ids[i]
            )
            if start + offset == 0:
                # If we are the beginning of the text.
                if new_text[0] == " ":
                    # Avoid inserting the space before the token if present.
                    new_text = new_text[1:]

            edits.append(TextEdit(new_text, start=start + offset, end=end + offset))
            offset += len(new_text) - (end - start)
        return edits
