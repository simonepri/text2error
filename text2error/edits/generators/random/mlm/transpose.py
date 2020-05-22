from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

from .abc.tokenizer import MaskedLMRandomTextEditsGeneratorWithTokenizer
from ....edit import TextEdit
from .....utils.random import non_adjacent_sample
from .....utils.transformers import chars, decode_ids


class TransposeRandomMLMToken(MaskedLMRandomTextEditsGeneratorWithTokenizer):
    # pylint: disable=too-few-public-methods

    def generate(self, text: str) -> List[TextEdit]:
        # pylint: disable=too-many-locals
        encoding = self._simple_encode(text)
        token_spans = chars(encoding)
        num_char_spans = len(token_spans)

        transpositions = self._get_edits_num(num_char_spans - 1, num_char_spans // 2)
        if transpositions == 0:
            return []
        if transpositions > num_char_spans // 2:
            raise ValueError("Too many transpositions")

        # The -1 becase the last char span can't be transposed.
        char_span_indexes = non_adjacent_sample(
            num_char_spans - 1, transpositions, rng=self.rng
        )
        # The output of non_adjacent_sample is already sorted.

        ids = encoding["input_ids"]

        edits = []
        offset = 0
        for char_span_index in char_span_indexes:
            token_span_1 = token_spans[char_span_index]
            token_span_2 = token_spans[char_span_index + 1]
            start_token_1, end_token_1 = token_span_1.start, token_span_1.end
            start_token_2, end_token_2 = token_span_2.start, token_span_2.end

            chars_span_1 = encoding.token_to_chars(start_token_1)
            chars_span_2 = encoding.token_to_chars(start_token_2)
            start, end = chars_span_1.start, chars_span_2.end

            if start > 0 and text[start - 1] == " ":
                # Remove the space before the first token if present.
                start -= 1

            new_text = decode_ids(self.tokenizer, ids[start_token_2:end_token_2])
            new_text += decode_ids(self.tokenizer, ids[start_token_1:end_token_1])

            if start + offset == 0:
                # If we are the beginning of the text.
                if new_text[0] == " ":
                    # Avoid inserting the space before the token if present.
                    new_text = new_text[1:]

            edits.append(TextEdit(new_text, start=start + offset, end=end + offset))
            offset += len(new_text) - (end - start)
        return edits
