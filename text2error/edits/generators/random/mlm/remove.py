from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

from .abc.tokenizer import MaskedLMRandomTextEditsGeneratorWithTokenizer
from ....edit import TextEdit
from .....utils.transformers import chars


class RemoveRandomMLMToken(MaskedLMRandomTextEditsGeneratorWithTokenizer):
    # pylint: disable=too-few-public-methods

    def generate(self, text: str) -> List[TextEdit]:
        encoding = self._simple_encode(text)
        token_spans = chars(encoding)
        num_char_spans = len(token_spans)

        remotions = self._get_edits_num(num_char_spans, num_char_spans)
        if remotions == 0:
            return []
        if remotions > num_char_spans:
            raise ValueError("Too many remotions")

        char_span_indexes = self.rng.sample(range(num_char_spans), k=remotions)
        char_span_indexes.sort()

        edits = []
        offset = 0
        for char_span_index in char_span_indexes:
            token_span = token_spans[char_span_index]
            start_token = token_span.start

            chars_span = encoding.token_to_chars(start_token)
            start, end = chars_span.start, chars_span.end

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
