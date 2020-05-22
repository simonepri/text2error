from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

from .abc.tokenizer import MaskedLMRandomTextEditsGeneratorWithTokenizer
from ....edit import TextEdit
from .....utils.transformers import chars, decode_ids


class DuplicateRandomMLMToken(MaskedLMRandomTextEditsGeneratorWithTokenizer):
    # pylint: disable=too-few-public-methods

    def generate(self, text: str) -> List[TextEdit]:
        # pylint: disable=too-many-locals
        encoding = self._simple_encode(text)
        token_spans = chars(encoding)
        num_char_spans = len(token_spans)

        duplications = self._get_edits_num(num_char_spans, None)
        if duplications == 0:
            return []

        char_span_indexes = self.rng.choices(range(num_char_spans), k=duplications)
        char_span_indexes.sort()

        ids = encoding["input_ids"]

        edits = []
        offset = 0
        for char_span_index in char_span_indexes:
            token_span = token_spans[char_span_index]
            start_token, end_token = token_span.start, token_span.end

            chars_span = encoding.token_to_chars(start_token)
            start = chars_span.end

            new_text = decode_ids(self.tokenizer, ids[start_token:end_token])

            edits.append(TextEdit(new_text, start=start + offset))
            offset += len(new_text)
        return edits
