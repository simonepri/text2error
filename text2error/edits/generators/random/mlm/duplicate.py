from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

from .abc.base import MaskedLMRandomTextEditsGenerator
from ....edit import TextEdit


class DuplicateRandomMLMToken(MaskedLMRandomTextEditsGenerator):
    # pylint: disable=too-few-public-methods

    def generate(self, text: str) -> List[TextEdit]:
        encoding = self._encode(text)
        token_ids = encoding["input_ids"]
        num_tokens = len(token_ids)

        duplications = self._get_edits_num(num_tokens, num_tokens)
        if duplications == 0:
            return []
        if duplications > num_tokens:
            raise ValueError("Too many duplications")

        indexes = self.rng.sample(range(num_tokens), k=duplications)
        indexes.sort()

        edits = []
        offset = 0
        for i in indexes:
            word_span = encoding.token_to_chars(i)

            start = word_span.end
            new_text = self._id_to_string(token_ids[i])

            edits.append(TextEdit(new_text, start=start + offset))
            offset += len(new_text)
        return edits
