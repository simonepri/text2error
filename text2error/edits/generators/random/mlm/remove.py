from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

from .abc.base import MaskedLMRandomTextEditsGenerator
from ....edit import TextEdit


class RemoveRandomMLMToken(MaskedLMRandomTextEditsGenerator):
    # pylint: disable=too-few-public-methods

    def generate(self, text: str) -> List[TextEdit]:
        encoding = self._encode(text)
        token_ids = encoding["input_ids"]
        num_tokens = len(token_ids)

        remotions = self._get_edits_num(num_tokens, num_tokens)
        if remotions == 0:
            return []
        if remotions > num_tokens:
            raise ValueError("Too many remotions")

        indexes = self.rng.choice(num_tokens, remotions, replace=False)
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
