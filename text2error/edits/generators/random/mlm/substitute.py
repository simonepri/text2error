from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

import random

import torch as pt

from .abc.model import MaskedLMRandomTextEditsGeneratorWithModel
from ....edit import TextEdit
from .....utils.transformers import (
    chars,
    decode_ids,
    logits_at_indexes,
    mask_at_indexes,
)

from .....utils.misc import resolve_optional


class SubstituteRandomMLMToken(MaskedLMRandomTextEditsGeneratorWithModel):
    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        model_name: str = "bert-base-cased",
        device: Optional[str] = None,
        candidate_selector: Optional[Callable[[pt.Tensor, int], int]] = None,
        rng: Optional[random.Random] = None,
        edits_num: Optional[Union[int, Callable[[int, Optional[int]], int]]] = None,
    ) -> None:
        # pylint: disable=too-many-arguments
        super().__init__(model_name, device, rng, edits_num)

        self.candidate_selector: Callable[[pt.Tensor, int], int] = resolve_optional(
            candidate_selector, self._default_candidate_selector
        )

    def generate(self, text: str) -> List[TextEdit]:
        # pylint: disable=too-many-locals
        encoding = self._simple_encode(text)

        if len(encoding["input_ids"]) > self.tokenizer.max_len:
            #  If the string is too long for the model only process the first chuck.
            # The -4 is to leave space for special tokens.
            end = encoding.token_to_chars(self.tokenizer.max_len - 4).end
            return self.generate(text[:end])

        token_spans = chars(encoding)
        num_char_spans = len(token_spans)

        substitutions = self._get_edits_num(num_char_spans, num_char_spans)
        if substitutions == 0:
            return []
        if substitutions > num_char_spans:
            raise ValueError("Too many substitutions")

        char_span_indexes = self.rng.sample(range(num_char_spans), k=substitutions)
        char_span_indexes.sort()
        token_indexes = list(map(lambda i: token_spans[i].start, char_span_indexes))
        # We mask all the tokens of a given char span to avoid to bias the model
        # at prediction time. This is useful if the text contains characters
        # like "ô" and we are using a tokenizer like the roberta tokenizer.
        ext_token_indexes: List[int] = []
        for i in char_span_indexes:
            ext_token_indexes.extend(range(token_spans[i].start, token_spans[i].end))

        model_encoding = self._model_encode(text)
        ids = model_encoding["input_ids"]
        tokens_mask = model_encoding["special_tokens_mask"] == 0
        attention_mask = model_encoding["attention_mask"]
        # pylint: disable=not-callable
        masks_indexes = cast(pt.LongTensor, pt.tensor(token_indexes))
        # pylint: disable=not-callable
        ext_mask_indexes = cast(pt.LongTensor, pt.tensor(ext_token_indexes))
        masked_ids = mask_at_indexes(self.tokenizer, ids, tokens_mask, ext_mask_indexes)
        masks_logits = logits_at_indexes(
            self.model, masked_ids, attention_mask, tokens_mask, masks_indexes
        )
        masks_logits[:, self.tokenizer.all_special_ids] = float("-inf")
        masks_log_probs = masks_logits.log_softmax(-1).cpu()

        predictions = []
        original_ids = ids[tokens_mask][masks_indexes].tolist()
        for mask_log_probs, original_id in zip(masks_log_probs, original_ids):
            token_id = self.candidate_selector(mask_log_probs, original_id)
            predictions.append(token_id)

        edits = []
        offset = 0
        for i, char_span_index in enumerate(char_span_indexes):
            token_span = token_spans[char_span_index]
            start_token = token_span.start

            chars_span = encoding.token_to_chars(start_token)
            start, end = chars_span.start, chars_span.end

            if start > 0 and text[start - 1] == " ":
                # Remove the space before the token if present.
                start -= 1

            new_text = decode_ids(self.tokenizer, predictions[i : i + 1])
            if start + offset == 0:
                # If we are at the beginning of a sentence.
                if new_text[0] == " ":
                    # Avoid inserting the space before the token if present.
                    new_text = new_text[1:]

            edits.append(TextEdit(new_text, start=start + offset, end=end + offset))
            offset += len(new_text) - (end - start)

        return edits

    @staticmethod
    def _default_candidate_selector(
        tokens_log_prob: pt.Tensor, original_id: int
    ) -> int:
        # NB: tokens_log_prob should contain at least one non-inf element.
        filter_mask = tokens_log_prob != float("-inf")
        filter_mask[original_id] = False
        filtered_ids = filter_mask.nonzero().squeeze(-1)
        filtered_logits = tokens_log_prob[filtered_ids]
        # We sample from the token distribution directly in log space.
        # NB: There is no need to normalize the logits.
        # See: Gumbel-max trick (https://w.wiki/S4s).
        gumbel_sampler = pt.distributions.Gumbel(loc=0, scale=1)  # type: ignore
        gumbel_draws = gumbel_sampler.sample(filtered_logits.shape)
        id = filtered_ids[(filtered_logits + gumbel_draws).argmax(0)]
        return id
