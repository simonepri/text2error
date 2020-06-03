from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

import random
from array import array

import numpy as np
import torch as pt
from transformers.tokenization_utils import TokenSpan

from .abc.model import MaskedLMRandomTextEditsGeneratorWithModel
from ....edit import TextEdit
from .....utils.transformers import chars, decode_ids, logits_at_indexes

from .....utils.misc import resolve_optional


class InsertRandomMLMToken(MaskedLMRandomTextEditsGeneratorWithModel):
    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        model_name: str = "bert-base-cased",
        device: Optional[str] = None,
        candidate_selector: Optional[Callable[[pt.Tensor], int]] = None,
        bypass_model: bool = False,
        rng: Optional[random.Random] = None,
        edits_num: Optional[Union[int, Callable[[int, Optional[int]], int]]] = None,
    ) -> None:
        # pylint: disable=too-many-arguments
        super().__init__(model_name, device, rng, edits_num)

        self.bypass_model = bypass_model
        self.candidate_selector: Callable[[pt.Tensor], int] = resolve_optional(
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
        # Add a fake TokenSpan at the end so can also insert after the last char span.
        token_spans.append(TokenSpan(token_spans[-1].end, token_spans[-1].end))
        num_char_spans = len(token_spans)

        insertions = self._get_edits_num(num_char_spans, None)
        if insertions == 0:
            return []

        char_span_indexes = self.rng.choices(range(num_char_spans), k=insertions)
        char_span_indexes.sort()
        token_indexes = list(map(lambda i: token_spans[i].start, char_span_indexes))

        token_ids = encoding["input_ids"] if num_char_spans > 0 else array("l")
        masked_token_ids = np.insert(
            token_ids, token_indexes, self.tokenizer.mask_token_id
        )

        # TODO: Avoid roundtrip decode-encode.
        new_text = self.tokenizer.decode(masked_token_ids.tolist())
        new_model_encoding = self._model_encode(new_text)

        masked_ids = new_model_encoding["input_ids"]
        attention_mask = new_model_encoding["attention_mask"]
        tokens_mask = new_model_encoding["special_tokens_mask"] == 0
        masks_indexes = cast(
            pt.LongTensor,
            pt.from_numpy(np.array(token_indexes) + np.arange(len(token_indexes))),
        )
        masks_logits = logits_at_indexes(
            self.model,
            masked_ids,
            attention_mask,
            tokens_mask,
            masks_indexes,
            dry_run=self.bypass_model,
        )
        masks_logits[:, self.tokenizer.all_special_ids] = float("-inf")
        masks_log_probs = masks_logits.log_softmax(-1).cpu()

        predictions = []
        for mask_log_probs in masks_log_probs:
            token_id = self.candidate_selector(mask_log_probs)
            predictions.append(token_id)

        edits = []
        offset = 0
        for i, char_span_index in enumerate(char_span_indexes):
            if char_span_index == 0:
                start = 0
            else:
                token_span = token_spans[char_span_index - 1]
                start_token = token_span.start

                chars_span = encoding.token_to_chars(start_token)
                start = chars_span.end

            new_text = decode_ids(self.tokenizer, predictions[i : i + 1])
            if start + offset == 0:
                # If we are at the beginning of a sentence.
                if len(new_text) > 1 and new_text[0] == " ":
                    # Avoid inserting the space before the token if present.
                    new_text = new_text[1:]
                if new_text[-1] != " ":
                    # Add a space after the token if not present.
                    new_text = new_text + " "

            edits.append(TextEdit(new_text, start=start + offset))
            offset += len(new_text)

        return edits

    @staticmethod
    def _default_candidate_selector(tokens_log_prob: pt.Tensor) -> int:
        # NB: tokens_log_prob should contain at least one non-inf element.
        filter_mask = tokens_log_prob != float("-inf")
        filtered_ids = filter_mask.nonzero().squeeze(-1)
        filtered_logits = tokens_log_prob[filtered_ids]
        # We sample from the token distribution directly in log space.
        # NB: There is no need to normalize the logits.
        # See: Gumbel-max trick (https://w.wiki/S4s).
        gumbel_sampler = pt.distributions.Gumbel(loc=0, scale=1)  # type: ignore
        gumbel_draws = gumbel_sampler.sample(filtered_logits.shape)
        return filtered_ids[(filtered_logits + gumbel_draws).argmax(0)]
