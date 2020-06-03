from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

import math

import torch as pt
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.tokenization_utils import BatchEncoding, TokenSpan


def mask_at_indexes(
    tokenizer: PreTrainedTokenizer,
    ids: pt.LongTensor,
    tokens_mask: pt.BoolTensor,
    indexes: pt.LongTensor,
) -> pt.LongTensor:
    masked_token_ids = ids.masked_select(tokens_mask)
    masked_token_ids[indexes] = tokenizer.mask_token_id
    masked_ids = ids.masked_scatter(tokens_mask, masked_token_ids)
    masked_ids = cast(pt.LongTensor, masked_ids)
    return masked_ids


def logits_at_indexes(
    model: PreTrainedModel,
    ids: pt.LongTensor,
    attention_mask: pt.LongTensor,
    tokens_mask: pt.BoolTensor,
    indexes: pt.LongTensor,
    dry_run: bool = False,
) -> pt.FloatTensor:
    # pylint: disable=too-many-arguments
    if dry_run:
        vocab_size = model.config.vocab_size
        num_indexes = indexes.size(0)
        unif_log_prob = math.log(1.0) - math.log(vocab_size)
        output_logits = pt.full((num_indexes, vocab_size), unif_log_prob)
    else:
        with pt.no_grad():
            ids = cast(pt.LongTensor, ids.to(model.device))
            attention_mask = cast(pt.LongTensor, attention_mask.to(model.device))
            logits = model(ids, attention_mask=attention_mask)[0]
        output_logits = logits[tokens_mask][indexes]
    output_logits = cast(pt.FloatTensor, output_logits)
    return output_logits


def chars(encoding: BatchEncoding) -> List[TokenSpan]:
    num_tokens = len(encoding["input_ids"])
    tokens: List[TokenSpan] = []
    i = 0
    while i < num_tokens:
        start, end = i, i + 1
        char_span = encoding.token_to_chars(start)
        while end < num_tokens and char_span == encoding.token_to_chars(end):
            # Some tokenizers map subsequent tokens to the same char span
            # (e.g. "ô" in the roberta tokenizer).
            end += 1
        tokens.append(TokenSpan(start, end))
        i = end
    return tokens


def decode_ids(tokenizer: PreTrainedTokenizer, token_ids: List[int]) -> str:
    tokens = [tokenizer.pad_token_id, *token_ids]
    offset = len(tokenizer.pad_token)
    text = tokenizer.decode(tokens)
    return text[offset:]
