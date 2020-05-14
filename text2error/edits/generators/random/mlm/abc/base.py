from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import
from abc import abstractmethod

import random
import functools
import os

import torch as pt
from transformers import AutoModelWithLMHead, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.tokenization_utils import BatchEncoding

from ...abc.base import RandomTextEditsGenerator
from .....edit import TextEdit
from ......utils.cache import KeyedSingletonLoader
from ......utils.misc import resolve_optional, resolve_value_or_callable


class MaskedLMRandomTextEditsGenerator(RandomTextEditsGenerator):
    # pylint: disable=too-few-public-methods
    models_cache = KeyedSingletonLoader()

    def __init__(
        self,
        model_name: str = "bert-base-cased",
        sequential_masking: bool = False,
        filter_best_k: Optional[Union[int, Callable[[int], int]]] = None,
        filter_worst_k: Optional[Union[int, Callable[[int], int]]] = None,
        device: Optional[str] = None,
        rng: Optional[random.Random] = None,
        edits_num: Optional[Union[int, Callable[[int, Optional[int]], int]]] = None,
    ) -> None:
        # pylint: disable=too-many-arguments
        super().__init__(rng, edits_num)

        self.model_name = model_name
        self.sequential_masking = sequential_masking
        self.filter_best_k: Union[int, Callable[[int], int]] = resolve_optional(
            filter_best_k, 0
        )
        self.filter_worst_k: Union[int, Callable[[int], int]] = resolve_optional(
            filter_worst_k, 0
        )
        self.model, self.tokenizer = self.__load_model(model_name, device=device)

    def __del__(self) -> None:
        self.__unload_model(self.model_name)

    @abstractmethod
    def generate(self, text: str) -> List[TextEdit]:
        ...  # pragma: no cover

    def _ids_to_string(self, token_ids: List[int]) -> str:
        tokens = [self.tokenizer.pad_token_id, *token_ids]
        offset = len(self.tokenizer.pad_token)
        text = self.tokenizer.decode(tokens)
        return text[offset:]

    def _encode(self, text: str) -> BatchEncoding:
        encoding = self.tokenizer.encode_plus(text, add_special_tokens=False)
        return encoding

    def _encode_for_model(self, text: str) -> BatchEncoding:
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.tokenizer.max_len,
            pad_to_max_length=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        return encoding

    @classmethod
    def _get_next_char_span_index(cls, encoding: BatchEncoding, index: int) -> int:
        num_tokens = len(encoding["input_ids"])
        last_span = encoding.token_to_chars(index)
        index += 1
        while index < num_tokens:
            next_span = encoding.token_to_chars(index)
            if next_span.start < last_span.end:
                # Some subsequent tokens get mapped to the same char span.
                last_span = next_span
                index += 1
                continue
            return index
        return index

    def _predict_masks_at_indexes(
        self,
        masked_indexes: pt.Tensor,
        masked_ids: pt.Tensor,
        non_special_mask: pt.Tensor,
        orig_ids_at_masked_indexes: pt.Tensor,
    ) -> Tuple[pt.Tensor, pt.Tensor]:
        # pylint: disable=too-many-locals,too-many-arguments
        masked_indexes = masked_indexes.to(self.model.device)
        pred_ids = masked_ids.to(self.model.device, copy=True)
        non_special_mask = non_special_mask.to(self.model.device)
        orig_ids_at_masked_indexes = orig_ids_at_masked_indexes.to(self.model.device)
        pred_non_special_ids = pred_ids[non_special_mask]

        indexes_iterator = zip(
            masked_indexes.unsqueeze(-1 if self.sequential_masking else 0),
            orig_ids_at_masked_indexes.unsqueeze(-1 if self.sequential_masking else 0),
        )
        # pylint: disable=not-callable
        special_ids = pt.tensor(
            self.tokenizer.all_special_ids, device=self.model.device
        )
        for indexes, orig_ids_at_indexes in indexes_iterator:
            with pt.no_grad():
                logits = self.model(pred_ids)[0]

            filter_best_k = resolve_value_or_callable(
                self.filter_best_k, self.tokenizer.vocab_size
            )  # type: int
            filter_worst_k = resolve_value_or_callable(
                self.filter_worst_k, self.tokenizer.vocab_size
            )  # type: int

            # masks_logits.shape = [tokens_to_replace, vocab_size]
            masks_logits = logits[non_special_mask][indexes].double()
            del logits
            if filter_best_k > 0:
                best_k_ids = masks_logits.topk(
                    filter_best_k, 1, largest=True, sorted=False
                )[1]
            if filter_worst_k > 0:
                worst_k_ids = masks_logits.topk(
                    filter_worst_k, 1, largest=False, sorted=False
                )[1]
            if filter_best_k > 0:
                # Filter best k predictions
                masks_logits.scatter_(1, best_k_ids, float("-inf"))
                del best_k_ids
            if filter_worst_k > 0:
                # Filter worst k predictions
                masks_logits.scatter_(1, worst_k_ids, float("-inf"))
                del worst_k_ids
            # Filter original ids
            masks_logits.scatter_(1, orig_ids_at_indexes.unsqueeze(1), float("-inf"))
            # Filter special ids
            masks_logits.index_fill_(1, special_ids, float("-inf"))
            # masks_probs.shape = [tokens_to_replace, vocab_size]
            masks_probs = masks_logits.softmax(1)
            del masks_logits

            masks_probs = masks_probs.cpu()
            # masks_pred_ids.shape = [tokens_to_replace]
            masks_pred_ids = pt.empty(masks_probs.size(0), dtype=pt.long)
            for i, mask_probs in enumerate(masks_probs):
                tokens = mask_probs.nonzero(as_tuple=True)[0]
                cweights = mask_probs[tokens].cumsum(0)
                tokens, cweights = tokens.tolist(), cweights.tolist()
                token_id = self.rng.choices(tokens, k=1, cum_weights=cweights)[0]
                masks_pred_ids[i] = token_id
            del masks_probs
            masks_pred_ids = masks_pred_ids.to(self.model.device)

            # Update masks with predictions
            pred_non_special_ids.scatter_(0, indexes, masks_pred_ids)
            del masks_pred_ids
            pred_ids.masked_scatter_(non_special_mask, pred_non_special_ids)

        return pred_ids, pred_non_special_ids

    @classmethod
    def __load_model(
        cls, model_name: str, **kwargs: Any
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        return cls.models_cache.load(
            model_name, functools.partial(cls.__load_transformers_model, **kwargs)
        )

    @classmethod
    def __unload_model(cls, model_name: str, **kwargs: Any) -> None:
        # pylint: disable=unused-argument
        cls.models_cache.unload(model_name)

    @classmethod
    def __load_transformers_model(
        cls, model_name: str, **kwargs: Any
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        cache_dir = os.environ.get("TRANSFORMERS_CACHE_DIR", ".transformers_cache")
        cache_dir = kwargs.get("cache_dir", cache_dir)
        model = AutoModelWithLMHead.from_pretrained(model_name, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, use_fast=True, add_special_tokens=True
        )
        if not model.__class__.__name__.endswith("ForMaskedLM"):
            raise ValueError("The model provided is not a Masked LM: %s" % model_name)
        if "device" in kwargs:
            model.to(kwargs.get("device"))
        model.eval()

        return model, tokenizer
