from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import
from abc import abstractmethod

import functools
import os

import numpy as np
import torch as pt
from transformers import AutoModelWithLMHead, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer

from ...abc.base import RandomTextEditsGenerator
from ......utils.cache import KeyedSingletonLoader
from ......utils.misc import resolve_optional, resolve_value_or_callable


class MaskedLMRandomTextEditsGenerator(RandomTextEditsGenerator):
    models_cache = KeyedSingletonLoader()
    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        model_name: str = "bert-base-cased",
        sequential_masking: bool = False,
        filter_best_k: Optional[Union[int, Callable[[int], int]]] = None,
        filter_worst_k: Optional[Union[int, Callable[[int], int]]] = None,
        device: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
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
    def generate(self, text: str) -> str:
        ...  # pragma: no cover

    def _tokenize(self, text: str) -> pt.Tensor:
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=False, return_tensors="pt",
        )

        # non_special_ids.shape = [?]
        non_special_ids = encoding["input_ids"].to(dtype=pt.long)[0]

        return non_special_ids

    def _tokenize_for_model(self, text: str) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.tokenizer.max_len,
            pad_to_max_length=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )

        # ids.shape = [splits, max_len]
        ids = encoding["input_ids"].to(dtype=pt.long)
        # non_special_mask.shape = [splits, max_len]
        non_special_mask = encoding["special_tokens_mask"] == 0
        # non_special_ids.shape = [?]
        non_special_ids = ids[non_special_mask]

        return ids, non_special_mask, non_special_ids

    def _predict_masks_at_indexes(
        self,
        masked_indexes: pt.Tensor,
        masked_ids: pt.Tensor,
        non_special_mask: pt.Tensor,
        orig_ids_at_masked_indexes: pt.Tensor,
    ) -> Tuple[pt.Tensor, pt.Tensor]:
        # pylint: disable=too-many-locals,too-many-arguments
        pred_ids = masked_ids.to(self.model.device, copy=True)
        pred_non_special_ids = pred_ids[non_special_mask]

        indexes_iterator = zip(
            masked_indexes.unsqueeze(-1 if self.sequential_masking else 0),
            orig_ids_at_masked_indexes.unsqueeze(-1 if self.sequential_masking else 0),
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
                masks_logits.scatter_(1, best_k_ids, -1e10)
                del best_k_ids
            if filter_worst_k > 0:
                # Filter worst k predictions
                masks_logits.scatter_(1, worst_k_ids, -1e10)
                del worst_k_ids
            # Filter original ids
            masks_logits.scatter_(1, orig_ids_at_indexes.unsqueeze(1), -1e10)
            # masks_probs.shape = [tokens_to_replace, vocab_size]
            masks_probs = masks_logits.softmax(1)
            del masks_logits

            masks_pred_ids = np.empty(masks_probs.size(0), dtype=np.long)
            for i, mask_prob in enumerate(masks_probs):
                token_id = self.rng.choice(mask_prob.size(0), 1, p=mask_prob)
                masks_pred_ids[i] = token_id
            del masks_probs
            # masks_pred_ids.shape = [tokens_to_replace]
            masks_pred_ids = pt.from_numpy(masks_pred_ids)

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