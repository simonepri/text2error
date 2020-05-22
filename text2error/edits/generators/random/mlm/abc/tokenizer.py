from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import
from abc import abstractmethod

import random
import functools
import os

from transformers import AutoTokenizer
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils import BatchEncoding

from .base import MaskedLMRandomTextEditsGenerator
from .....edit import TextEdit
from ......utils.cache import KeyedSingletonLoader


class MaskedLMRandomTextEditsGeneratorWithTokenizer(MaskedLMRandomTextEditsGenerator):
    # pylint: disable=too-few-public-methods
    tokenizers_cache = KeyedSingletonLoader()

    def __init__(
        self,
        model_name: str = "bert-base-cased",
        rng: Optional[random.Random] = None,
        edits_num: Optional[Union[int, Callable[[int, Optional[int]], int]]] = None,
    ) -> None:
        # pylint: disable=too-many-arguments
        super().__init__(model_name, rng, edits_num)

        self.tokenizer = self.__load_tokenizer(self.model_name)

    def __del__(self) -> None:
        super().__del__()
        self.__unload_tokenizer(self.model_name)

    @abstractmethod
    def generate(self, text: str) -> List[TextEdit]:
        ...  # pragma: no cover

    def _simple_encode(self, text: str) -> BatchEncoding:
        encoding = self.tokenizer.encode_plus(text, add_special_tokens=False)
        return encoding

    @classmethod
    def __load_tokenizer(cls, model_name: str) -> PreTrainedTokenizer:
        key = model_name
        tokenizer_provider = functools.partial(
            cls.__load_transformers_tokenizer, model_name
        )
        return cls.tokenizers_cache.load(key, tokenizer_provider)

    @classmethod
    def __unload_tokenizer(cls, model_name: str) -> None:
        key = model_name
        cls.tokenizers_cache.unload(key)

    @classmethod
    def __load_transformers_tokenizer(cls, model_name: str) -> PreTrainedTokenizer:
        cache_dir = os.environ.get("TRANSFORMERS_CACHE_DIR", ".transformers_cache")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, use_fast=True
        )
        return tokenizer
